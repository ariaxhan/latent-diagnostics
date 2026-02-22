"""
Modal Runner for General Attribution Analysis

Computes attribution graph metrics for any JSON dataset.
Used for mechanistic interpretability research across domains.

Usage:
    # Domain analysis
    modal run scripts/modal_general_attribution.py \\
        --input-file data/domain_analysis/domain_samples.json \\
        --output-file data/results/domain_attribution_metrics.json

    # Truthfulness analysis
    modal run scripts/modal_general_attribution.py \\
        --input-file data/truthfulness/samples.json \\
        --output-file data/results/truthfulness_metrics.json

    # With sample limit (for testing)
    modal run scripts/modal_general_attribution.py \\
        --input-file data/domain_analysis/domain_samples.json \\
        --n-samples 50

Input JSON format:
    {
        "metadata": {...},
        "samples": [
            {"idx": 0, "text": "...", "domain": "...", "label": "..."},
            ...
        ]
    }
"""

import modal
import json
from datetime import datetime

# Image with circuit-tracer
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.20.0",
        "pyyaml>=6.0",
    )
    .pip_install(
        "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git",
    )
)

app = modal.App("general-attribution", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("attribution-results", create_if_missing=True)


@app.function(
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,  # 4 hours
)
def compute_attribution_metrics(
    input_data: dict,
    max_tokens: int = 200,
    save_every: int = 10,
):
    """
    Compute attribution metrics for samples.

    Args:
        input_data: Dict with 'samples' list containing text samples
        max_tokens: Maximum tokens per sample
        save_every: Save checkpoint every N samples

    Returns:
        Dict with samples and their attribution metrics
    """
    import os
    import torch
    from huggingface_hub import login

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[setup] HuggingFace authenticated")

    # Import circuit_tracer
    from circuit_tracer import ReplacementModel, attribute

    print("=" * 70)
    print("GENERAL ATTRIBUTION ANALYSIS")
    print("=" * 70)

    samples = input_data.get("samples", [])
    print(f"\n[1/3] Loaded {len(samples)} samples")

    # Domain distribution
    domains = {}
    for s in samples:
        d = s.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print(f"  Domains: {dict(sorted(domains.items()))}")

    # Initialize model
    print("\n[2/3] Loading model + transcoders...")

    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        backend="transformerlens",
    )
    print("  Model ready!")

    # Compute metrics
    print("\n[3/3] Computing attribution metrics...")

    results = []
    failed = 0
    failed_indices = []
    checkpoint_path = "/results/attribution_checkpoint.json"
    start_idx = 0

    # Resume from checkpoint if exists
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint.get("samples", [])
            start_idx = checkpoint.get("processed", 0)
            if start_idx > 0:
                print(f"  [RESUME] Found {len(results)} results, resuming from sample {start_idx}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(samples)} ({100*(i+1)/len(samples):.1f}%) - ok:{len(results)} fail:{failed}")

        # Save checkpoint + commit to volume
        if (i + 1) % save_every == 0 and results:
            with open(checkpoint_path, 'w') as f:
                json.dump({"samples": results, "processed": i + 1, "failed": failed_indices}, f)
            results_volume.commit()
            print(f"  [SAVE] {len(results)} results persisted")

        try:
            # Truncate text to avoid OOM (~4 chars per token)
            text = sample["text"][:max_tokens * 4]

            # Clear CUDA cache before each sample
            torch.cuda.empty_cache()

            # Get attribution graph - API: attribute(text, model, ...)
            graph = attribute(text, model, verbose=False)

            # Extract metrics (following working modal_pint_benchmark.py pattern)
            metrics = {
                **sample,  # Preserve original fields (idx, domain, label, etc.)
            }

            # Active features
            if hasattr(graph, 'active_features'):
                metrics['n_active'] = len(graph.active_features)

            # Activation values
            if hasattr(graph, 'activation_values'):
                acts = graph.activation_values
                if hasattr(acts, 'abs'):
                    metrics['mean_activation'] = float(acts.abs().mean().item())
                    metrics['max_activation'] = float(acts.abs().max().item())

            # Adjacency matrix (causal influence)
            if hasattr(graph, 'adjacency_matrix'):
                adj = graph.adjacency_matrix
                if hasattr(adj, 'abs'):
                    n_edges = int((adj.abs() > 0.01).sum().item())
                    metrics['n_edges'] = n_edges
                    metrics['mean_influence'] = float(adj.abs().mean().item())
                    metrics['max_influence'] = float(adj.abs().max().item())

                    # Concentration metric
                    flat = adj.abs().flatten()
                    sorted_inf, _ = flat.sort(descending=True)
                    total = flat.sum().item()
                    top_100 = sorted_inf[:100].sum().item()
                    metrics['top_100_concentration'] = float(top_100 / (total + 1e-10))

            # Logit probabilities
            if hasattr(graph, 'logit_probabilities'):
                probs = graph.logit_probabilities
                if hasattr(probs, 'max'):
                    metrics['max_logit_prob'] = float(probs.max().item())
                    metrics['logit_entropy'] = float(-(probs * (probs + 1e-10).log()).sum().item())

            results.append(metrics)

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Sample {sample.get('idx', i)}: text too long")
            failed += 1
            failed_indices.append(i)
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            print(f"  [FAIL] Sample {sample.get('idx', i)}: {str(e)[:80]}")
            failed += 1
            failed_indices.append(i)
            torch.cuda.empty_cache()
            continue

    # Final save
    print(f"\n  Completed: {len(results)}/{len(samples)} (failed: {failed})")
    with open(checkpoint_path, 'w') as f:
        json.dump({"samples": results, "processed": len(samples), "failed": failed_indices, "done": True}, f)
    results_volume.commit()
    print(f"  [FINAL] Persisted to volume")

    return {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": "google/gemma-2-2b",
            "transcoder_set": "gemma",
            "n_samples": len(samples),
            "n_computed": len(results),
            "n_failed": failed,
            "source_metadata": input_data.get("metadata", {}),
        },
        "samples": results,
    }


@app.local_entrypoint()
def main(
    input_file: str,
    output_file: str = None,
    n_samples: int = None,
):
    """
    Local entrypoint for Modal.

    Args:
        input_file: Path to input JSON
        output_file: Path to output JSON (default: same dir with _metrics suffix)
        n_samples: Limit number of samples (for testing)
    """
    from pathlib import Path

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_file}")
        return

    with open(input_path) as f:
        data = json.load(f)

    if n_samples:
        data["samples"] = data["samples"][:n_samples]

    print(f"Processing {len(data['samples'])} samples from {input_file}")

    # Run on Modal
    results = compute_attribution_metrics.remote(data)

    # Save results
    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_metrics.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
