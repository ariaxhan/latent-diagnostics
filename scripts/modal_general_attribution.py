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
    checkpoint_path = "/results/attribution_checkpoint.json"

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(samples)} ({100 * (i + 1) / len(samples):.1f}%)")

        # Save checkpoint
        if (i + 1) % save_every == 0 and results:
            with open(checkpoint_path, 'w') as f:
                json.dump({"samples": results, "processed": i + 1}, f)

        try:
            text = sample["text"]

            # Tokenize with truncation
            tokens = model.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = model.tokenizer.decode(tokens)

            # Get attribution graph
            graph = attribute(
                model,
                tokens,
                max_n_logits=1,
                verbose=False,
            )

            # Compute metrics from graph
            activations = graph.activations
            edges = graph.edges

            n_active = len(activations)
            n_edges = len(edges)

            edge_weights = [abs(e.weight) for e in edges] if edges else [0]
            mean_influence = sum(edge_weights) / len(edge_weights) if edge_weights else 0
            max_influence = max(edge_weights) if edge_weights else 0

            # Top feature concentration
            if activations:
                act_values = sorted([abs(a.value) for a in activations], reverse=True)
                total_act = sum(act_values)
                top_100 = sum(act_values[:100])
                top_100_concentration = top_100 / total_act if total_act > 0 else 0
                mean_activation = total_act / len(act_values) if act_values else 0
                max_activation = max(act_values)
            else:
                top_100_concentration = 0
                mean_activation = 0
                max_activation = 0

            # Output logit stats
            logits = graph.logits if hasattr(graph, 'logits') else []
            if logits:
                import math
                probs = [math.exp(l.value) for l in logits]
                total_prob = sum(probs)
                probs = [p / total_prob for p in probs]
                max_logit_prob = max(probs)
                logit_entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            else:
                max_logit_prob = 0
                logit_entropy = 0

            result = {
                **sample,  # Preserve original fields (idx, domain, label, etc.)
                "n_active": n_active,
                "n_edges": n_edges,
                "mean_activation": mean_activation,
                "max_activation": max_activation,
                "mean_influence": mean_influence,
                "max_influence": max_influence,
                "top_100_concentration": top_100_concentration,
                "max_logit_prob": max_logit_prob,
                "logit_entropy": logit_entropy,
                "n_tokens": len(tokens),
            }
            results.append(result)

        except Exception as e:
            print(f"  [FAIL] Sample {sample.get('idx', i)}: {str(e)[:80]}")
            failed += 1
            continue

    print(f"\n  Completed: {len(results)}/{len(samples)} (failed: {failed})")

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
