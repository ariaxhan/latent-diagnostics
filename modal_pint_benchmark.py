"""
Modal Runner for PINT Benchmark Attribution Analysis

Computes attribution graph metrics for all PINT benchmark samples.
Outputs pint_metrics.json for local analysis.

Usage:
    # Run on example subset (fast)
    modal run modal_pint_benchmark.py --n-samples 100

    # Run full PINT benchmark
    modal run modal_pint_benchmark.py --full

    # Output saved to: pint_metrics.json
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

app = modal.App("pint-benchmark", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


results_volume = modal.Volume.from_name("pint-results", create_if_missing=True)


@app.function(
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours for full benchmark
)
def compute_pint_metrics(
    n_samples: int = None,
    use_deepset: bool = True,
    pint_yaml_path: str = None,
    max_tokens: int = 200,  # Aggressive truncation to avoid OOM
    save_every: int = 5,  # Save checkpoint every N samples
):
    """
    Compute attribution metrics for PINT benchmark samples.

    Args:
        n_samples: Limit number of samples (None = all)
        use_deepset: If True, use deepset/prompt-injections dataset
        pint_yaml_path: Path to PINT YAML file (if not using deepset)

    Returns:
        Dict with samples and their metrics
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
    print("PINT BENCHMARK - ATTRIBUTION ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\n[1/4] Loading dataset...")

    samples = []
    if use_deepset:
        from datasets import load_dataset
        ds = load_dataset("deepset/prompt-injections", split="train")

        for i, item in enumerate(ds):
            if n_samples and i >= n_samples:
                break
            samples.append({
                "text": item["text"],
                "label": item["label"] == 1,  # 1 = injection
                "category": "prompt_injection" if item["label"] == 1 else "benign",
                "idx": i,
            })
    else:
        # Load from YAML (PINT format)
        import yaml
        with open(pint_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        for i, item in enumerate(data):
            if n_samples and i >= n_samples:
                break
            samples.append({
                "text": item["text"],
                "label": item["label"],
                "category": item.get("category", "unknown"),
                "idx": i,
            })

    print(f"  Loaded {len(samples)} samples")
    n_inj = sum(1 for s in samples if s["label"])
    print(f"  Injection: {n_inj}, Benign: {len(samples) - n_inj}")

    # Initialize model
    print("\n[2/4] Loading model + transcoders...")

    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        backend="transformerlens",
    )
    print("  Model ready!")

    # Compute metrics
    print("\n[3/4] Computing attribution metrics...")

    results = []
    failed = 0
    checkpoint_path = "/results/pint_checkpoint.json"

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(samples)} ({100 * (i + 1) / len(samples):.1f}%)")

        # Save checkpoint periodically
        if (i + 1) % save_every == 0 and results:
            with open(checkpoint_path, 'w') as f:
                json.dump({"samples": results, "processed": i + 1}, f)
            results_volume.commit()
            print(f"  [checkpoint] Saved {len(results)} results to volume")

        try:
            # Aggressive truncation to avoid OOM
            text = sample["text"][:max_tokens * 4]  # ~4 chars per token estimate

            # Clear CUDA cache before each sample
            torch.cuda.empty_cache()

            graph = attribute(text, model, verbose=False)

            # Extract metrics
            metrics = {
                "idx": sample["idx"],
                "text": sample["text"][:200],  # Truncate for storage
                "label": sample["label"],
                "category": sample["category"],
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

        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Failed on sample {i}: {str(e)[:50]}")
            # Clear cache on failure
            torch.cuda.empty_cache()

    print(f"\n  Completed: {len(results)} / {len(samples)} ({failed} failed)")

    # Summary statistics
    print("\n[4/4] Computing summary...")

    def mean_by_label(key, label_val):
        vals = [r.get(key, 0) for r in results if r["label"] == label_val]
        return sum(vals) / len(vals) if vals else 0

    print("\n  INJECTION vs BENIGN:")
    for metric in ['n_active', 'top_100_concentration', 'mean_influence', 'n_edges']:
        inj = mean_by_label(metric, True)
        ben = mean_by_label(metric, False)
        print(f"    {metric}: INJ={inj:.4f}, BEN={ben:.4f}")

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": "google/gemma-2-2b",
            "transcoder_set": "gemma",
            "n_samples": len(samples),
            "n_computed": len(results),
            "n_failed": failed,
            "dataset": "deepset/prompt-injections" if use_deepset else pint_yaml_path,
        },
        "samples": results,
    }

    # Save final results to volume
    final_path = "/results/pint_metrics.json"
    with open(final_path, 'w') as f:
        json.dump(output, f, indent=2)
    results_volume.commit()
    print(f"\n✓ Final results saved to Modal volume: {final_path}")

    print("\n" + "=" * 70)
    print("DONE - Return metrics for local analysis")
    print("=" * 70)

    return output


@app.function(volumes={"/results": results_volume})
def download_results():
    """Download results from Modal volume."""
    import os

    files = []
    for f in os.listdir("/results"):
        path = f"/results/{f}"
        with open(path, 'r') as fp:
            content = fp.read()
        files.append({"name": f, "content": content})
        print(f"  Found: {f} ({len(content)} bytes)")

    return files


@app.local_entrypoint()
def main(
    n_samples: int = None,
    full: bool = False,
    download: bool = False,
):
    """
    Run PINT benchmark attribution analysis.

    Args:
        n_samples: Limit samples (default: 100 for testing)
        full: Run on all samples
        download: Download results from Modal volume (if job ran with --detach)
    """
    if download:
        print("Downloading results from Modal volume...")
        files = download_results.remote()
        for f in files:
            output_path = f["name"]
            with open(output_path, 'w') as fp:
                fp.write(f["content"])
            print(f"✓ Downloaded: {output_path}")
        return

    if full:
        n_samples = None
    elif n_samples is None:
        n_samples = 100  # Default for quick testing

    print(f"Running PINT benchmark with n_samples={n_samples or 'ALL'}")
    print()
    print("TIP: Use 'modal run --detach modal_pint_benchmark.py --full' to avoid client disconnect")
    print("     Then 'modal run modal_pint_benchmark.py --download' to retrieve results")
    print()

    result = compute_pint_metrics.remote(n_samples=n_samples)

    # Save results locally
    output_path = "pint_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Samples: {result['metadata']['n_computed']}")
    print(f"  Failed: {result['metadata']['n_failed']}")
    print()
    print("Next: python experiments/08_injection_detection.py")
