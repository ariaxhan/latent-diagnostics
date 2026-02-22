"""
Modal Runner for BALANCED Injection Detection Experiment

Creates equal-sized injection and benign samples for proper statistical analysis.
Addresses the class imbalance issue (previous: 21 vs 115).

Usage:
    # Run balanced experiment (50 per class = 100 total)
    modal run scripts/modal_balanced_benchmark.py --n-per-class 50

    # Smaller test run
    modal run scripts/modal_balanced_benchmark.py --n-per-class 20
"""

import modal
import json
from datetime import datetime

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.20.0",
    )
    .pip_install(
        "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git",
    )
)

app = modal.App("balanced-injection", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("balanced-results", create_if_missing=True)


@app.function(
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def compute_balanced_metrics(
    n_per_class: int = 50,
    max_tokens: int = 200,
    save_every: int = 5,
):
    """
    Compute attribution metrics with BALANCED sampling.

    Args:
        n_per_class: Number of samples per class (injection + benign)
        max_tokens: Max tokens per prompt to avoid OOM
        save_every: Checkpoint frequency

    Returns:
        Dict with balanced samples and metrics
    """
    import os
    import random
    import torch
    from huggingface_hub import login
    from datasets import load_dataset
    from circuit_tracer import ReplacementModel, attribute

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("=" * 70)
    print("BALANCED INJECTION GEOMETRY EXPERIMENT")
    print("=" * 70)

    # Load full dataset
    print("\n[1/4] Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")

    all_injections = [x for x in ds if x["label"] == 1]
    all_benign = [x for x in ds if x["label"] == 0]

    print(f"  Full dataset: {len(all_injections)} injection, {len(all_benign)} benign")

    # Balanced sampling
    random.seed(42)
    n_per_class = min(n_per_class, len(all_injections), len(all_benign))

    sampled_injections = random.sample(all_injections, n_per_class)
    sampled_benign = random.sample(all_benign, n_per_class)

    samples = []
    for i, item in enumerate(sampled_injections):
        samples.append({
            "text": item["text"],
            "label": True,
            "category": "injection",
            "idx": i,
        })
    for i, item in enumerate(sampled_benign):
        samples.append({
            "text": item["text"],
            "label": False,
            "category": "benign",
            "idx": len(sampled_injections) + i,
        })

    # Shuffle
    random.shuffle(samples)

    print(f"  BALANCED sample: {n_per_class} injection + {n_per_class} benign = {len(samples)} total")
    print(f"  Class ratio: 1:1 (baseline accuracy = 50%)")

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
    checkpoint_path = "/results/balanced_checkpoint.json"

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(samples)}")

        if (i + 1) % save_every == 0 and results:
            with open(checkpoint_path, 'w') as f:
                json.dump({"samples": results, "processed": i + 1}, f)
            results_volume.commit()

        try:
            text = sample["text"][:max_tokens * 4]
            torch.cuda.empty_cache()

            graph = attribute(text, model, verbose=False)

            metrics = {
                "idx": sample["idx"],
                "text": sample["text"][:200],
                "label": sample["label"],
                "category": sample["category"],
            }

            if hasattr(graph, 'active_features'):
                metrics['n_active'] = len(graph.active_features)

            if hasattr(graph, 'activation_values'):
                acts = graph.activation_values
                if hasattr(acts, 'abs'):
                    metrics['mean_activation'] = float(acts.abs().mean().item())
                    metrics['max_activation'] = float(acts.abs().max().item())

            if hasattr(graph, 'adjacency_matrix'):
                adj = graph.adjacency_matrix
                if hasattr(adj, 'abs'):
                    n_edges = int((adj.abs() > 0.01).sum().item())
                    metrics['n_edges'] = n_edges
                    metrics['mean_influence'] = float(adj.abs().mean().item())
                    metrics['max_influence'] = float(adj.abs().max().item())

                    flat = adj.abs().flatten()
                    sorted_inf, _ = flat.sort(descending=True)
                    total = flat.sum().item()
                    top_100 = sorted_inf[:100].sum().item()
                    metrics['top_100_concentration'] = float(top_100 / (total + 1e-10))

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
            torch.cuda.empty_cache()

    print(f"\n  Completed: {len(results)} / {len(samples)} ({failed} failed)")

    # Summary
    print("\n[4/4] Summary statistics...")

    def mean_by_label(key, label_val):
        vals = [r.get(key, 0) for r in results if r["label"] == label_val]
        return sum(vals) / len(vals) if vals else 0

    print("\n  INJECTION vs BENIGN (balanced):")
    for metric in ['n_active', 'top_100_concentration', 'mean_influence', 'n_edges']:
        inj = mean_by_label(metric, True)
        ben = mean_by_label(metric, False)
        ratio = inj / ben if ben > 0 else 0
        print(f"    {metric}: INJ={inj:.4f}, BEN={ben:.4f}, ratio={ratio:.2f}x")

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": "google/gemma-2-2b",
            "transcoder_set": "gemma",
            "n_per_class": n_per_class,
            "n_total": len(samples),
            "n_computed": len(results),
            "n_failed": failed,
            "dataset": "deepset/prompt-injections",
            "balanced": True,
        },
        "samples": results,
    }

    final_path = "/results/balanced_metrics.json"
    with open(final_path, 'w') as f:
        json.dump(output, f, indent=2)
    results_volume.commit()

    print(f"\n✓ Results saved to: {final_path}")
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
        print(f"  Found: {f}")
    return files


@app.local_entrypoint()
def main(
    n_per_class: int = 50,
    download: bool = False,
):
    """
    Run balanced injection geometry experiment.

    Args:
        n_per_class: Samples per class (default 50)
        download: Download results from previous run
    """
    if download:
        print("Downloading results...")
        files = download_results.remote()
        for f in files:
            with open(f["name"], 'w') as fp:
                fp.write(f["content"])
            print(f"✓ Downloaded: {f['name']}")
        return

    print(f"Running BALANCED experiment: {n_per_class} per class")
    print(f"Total samples: {n_per_class * 2}")
    print()

    result = compute_balanced_metrics.remote(n_per_class=n_per_class)

    output_path = "balanced_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Computed: {result['metadata']['n_computed']}")
    print()
    print("Next: Run the balanced_injection_geometry.ipynb notebook")
