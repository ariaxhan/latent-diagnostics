"""
Modal Runner for LENGTH-MATCHED Injection Detection Experiment

Critical improvement: Controls for prompt length as a confound.
Previous experiments showed length correlates r=0.96 with n_active.

Usage:
    modal run scripts/modal_length_matched.py --n-per-class 50
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

app = modal.App("length-matched-injection", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("length-matched-results", create_if_missing=True)


@app.function(
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def compute_length_matched_metrics(
    n_per_class: int = 50,
    length_tolerance: float = 0.2,  # Match within 20%
    max_tokens: int = 200,
    save_every: int = 5,
):
    """
    Compute attribution metrics with LENGTH-MATCHED sampling.

    This addresses the critical confound: length → n_active (r=0.96).

    Sampling strategy:
    1. Pool all injections
    2. For each injection, find benign prompts within length_tolerance
    3. Sample pairs ensuring matched lengths
    4. Verify mean lengths are within 10%

    Args:
        n_per_class: Number of samples per class
        length_tolerance: Max relative length difference for matching (0.2 = 20%)
        max_tokens: Max tokens per prompt
        save_every: Checkpoint frequency
    """
    import os
    import random
    import numpy as np
    import torch
    from huggingface_hub import login
    from datasets import load_dataset
    from circuit_tracer import ReplacementModel, attribute

    random.seed(42)
    np.random.seed(42)

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("=" * 70)
    print("LENGTH-MATCHED INJECTION GEOMETRY EXPERIMENT")
    print("=" * 70)
    print(f"Controlling for length confound (r=0.96 in original data)")
    print()

    # Load full dataset
    print("[1/5] Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")

    all_injections = [{"text": x["text"], "label": True, "len": len(x["text"])}
                      for x in ds if x["label"] == 1]
    all_benign = [{"text": x["text"], "label": False, "len": len(x["text"])}
                  for x in ds if x["label"] == 0]

    print(f"  Full dataset: {len(all_injections)} injection, {len(all_benign)} benign")
    print(f"  Injection mean length: {np.mean([x['len'] for x in all_injections]):.0f} chars")
    print(f"  Benign mean length: {np.mean([x['len'] for x in all_benign]):.0f} chars")
    print()

    # Length-matched sampling
    print("[2/5] Length-matched sampling...")

    def length_match_sample(inj_pool, ben_pool, n, tolerance):
        """Sample n pairs where lengths are within tolerance."""
        matched_inj = []
        matched_ben = []
        remaining_ben = ben_pool.copy()
        random.shuffle(inj_pool)

        for inj in inj_pool:
            if len(matched_inj) >= n:
                break

            inj_len = inj['len']
            candidates = [b for b in remaining_ben
                          if abs(b['len'] - inj_len) / max(inj_len, 1) < tolerance]

            if candidates:
                match = random.choice(candidates)
                matched_inj.append(inj)
                matched_ben.append(match)
                remaining_ben.remove(match)

        return matched_inj, matched_ben

    matched_inj, matched_ben = length_match_sample(
        all_injections.copy(), all_benign.copy(), n_per_class, length_tolerance
    )

    if len(matched_inj) < n_per_class:
        print(f"  ⚠️ Could only match {len(matched_inj)} pairs (requested {n_per_class})")

    inj_mean_len = np.mean([x['len'] for x in matched_inj])
    ben_mean_len = np.mean([x['len'] for x in matched_ben])
    len_diff = abs(inj_mean_len - ben_mean_len) / max(inj_mean_len, ben_mean_len)

    print(f"  Matched pairs: {len(matched_inj)}")
    print(f"  Injection mean length: {inj_mean_len:.0f} chars")
    print(f"  Benign mean length: {ben_mean_len:.0f} chars")
    print(f"  Length difference: {100*len_diff:.1f}%")

    if len_diff > 0.15:
        print("  ⚠️ WARNING: Length difference > 15% - confound may remain")
    else:
        print("  ✓ Length controlled (<15% difference)")

    # Combine samples
    samples = []
    for i, item in enumerate(matched_inj):
        samples.append({
            "text": item["text"],
            "label": True,
            "category": "injection",
            "idx": i,
            "length": item["len"],
        })
    for i, item in enumerate(matched_ben):
        samples.append({
            "text": item["text"],
            "label": False,
            "category": "benign",
            "idx": len(matched_inj) + i,
            "length": item["len"],
        })

    random.shuffle(samples)
    print(f"  Total samples: {len(samples)}")
    print()

    # Initialize model
    print("[3/5] Loading model + transcoders...")
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        backend="transformerlens",
    )
    print("  Model ready!")

    # Compute metrics
    print("\n[4/5] Computing attribution metrics...")

    results = []
    failed = 0
    checkpoint_path = "/results/length_matched_checkpoint.json"

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
                "length": sample["length"],
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

            # Normalized metrics (length-controlled)
            length = sample["length"]
            if length > 0:
                metrics['n_active_per_char'] = metrics.get('n_active', 0) / length
                metrics['n_edges_per_char'] = metrics.get('n_edges', 0) / length

            results.append(metrics)

        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  Failed on sample {i}: {str(e)[:50]}")
            torch.cuda.empty_cache()

    print(f"\n  Completed: {len(results)} / {len(samples)} ({failed} failed)")

    # Summary with length control verification
    print("\n[5/5] Summary statistics...")

    def mean_by_label(key, label_val):
        vals = [r.get(key, 0) for r in results if r["label"] == label_val]
        return np.mean(vals) if vals else 0

    # Verify length matching held
    inj_lens = [r['length'] for r in results if r['label']]
    ben_lens = [r['length'] for r in results if not r['label']]

    print(f"\n  LENGTH VERIFICATION:")
    print(f"    Injection: {np.mean(inj_lens):.0f} chars")
    print(f"    Benign: {np.mean(ben_lens):.0f} chars")
    print(f"    Difference: {100*abs(np.mean(inj_lens)-np.mean(ben_lens))/np.mean(inj_lens):.1f}%")

    print(f"\n  RAW METRICS:")
    for metric in ['n_active', 'top_100_concentration', 'mean_influence']:
        inj = mean_by_label(metric, True)
        ben = mean_by_label(metric, False)
        print(f"    {metric}: INJ={inj:.4f}, BEN={ben:.4f}")

    print(f"\n  LENGTH-NORMALIZED METRICS:")
    for metric in ['n_active_per_char']:
        inj = mean_by_label(metric, True)
        ben = mean_by_label(metric, False)
        print(f"    {metric}: INJ={inj:.4f}, BEN={ben:.4f}")

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
            "length_matched": True,
            "length_tolerance": length_tolerance,
            "inj_mean_length": float(np.mean(inj_lens)),
            "ben_mean_length": float(np.mean(ben_lens)),
        },
        "samples": results,
    }

    final_path = "/results/length_matched_metrics.json"
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
    Run length-matched injection geometry experiment.

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

    print(f"Running LENGTH-MATCHED experiment: {n_per_class} per class")
    print("This controls for the r=0.96 length confound")
    print()

    result = compute_length_matched_metrics.remote(n_per_class=n_per_class)

    output_path = "length_matched_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Computed: {result['metadata']['n_computed']}")
    print()
    print("Next: Run the injection_geometry_truth.ipynb notebook to analyze")
