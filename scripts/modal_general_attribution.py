"""
Modal Runner for General Attribution Analysis (BULLETPROOF + GRAPH METRICS)

- Parallel batches (model loads once per container)
- Saves to Modal Volume after EVERY sample
- Volume persists even if crash
- Can pull data from volume anytime

NEW METRICS (v2):
- Degree distribution: out/in degree mean, std, max, skew
- Hub analysis: top-k hub concentration
- Spectral metrics: top eigenvalues, spectral gap, eigenvalue entropy, effective rank
- Influence distribution: median, std, Gini coefficient

Usage:
    modal run scripts/modal_general_attribution.py \
        --input-file data/cognitive_regimes/samples.json \
        --output-file data/results/cognitive_regimes_metrics.json
"""

import modal
import json
from datetime import datetime
from pathlib import Path

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

app = modal.App("attribution-bulletproof", image=image)
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("attribution-results", create_if_missing=True)

RESULTS_DIR = "/results"


@app.function(
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": hf_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,  # 1 hour per batch
    retries=0,
)
def process_batch(batch: list, batch_id: int, run_id: str, max_tokens: int = 200):
    """
    Process a batch. Model loads ONCE. Saves EACH sample to volume.
    """
    import os
    import torch
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    from circuit_tracer import ReplacementModel, attribute

    print(f"[Batch {batch_id}] Loading model...")
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        backend="transformerlens",
    )
    print(f"[Batch {batch_id}] Model ready, processing {len(batch)} samples")

    results = []
    failed = []

    for i, sample in enumerate(batch):
        idx = sample.get("idx", i)
        result = None

        try:
            text = sample["text"][:max_tokens * 4]
            torch.cuda.empty_cache()

            graph = attribute(text, model, verbose=False)

            metrics = {**sample, "status": "ok"}

            # Log available attributes for debugging
            graph_attrs = [a for a in dir(graph) if not a.startswith('_')]
            metrics['_graph_attrs'] = graph_attrs[:20]  # First 20 for debugging

            # === ACTIVE FEATURES ===
            try:
                if hasattr(graph, 'active_features'):
                    metrics['n_active'] = len(graph.active_features)
                    # Store feature IDs for top-k (useful for intervention later)
                    if len(graph.active_features) > 0:
                        metrics['n_unique_features'] = len(set(graph.active_features)) if hasattr(graph.active_features, '__iter__') else metrics['n_active']
            except Exception as e:
                metrics['_error_active_features'] = str(e)[:50]

            # === ACTIVATION VALUES ===
            try:
                if hasattr(graph, 'activation_values'):
                    acts = graph.activation_values
                    if hasattr(acts, 'abs'):
                        acts_abs = acts.abs()
                        metrics['mean_activation'] = float(acts_abs.mean().item())
                        metrics['max_activation'] = float(acts_abs.max().item())
                        metrics['activation_std'] = float(acts_abs.std().item())
                        # Activation sparsity: what fraction are effectively zero?
                        metrics['activation_sparsity'] = float((acts_abs < 1e-6).sum().item() / max(acts_abs.numel(), 1))
            except Exception as e:
                metrics['_error_activation_values'] = str(e)[:50]

            # === LOGIT EFFECTS (NEW - effect of each feature on output) ===
            try:
                if hasattr(graph, 'logit_effects'):
                    effects = graph.logit_effects
                    if hasattr(effects, 'abs'):
                        effects_abs = effects.abs()
                        metrics['mean_logit_effect'] = float(effects_abs.mean().item())
                        metrics['max_logit_effect'] = float(effects_abs.max().item())
                        metrics['logit_effect_std'] = float(effects_abs.std().item())
                        # Concentration of logit effects
                        sorted_effects, _ = effects_abs.flatten().sort(descending=True)
                        total_effect = sorted_effects.sum().item()
                        metrics['top_10_logit_effect_concentration'] = float(sorted_effects[:10].sum().item() / (total_effect + 1e-10))
                        metrics['top_100_logit_effect_concentration'] = float(sorted_effects[:100].sum().item() / (total_effect + 1e-10))
            except Exception as e:
                metrics['_error_logit_effects'] = str(e)[:50]

            # === ADJACENCY MATRIX (CAUSAL GRAPH) ===
            try:
                adj = None
                # Try different possible attribute names
                for attr_name in ['adjacency_matrix', 'adj', 'edge_weights']:
                    if hasattr(graph, attr_name):
                        adj = getattr(graph, attr_name)
                        metrics['_adj_attr_name'] = attr_name
                        break

                if adj is not None and hasattr(adj, 'abs'):
                    adj_abs = adj.abs()
                    metrics['adj_shape'] = list(adj_abs.shape)

                    # === BASIC METRICS ===
                    metrics['n_edges'] = int((adj_abs > 0.01).sum().item())
                    metrics['mean_influence'] = float(adj_abs.mean().item())
                    metrics['max_influence'] = float(adj_abs.max().item())

                    flat = adj_abs.flatten()
                    sorted_inf, _ = flat.sort(descending=True)
                    total = flat.sum().item()
                    metrics['total_influence'] = float(total)
                    top_100 = sorted_inf[:100].sum().item()
                    metrics['top_100_concentration'] = float(top_100 / (total + 1e-10))
                    metrics['top_1000_concentration'] = float(sorted_inf[:1000].sum().item() / (total + 1e-10))

                    # === DEGREE DISTRIBUTION ===
                    try:
                        threshold = 0.01
                        out_degree = (adj_abs > threshold).sum(dim=1).float()
                        in_degree = (adj_abs > threshold).sum(dim=0).float()

                        metrics['out_degree_mean'] = float(out_degree.mean().item())
                        metrics['out_degree_std'] = float(out_degree.std().item())
                        metrics['out_degree_max'] = float(out_degree.max().item())
                        metrics['in_degree_mean'] = float(in_degree.mean().item())
                        metrics['in_degree_std'] = float(in_degree.std().item())
                        metrics['in_degree_max'] = float(in_degree.max().item())

                        # Degree skewness
                        out_mean = out_degree.mean()
                        out_std = out_degree.std() + 1e-10
                        metrics['out_degree_skew'] = float(((out_degree - out_mean) ** 3).mean().item() / (out_std.item() ** 3))
                    except Exception as e:
                        metrics['_error_degree'] = str(e)[:50]

                    # === HUB ANALYSIS ===
                    try:
                        out_influence = adj_abs.sum(dim=1)
                        sorted_out, top_hub_indices = out_influence.sort(descending=True)
                        total_out = out_influence.sum().item()

                        metrics['top_10_hub_concentration'] = float(sorted_out[:10].sum().item() / (total_out + 1e-10))
                        metrics['top_50_hub_concentration'] = float(sorted_out[:50].sum().item() / (total_out + 1e-10))
                        metrics['top_100_hub_concentration'] = float(sorted_out[:100].sum().item() / (total_out + 1e-10))

                        # Store indices of top hub features (for intervention experiments)
                        metrics['top_10_hub_indices'] = top_hub_indices[:10].tolist()
                    except Exception as e:
                        metrics['_error_hub'] = str(e)[:50]

                    # === SPECTRAL METRICS ===
                    try:
                        n_features = adj_abs.shape[0]
                        # Limit size for computational feasibility
                        max_size = min(n_features, 500)

                        if n_features > max_size:
                            # Sample top features by influence
                            top_indices = out_influence.topk(max_size).indices
                            adj_sample = adj_abs[top_indices][:, top_indices]
                        else:
                            adj_sample = adj_abs

                        # SVD is more stable than eigendecomposition
                        U, S, V = torch.linalg.svd(adj_sample, full_matrices=False)
                        singular_values = S.sort(descending=True).values

                        metrics['singular_value_1'] = float(singular_values[0].item())
                        metrics['singular_value_2'] = float(singular_values[1].item()) if len(singular_values) > 1 else 0.0
                        metrics['singular_value_3'] = float(singular_values[2].item()) if len(singular_values) > 2 else 0.0

                        # Spectral gap
                        if singular_values[1] > 1e-10:
                            metrics['spectral_gap'] = float((singular_values[0] / singular_values[1]).item())
                        else:
                            metrics['spectral_gap'] = 1000.0  # Cap instead of inf

                        # Spectral entropy
                        sv_pos = singular_values[singular_values > 1e-10]
                        if len(sv_pos) > 0:
                            sv_probs = sv_pos / sv_pos.sum()
                            metrics['spectral_entropy'] = float(-(sv_probs * sv_probs.log()).sum().item())
                            metrics['effective_rank'] = float(torch.exp(torch.tensor(metrics['spectral_entropy'])).item())
                        else:
                            metrics['spectral_entropy'] = 0.0
                            metrics['effective_rank'] = 1.0
                    except Exception as e:
                        metrics['_error_spectral'] = str(e)[:50]

                    # === INFLUENCE DISTRIBUTION ===
                    try:
                        nonzero = flat[flat > 1e-10]
                        if len(nonzero) > 0:
                            metrics['influence_median'] = float(nonzero.median().item())
                            metrics['influence_std'] = float(nonzero.std().item())
                            metrics['influence_p90'] = float(nonzero.quantile(0.9).item())
                            metrics['influence_p99'] = float(nonzero.quantile(0.99).item())

                            # Gini coefficient
                            sorted_nz = nonzero.sort().values
                            n = len(sorted_nz)
                            indices = torch.arange(1, n + 1, device=sorted_nz.device, dtype=torch.float32)
                            gini = (2 * (indices @ sorted_nz) - (n + 1) * sorted_nz.sum()) / (n * sorted_nz.sum() + 1e-10)
                            metrics['influence_gini'] = float(gini.item())
                    except Exception as e:
                        metrics['_error_influence_dist'] = str(e)[:50]

            except Exception as e:
                metrics['_error_adjacency'] = str(e)[:100]

            # === LOGIT PROBABILITIES ===
            try:
                if hasattr(graph, 'logit_probabilities'):
                    probs = graph.logit_probabilities
                    if hasattr(probs, 'max'):
                        metrics['max_logit_prob'] = float(probs.max().item())
                        metrics['logit_entropy'] = float(-(probs * (probs + 1e-10).log()).sum().item())
                        # Top-k probability mass
                        sorted_probs, _ = probs.flatten().sort(descending=True)
                        metrics['top_5_prob_mass'] = float(sorted_probs[:5].sum().item())
                        metrics['top_10_prob_mass'] = float(sorted_probs[:10].sum().item())
            except Exception as e:
                metrics['_error_logit_probs'] = str(e)[:50]

            # === FEATURE ACTIVATIONS (alternative attribute name) ===
            try:
                if hasattr(graph, 'feature_acts') and not metrics.get('mean_activation'):
                    acts = graph.feature_acts
                    if hasattr(acts, 'abs'):
                        metrics['mean_activation'] = float(acts.abs().mean().item())
                        metrics['max_activation'] = float(acts.abs().max().item())
            except Exception as e:
                metrics['_error_feature_acts'] = str(e)[:50]

            result = metrics
            results.append(metrics)
            print(f"[Batch {batch_id}] {i+1}/{len(batch)} idx={idx} OK")

        except torch.cuda.OutOfMemoryError:
            result = {**sample, "status": "oom", "error": "CUDA OOM"}
            failed.append(result)
            print(f"[Batch {batch_id}] {i+1}/{len(batch)} idx={idx} OOM")
            torch.cuda.empty_cache()

        except Exception as e:
            result = {**sample, "status": "error", "error": str(e)[:200]}
            failed.append(result)
            print(f"[Batch {batch_id}] {i+1}/{len(batch)} idx={idx} ERROR: {str(e)[:50]}")
            torch.cuda.empty_cache()

        # SAVE TO VOLUME AFTER EVERY SAMPLE
        if result:
            sample_file = f"{RESULTS_DIR}/{run_id}/sample_{idx:05d}.json"
            os.makedirs(os.path.dirname(sample_file), exist_ok=True)
            with open(sample_file, "w") as f:
                json.dump(result, f)
            results_volume.commit()  # Persist to volume immediately

    print(f"[Batch {batch_id}] Complete: {len(results)} ok, {len(failed)} failed")
    return {"results": results, "failed": failed, "batch_id": batch_id}


@app.local_entrypoint()
def main(
    input_file: str,
    output_file: str = None,
    n_samples: int = None,
    n_workers: int = 8,
    batch_size: int = None,
    run_id: str = None,
):
    """
    Parallel attribution with volume backup.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_file}")
        return

    with open(input_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if n_samples:
        samples = samples[:n_samples]

    if batch_size is None:
        batch_size = max(1, len(samples) // n_workers)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_metrics.json")

    # Check volume for already-completed samples
    completed_ids = set()
    try:
        for item in results_volume.listdir(f"/{run_id}"):
            if item.path.endswith(".json"):
                idx = int(item.path.split("_")[-1].replace(".json", ""))
                completed_ids.add(idx)
        if completed_ids:
            print(f"RESUMING: {len(completed_ids)} samples already on volume")
    except Exception:
        pass

    # Filter out completed
    remaining = [s for s in samples if s.get("idx") not in completed_ids]

    if not remaining:
        print("All samples completed! Pulling from volume...")
        pull_from_volume(run_id, output_file, data.get("metadata", {}), len(samples))
        return

    # Create batches
    batches = []
    for i in range(0, len(remaining), batch_size):
        batches.append(remaining[i:i + batch_size])

    print(f"Processing {len(remaining)} samples in {len(batches)} batches")
    print(f"Run ID: {run_id}")
    print(f"Volume: attribution-results (persistent)")
    print(f"Saving EACH sample to volume\n")

    # Process in parallel - return_exceptions=True
    all_results = []
    all_failed = []

    for batch_result in process_batch.starmap(
        [(batch, i, run_id) for i, batch in enumerate(batches)],
        return_exceptions=True
    ):
        if isinstance(batch_result, Exception):
            print(f"BATCH FAILED: {type(batch_result).__name__}: {str(batch_result)[:100]}")
            continue

        all_results.extend(batch_result["results"])
        all_failed.extend(batch_result["failed"])
        print(f"Batch {batch_result['batch_id']}: +{len(batch_result['results'])} ok")

    # Pull everything from volume and save locally
    print(f"\nPulling all results from volume...")
    pull_from_volume(run_id, output_file, data.get("metadata", {}), len(samples))

    print(f"\nDONE: {len(all_results)} ok, {len(all_failed)} failed")
    print(f"Saved to: {output_file}")
    print(f"Volume backup: attribution-results/{run_id}/")


def pull_from_volume(run_id: str, output_file: str, source_metadata: dict, total_samples: int):
    """Pull all results from volume and save locally."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download from volume
        subprocess.run([
            "modal", "volume", "get", "attribution-results",
            f"/{run_id}", tmpdir
        ], check=True)

        # Collect all sample files
        results = []
        failed = []
        sample_dir = Path(tmpdir) / run_id

        if sample_dir.exists():
            for f in sample_dir.glob("sample_*.json"):
                with open(f) as fp:
                    r = json.load(fp)
                    if r.get("status") == "ok":
                        results.append(r)
                    else:
                        failed.append(r)

        results.sort(key=lambda x: x.get("idx", 0))

        output = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "model": "google/gemma-2-2b",
                "transcoder_set": "gemma",
                "n_samples": total_samples,
                "n_computed": len(results),
                "n_failed": len(failed),
                "run_id": run_id,
                "source_metadata": source_metadata,
            },
            "samples": results,
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Pulled {len(results)} results from volume -> {output_file}")
