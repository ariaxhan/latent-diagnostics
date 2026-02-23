"""
Modal Runner for General Attribution Analysis (PARALLEL)

Computes attribution graph metrics for any JSON dataset.
Uses parallel containers for ~10x speedup.

Usage:
    modal run scripts/modal_general_attribution.py \
        --input-file data/truthfulness/truthfulness_samples.json \
        --output-file data/results/truthfulness_metrics.json

    # With custom parallelism
    modal run scripts/modal_general_attribution.py \
        --input-file data/domain_analysis/domain_samples.json \
        --n-workers 5
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

app = modal.App("general-attribution-parallel", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="A100",
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,  # 30 min per batch (should be plenty)
    retries=1,
)
def process_batch(batch: list, batch_id: int, max_tokens: int = 200):
    """
    Process a batch of samples on one GPU.
    Model loads once per container, then processes all samples in batch.
    """
    import os
    import torch
    from huggingface_hub import login

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

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
        try:
            text = sample["text"][:max_tokens * 4]
            torch.cuda.empty_cache()

            graph = attribute(text, model, verbose=False)

            metrics = {**sample}

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
                    metrics['n_edges'] = int((adj.abs() > 0.01).sum().item())
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

            if (i + 1) % 5 == 0:
                print(f"[Batch {batch_id}] {i+1}/{len(batch)} done")

        except torch.cuda.OutOfMemoryError:
            print(f"[Batch {batch_id}] OOM on sample {sample.get('idx', i)}")
            failed.append(sample.get('idx', i))
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Batch {batch_id}] Error on sample {sample.get('idx', i)}: {str(e)[:60]}")
            failed.append(sample.get('idx', i))
            torch.cuda.empty_cache()

    print(f"[Batch {batch_id}] Complete: {len(results)} ok, {len(failed)} failed")
    return {"results": results, "failed": failed, "batch_id": batch_id}


@app.local_entrypoint()
def main(
    input_file: str,
    output_file: str = None,
    n_samples: int = None,
    n_workers: int = 8,
    batch_size: int = None,
):
    """
    Parallel attribution computation.

    Args:
        input_file: Path to input JSON
        output_file: Path to output JSON
        n_samples: Limit samples (for testing)
        n_workers: Number of parallel containers (default 8)
        batch_size: Samples per container (default: auto)
    """
    from pathlib import Path

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_file}")
        return

    with open(input_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if n_samples:
        samples = samples[:n_samples]

    # Auto batch size: divide samples among workers
    if batch_size is None:
        batch_size = max(1, len(samples) // n_workers)

    # Create batches
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])

    print(f"Processing {len(samples)} samples in {len(batches)} batches ({batch_size} per batch)")
    print(f"Using up to {n_workers} parallel A100 containers")

    # Process in parallel with .map()
    batch_args = [(batch, i) for i, batch in enumerate(batches)]

    all_results = []
    all_failed = []

    # Use starmap for parallel execution
    for batch_result in process_batch.starmap(batch_args):
        all_results.extend(batch_result["results"])
        all_failed.extend(batch_result["failed"])
        print(f"  Batch {batch_result['batch_id']} returned {len(batch_result['results'])} results")

    # Sort by original index
    all_results.sort(key=lambda x: x.get("idx", 0))

    # Build output
    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": "google/gemma-2-2b",
            "transcoder_set": "gemma",
            "n_samples": len(samples),
            "n_computed": len(all_results),
            "n_failed": len(all_failed),
            "n_batches": len(batches),
            "source_metadata": data.get("metadata", {}),
        },
        "samples": all_results,
    }

    # Save
    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_metrics.json")

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! {len(all_results)}/{len(samples)} samples computed")
    print(f"Failed: {len(all_failed)}")
    print(f"Saved to: {output_file}")
