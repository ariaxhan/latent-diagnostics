"""
Modal Runner for Vector Native Density Test

Computes attribution metrics for VN symbols vs natural language equivalents.
Tests the core VN hypothesis: structured symbols activate denser representations.

Usage:
    modal run scripts/modal_vn_density.py

    # Download results
    modal run scripts/modal_vn_density.py --download
"""

import modal
import json
from datetime import datetime

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "huggingface-hub>=0.20.0",
    )
    .pip_install(
        "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git",
    )
)

app = modal.App("vn-density-test", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("vn-density-results", create_if_missing=True)


# =============================================================================
# TEST CASES (same as vn_density_test.py)
# =============================================================================

TEST_CASES = [
    # Symbol comparisons
    {
        "name": "attention_symbol",
        "category": "symbol",
        "variants": {
            "vn_symbol": "●",
            "word_attention": "attention",
            "word_important": "important",
            "word_focus": "focus",
            "phrase": "pay attention to this",
        },
        "hypothesis": "● has highest density"
    },
    {
        "name": "merge_symbol",
        "category": "symbol",
        "variants": {
            "vn_symbol": "⊕",
            "word_add": "add",
            "word_merge": "merge",
            "word_combine": "combine",
            "phrase": "add these together",
        },
        "hypothesis": "⊕ has highest density"
    },
    {
        "name": "flow_symbol",
        "category": "symbol",
        "variants": {
            "vn_symbol": "→",
            "word_then": "then",
            "word_next": "next",
            "phrase": "and then do",
        },
        "hypothesis": "→ has high density"
    },
    {
        "name": "block_symbol",
        "category": "symbol",
        "variants": {
            "vn_symbol": "≠",
            "word_not": "not",
            "word_block": "block",
            "word_reject": "reject",
            "phrase": "do not allow",
        },
        "hypothesis": "≠ has highest density"
    },
    # Format comparisons
    {
        "name": "simple_instruction",
        "category": "format",
        "variants": {
            "vn": "●analyze|data:sales",
            "natural": "Please analyze the sales data",
            "terse": "analyze sales data",
        },
        "hypothesis": "VN format has highest density"
    },
    {
        "name": "complex_instruction",
        "category": "format",
        "variants": {
            "vn": "●analyze|dataset:Q4_sales|focus:revenue|output:summary",
            "natural": "Please analyze the Q4 sales dataset, focusing on revenue, and output a summary",
            "terse": "analyze Q4 sales revenue summary",
        },
        "hypothesis": "VN maintains density advantage"
    },
    {
        "name": "workflow",
        "category": "format",
        "variants": {
            "vn": "●workflow|id:review→●analyze|data:Q4→●generate|type:report",
            "natural": "Start a workflow called review. First analyze the Q4 data. Then generate a report.",
            "terse": "workflow review: analyze Q4, generate report",
        },
        "hypothesis": "VN workflow syntax is denser"
    },
    {
        "name": "state_declaration",
        "category": "format",
        "variants": {
            "vn": "●STATE|Ψ:editor|Ω:documentation|mode:execute",
            "natural": "You are an editor. Your goal is documentation. Execute mode.",
            "terse": "editor, goal: documentation, mode: execute",
        },
        "hypothesis": "VN state symbols are densest"
    },
    # Structure comparisons
    {
        "name": "key_value_pairs",
        "category": "structure",
        "variants": {
            "config": "name:John|age:30|role:engineer",
            "json_like": '{"name": "John", "age": 30, "role": "engineer"}',
            "prose": "John is 30 years old and works as an engineer",
        },
        "hypothesis": "Pipe-delimited config is densest"
    },
    {
        "name": "parameter_list",
        "category": "structure",
        "variants": {
            "vn_pipes": "dataset:sales|metrics:revenue,profit|period:Q4",
            "comma_list": "dataset=sales, metrics=revenue and profit, period=Q4",
            "prose": "using the sales dataset, looking at revenue and profit metrics for Q4",
        },
        "hypothesis": "VN pipe syntax is densest"
    },
]


@app.function(
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def compute_vn_density():
    """
    Compute attribution metrics for all VN test cases.

    Returns dict with results for each test case and variant.
    """
    import os
    import torch
    from huggingface_hub import login
    from circuit_tracer import ReplacementModel, attribute

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("=" * 70)
    print("VECTOR NATIVE DENSITY TEST")
    print("=" * 70)
    print()

    # Count total variants
    total_variants = sum(len(tc["variants"]) for tc in TEST_CASES)
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Total variants: {total_variants}")
    print()

    # Initialize model
    print("[1/3] Loading model + transcoders...")
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        backend="transformerlens",
    )
    print("  Model ready!")

    # Process all variants
    print("\n[2/3] Computing attribution metrics...")

    results = []
    failed = 0
    processed = 0

    for test in TEST_CASES:
        print(f"\n  Test: {test['name']} ({test['category']})")

        for variant_name, text in test["variants"].items():
            processed += 1
            print(f"    [{processed}/{total_variants}] {variant_name}: '{text[:30]}...' ", end="")

            try:
                torch.cuda.empty_cache()
                graph = attribute(text, model, verbose=False)

                # Extract metrics
                metrics = {
                    "test_name": test["name"],
                    "category": test["category"],
                    "variant": variant_name,
                    "text": text,
                    "length": len(text),
                    "hypothesis": test["hypothesis"],
                }

                # Active features
                if hasattr(graph, 'active_features'):
                    metrics['n_active'] = len(graph.active_features)
                    metrics['density'] = metrics['n_active'] / max(len(text), 1)

                # Activation values
                if hasattr(graph, 'activation_values'):
                    acts = graph.activation_values
                    if hasattr(acts, 'abs'):
                        metrics['mean_activation'] = float(acts.abs().mean().item())
                        metrics['max_activation'] = float(acts.abs().max().item())

                # Edges
                if hasattr(graph, 'adjacency_matrix'):
                    adj = graph.adjacency_matrix
                    if hasattr(adj, 'abs'):
                        n_edges = int((adj.abs() > 0.01).sum().item())
                        metrics['n_edges'] = n_edges
                        metrics['mean_influence'] = float(adj.abs().mean().item())

                        # Concentration
                        flat = adj.abs().flatten()
                        sorted_inf, _ = flat.sort(descending=True)
                        total = flat.sum().item()
                        top_100 = sorted_inf[:100].sum().item()
                        metrics['concentration'] = float(top_100 / (total + 1e-10))

                results.append(metrics)
                print(f"✓ density={metrics.get('density', 0):.0f}")

            except Exception as e:
                failed += 1
                print(f"✗ {str(e)[:30]}")
                torch.cuda.empty_cache()

    print(f"\n  Completed: {len(results)}/{total_variants} ({failed} failed)")

    # Analyze results
    print("\n[3/3] Analyzing results...")

    # Group by test
    by_test = {}
    for r in results:
        test_name = r["test_name"]
        if test_name not in by_test:
            by_test[test_name] = {}
        by_test[test_name][r["variant"]] = r

    # Summary
    vn_wins = 0
    total_tests = 0

    print("\n" + "=" * 70)
    print("RESULTS BY TEST")
    print("=" * 70)

    for test in TEST_CASES:
        if test["name"] not in by_test:
            continue

        variants = by_test[test["name"]]
        densities = {v: variants[v].get("density", 0) for v in variants}

        if not densities:
            continue

        total_tests += 1
        max_variant = max(densities, key=densities.get)

        # Check if VN won
        vn_variants = ["vn_symbol", "vn", "config", "vn_pipes"]
        vn_won = any(v in max_variant for v in vn_variants)
        if vn_won:
            vn_wins += 1

        status = "✓ VN" if vn_won else "✗"
        print(f"\n{status} {test['name']}")
        for v, d in sorted(densities.items(), key=lambda x: -x[1]):
            marker = "←" if v == max_variant else ""
            print(f"    {v}: {d:.0f} {marker}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: VN won {vn_wins}/{total_tests} ({100*vn_wins/max(total_tests,1):.0f}%)")
    print("=" * 70)

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": "google/gemma-2-2b",
            "transcoder_set": "gemma",
            "n_tests": len(TEST_CASES),
            "n_variants": total_variants,
            "n_computed": len(results),
            "n_failed": failed,
        },
        "test_cases": TEST_CASES,
        "results": results,
        "summary": {
            "vn_wins": vn_wins,
            "total_tests": total_tests,
            "vn_win_rate": vn_wins / max(total_tests, 1),
        },
    }

    # Save to volume
    final_path = "/results/vn_density_results.json"
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
def main(download: bool = False):
    """
    Run Vector Native density test.

    Args:
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

    print("Running Vector Native Density Test")
    print("=" * 50)
    print()
    print("This will test whether VN symbols activate denser representations.")
    print()

    result = compute_vn_density.remote()

    output_path = "vn_density_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"  VN win rate: {100*result['summary']['vn_win_rate']:.0f}%")
