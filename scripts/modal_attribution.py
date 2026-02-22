"""
Attribution Graph Analysis on Modal GPU

Traces causal influence instead of comparing feature similarity.

Usage:
    modal run modal_attribution.py
"""

import modal

# Image with circuit-tracer
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Need git for pip install from GitHub
    .pip_install(
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.20.0",
    )
    .pip_install(
        "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git",
    )
)

app = modal.App("attribution-analysis", image=image)
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="A100",  # More VRAM for attribution
    volumes={"/root/.cache/huggingface": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,  # 30 min for attribution
)
def run_attribution_analysis(n_samples: int = 3):
    """Run attribution graph analysis comparing injection vs benign."""
    import os
    import torch
    from huggingface_hub import login
    from datasets import load_dataset

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[setup] HuggingFace authenticated")

    # Import circuit_tracer after installing
    from circuit_tracer import ReplacementModel, attribute

    print("=" * 70)
    print("ATTRIBUTION GRAPH INJECTION ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\nLoading benchmark dataset...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    injections = [x["text"] for x in ds if x["label"] == 1][:n_samples]
    benigns = [x["text"] for x in ds if x["label"] == 0][:n_samples]
    print(f"Testing {len(injections)} injections + {len(benigns)} benign")

    # Initialize model
    print("\nInitializing ReplacementModel...")
    print("(Loading model + transcoders)")

    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        backend="transformerlens",
    )
    print("Model ready!")

    def analyze_prompt(prompt: str, label: str):
        """Analyze a single prompt."""
        print(f"\n[{label}] {prompt[:50]}...")

        try:
            graph = attribute(prompt, model, verbose=False)
        except Exception as e:
            print(f"  Failed: {e}")
            return None

        # Extract metrics from graph
        # The Graph object has various attributes - let's explore
        print(f"  Graph attributes: {[a for a in dir(graph) if not a.startswith('_')]}")

        # Basic info
        result = {
            'prompt': prompt[:80],
            'label': label,
        }

        # Active features count
        if hasattr(graph, 'active_features'):
            result['n_active'] = len(graph.active_features)

        # Feature activations
        if hasattr(graph, 'activation_values'):
            acts = graph.activation_values
            if hasattr(acts, 'abs'):
                result['mean_activation'] = acts.abs().mean().item()
                result['max_activation'] = acts.abs().max().item()
                result['activation_std'] = acts.std().item()

        # Adjacency matrix (causal influence between features)
        if hasattr(graph, 'adjacency_matrix'):
            adj = graph.adjacency_matrix
            if hasattr(adj, 'abs'):
                n_edges = (adj.abs() > 0.01).sum().item()
                result['n_edges'] = n_edges
                result['mean_influence'] = adj.abs().mean().item()
                result['max_influence'] = adj.abs().max().item()
                result['influence_std'] = adj.std().item()

                # Concentration: how much of total influence is in top edges?
                flat = adj.abs().flatten()
                sorted_inf, _ = flat.sort(descending=True)
                total = flat.sum().item()
                top_100 = sorted_inf[:100].sum().item()
                result['top_100_concentration'] = top_100 / (total + 1e-10)

        # Logit probabilities
        if hasattr(graph, 'logit_probabilities'):
            probs = graph.logit_probabilities
            if hasattr(probs, 'max'):
                result['max_logit_prob'] = probs.max().item()
                result['logit_entropy'] = -(probs * (probs + 1e-10).log()).sum().item()

        print(f"  Results: {result}")
        return result

    # Analyze prompts
    print("\n" + "=" * 70)
    print("ANALYZING INJECTION PROMPTS")
    print("=" * 70)

    injection_results = []
    for prompt in injections:
        r = analyze_prompt(prompt, "INJECTION")
        if r:
            injection_results.append(r)

    print("\n" + "=" * 70)
    print("ANALYZING BENIGN PROMPTS")
    print("=" * 70)

    benign_results = []
    for prompt in benigns:
        r = analyze_prompt(prompt, "BENIGN")
        if r:
            benign_results.append(r)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    def std(vals):
        if len(vals) < 2:
            return 0
        m = mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    metrics = ['top_100_concentration', 'n_active', 'max_influence', 'n_edges', 'mean_influence', 'logit_entropy']

    for metric in metrics:
        inj_vals = [r.get(metric, 0) for r in injection_results if isinstance(r.get(metric), (int, float))]
        ben_vals = [r.get(metric, 0) for r in benign_results if isinstance(r.get(metric), (int, float))]

        if inj_vals and ben_vals:
            inj_m, ben_m = mean(inj_vals), mean(ben_vals)
            gap = inj_m - ben_m
            pooled_std = ((std(inj_vals)**2 + std(ben_vals)**2) / 2) ** 0.5
            cohen_d = abs(gap) / pooled_std if pooled_std > 0 else 0

            print(f"\n{metric.upper()}:")
            print(f"  Injection: {inj_m:.4f} (±{std(inj_vals):.4f})")
            print(f"  Benign:    {ben_m:.4f} (±{std(ben_vals):.4f})")
            print(f"  Gap:       {gap:+.4f}")
            print(f"  Cohen's d: {cohen_d:.3f}")

            if cohen_d > 0.8:
                print(f"  Effect: LARGE ✅")
            elif cohen_d > 0.5:
                print(f"  Effect: MEDIUM")
            else:
                print(f"  Effect: SMALL ⚠️")

    return {
        'injection': injection_results,
        'benign': benign_results,
    }


@app.local_entrypoint()
def main():
    result = run_attribution_analysis.remote(n_samples=3)
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
