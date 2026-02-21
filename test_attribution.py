"""
Test: Attribution Graph Analysis for Injection Detection

Instead of comparing feature similarity, we trace CAUSAL INFLUENCE:
- Which features are driving the model's output?
- Do injection prompts show different influence patterns?

Uses Anthropic's circuit-tracer library.
"""

import torch
from circuit_tracer import ReplacementModel, attribute, Graph
from datasets import load_dataset
import json

# Config
N_SAMPLES = 3  # Start very small - attribution is expensive
MODEL = "google/gemma-2-2b"
TRANSCODER_SET = "gemma"  # GemmaScope transcoders


def analyze_prompt(replacement_model, prompt: str, label: str):
    """
    Generate attribution graph for a prompt.
    Returns summary statistics about feature influence.
    """
    print(f"\n[{label}] Analyzing: {prompt[:50]}...")

    # Run attribution
    try:
        graph = attribute(prompt, replacement_model, verbose=True)
    except Exception as e:
        print(f"  Attribution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Extract key metrics from the attribution graph
    # Graph has: feature_acts (activations), adj (adjacency matrix), logit_effects

    # Get feature activations and their effects on logits
    feature_acts = graph.feature_acts  # Tensor of feature activations
    adj = graph.adj  # Adjacency matrix (influence between features)
    logit_effects = graph.logit_effects  # Effect of each feature on output logits

    # Compute metrics
    n_active = (feature_acts.abs() > 0.01).sum().item()

    # Total influence = sum of absolute logit effects
    total_logit_effect = logit_effects.abs().sum().item()

    # Concentration: what fraction of effect comes from top features?
    sorted_effects, _ = logit_effects.abs().flatten().sort(descending=True)
    top_10_effect = sorted_effects[:10].sum().item()
    concentration = top_10_effect / total_logit_effect if total_logit_effect > 0 else 0

    # Edge density in adjacency (how interconnected are features?)
    n_edges = (adj.abs() > 0.01).sum().item()
    max_edges = adj.numel()
    edge_density = n_edges / max_edges if max_edges > 0 else 0

    # Mean and max influence strength
    mean_influence = adj.abs().mean().item()
    max_influence = adj.abs().max().item()

    result = {
        'prompt': prompt[:100],
        'label': label,
        'n_active_features': n_active,
        'total_logit_effect': total_logit_effect,
        'top_10_concentration': concentration,
        'edge_density': edge_density,
        'mean_influence': mean_influence,
        'max_influence': max_influence,
    }

    print(f"  Active features: {result['n_active_features']}")
    print(f"  Top-10 concentration: {result['top_10_concentration']:.3f}")
    print(f"  Edge density: {result['edge_density']:.6f}")
    print(f"  Max influence: {result['max_influence']:.4f}")

    return result


def main():
    print("=" * 70)
    print("ATTRIBUTION GRAPH INJECTION ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\nLoading benchmark dataset...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    injections = [x["text"] for x in ds if x["label"] == 1][:N_SAMPLES]
    benigns = [x["text"] for x in ds if x["label"] == 0][:N_SAMPLES]

    print(f"Testing {len(injections)} injections + {len(benigns)} benign")

    # Initialize replacement model
    print(f"\nInitializing ReplacementModel with {MODEL}...")
    print("(This loads model + transcoders - may take a few minutes)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # MPS not well supported for this library
    print(f"Device: {device}")

    replacement_model = ReplacementModel.from_pretrained(
        MODEL,
        TRANSCODER_SET,
        dtype=torch.bfloat16,
        device=torch.device(device),
        backend="transformerlens",
    )
    print("Model ready!")

    # Analyze injections
    print("\n" + "=" * 70)
    print("ANALYZING INJECTION PROMPTS")
    print("=" * 70)

    injection_results = []
    for prompt in injections:
        result = analyze_prompt(replacement_model, prompt, "INJECTION")
        if result:
            injection_results.append(result)

    # Analyze benign
    print("\n" + "=" * 70)
    print("ANALYZING BENIGN PROMPTS")
    print("=" * 70)

    benign_results = []
    for prompt in benigns:
        result = analyze_prompt(replacement_model, prompt, "BENIGN")
        if result:
            benign_results.append(result)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    if injection_results and benign_results:
        inj_concentration = mean([r['top_10_concentration'] for r in injection_results])
        ben_concentration = mean([r['top_10_concentration'] for r in benign_results])

        inj_n_features = mean([r['n_active_features'] for r in injection_results])
        ben_n_features = mean([r['n_active_features'] for r in benign_results])

        print(f"\nTop-10 Influence Concentration:")
        print(f"  Injection: {inj_concentration:.3f}")
        print(f"  Benign:    {ben_concentration:.3f}")
        print(f"  Gap:       {inj_concentration - ben_concentration:+.3f}")

        print(f"\nActive Features Count:")
        print(f"  Injection: {inj_n_features:.1f}")
        print(f"  Benign:    {ben_n_features:.1f}")

        # Hypothesis: Injection might show MORE concentrated influence
        # (fewer features dominating) or DIFFERENT feature types
        if inj_concentration > ben_concentration + 0.05:
            print("\n✅ Injection shows MORE concentrated influence")
        elif ben_concentration > inj_concentration + 0.05:
            print("\n⚠️ Benign shows more concentrated influence (unexpected)")
        else:
            print("\n⚠️ No clear difference in concentration")

    # Save detailed results
    output_file = "attribution_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'injection': injection_results,
            'benign': benign_results,
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
