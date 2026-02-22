"""
Test: SAE vs Transcoder Feature Discrimination

Uses real benchmark data from deepset/prompt-injections dataset.
Compares which feature extractor better discriminates injection from benign.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from datasets import load_dataset
import time
import sys

sys.path.insert(0, "src")
from neural_polygraph.feature_extractors import (
    SAEExtractor,
    TranscoderExtractor,
)

# Config
N_INJECTION = 10
N_BENIGN = 10
LAYER = 5


def mean(vals):
    return sum(vals) / len(vals) if vals else 0


def std(vals):
    if len(vals) < 2:
        return 0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def extract_features(text, model, extractor):
    """Extract features from text using given extractor."""
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)

    # Verify hook point exists
    if extractor.hook_point not in cache:
        available = [k for k in cache.keys() if f"blocks.{extractor.layer}" in k]
        raise ValueError(
            f"Hook point {extractor.hook_point} not found. "
            f"Available for layer {extractor.layer}: {available}"
        )

    activations = cache[extractor.hook_point][0, -1, :]
    return extractor.encode(activations)


def compute_metrics(features1, features2):
    """Compute similarity metrics between two feature vectors."""
    cos_sim = F.cosine_similarity(
        features1.unsqueeze(0),
        features2.unsqueeze(0)
    ).item()

    l2_dist = torch.norm(features1 - features2).item()

    # Feature overlap
    active1 = features1 > 0
    active2 = features2 > 0
    intersection = (active1 & active2).sum().item()
    union = (active1 | active2).sum().item()
    overlap = intersection / union if union > 0 else 0

    return {
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
        "feature_overlap": overlap,
    }


def compute_discrimination(injection_vals, benign_vals):
    """Compute Cohen's d effect size."""
    inj_mean = mean(injection_vals)
    ben_mean = mean(benign_vals)
    inj_std = std(injection_vals)
    ben_std = std(benign_vals)

    gap = inj_mean - ben_mean
    pooled_std = ((inj_std**2 + ben_std**2) / 2) ** 0.5
    cohen_d = abs(gap) / pooled_std if pooled_std > 0 else 0

    return {
        "inj_mean": inj_mean,
        "ben_mean": ben_mean,
        "gap": gap,
        "cohen_d": cohen_d,
        "effect": "LARGE" if cohen_d > 0.8 else "MEDIUM" if cohen_d > 0.5 else "SMALL",
    }


def main():
    print("=" * 70)
    print("SAE vs TRANSCODER DISCRIMINATION TEST")
    print("Using: deepset/prompt-injections benchmark")
    print("=" * 70)
    print()

    # Load dataset
    print("Loading benchmark dataset...")
    ds = load_dataset("deepset/prompt-injections", split="train")

    injections = [x["text"] for x in ds if x["label"] == 1][:N_INJECTION]
    benigns = [x["text"] for x in ds if x["label"] == 0][:N_BENIGN]

    print(f"  Loaded {len(injections)} injection samples")
    print(f"  Loaded {len(benigns)} benign samples")
    print()

    # Show samples
    print("Sample INJECTIONS:")
    for i, text in enumerate(injections[:3]):
        print(f"  [{i+1}] {text[:60]}...")
    print()

    print("Sample BENIGN:")
    for i, text in enumerate(benigns[:3]):
        print(f"  [{i+1}] {text[:60]}...")
    print()

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model (Gemma-2-2B)...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load extractors
    print("\nLoading feature extractors...")

    print(f"  Loading SAE (layer {LAYER})...")
    t0 = time.time()
    sae_extractor = SAEExtractor(device=device, layer=LAYER)
    print(f"    Loaded in {time.time() - t0:.1f}s")

    print(f"  Loading Transcoder (layer {LAYER})...")
    t0 = time.time()
    try:
        tc_extractor = TranscoderExtractor(device=device, layer=LAYER)
        print(f"    Loaded in {time.time() - t0:.1f}s")
        has_transcoder = True
    except Exception as e:
        print(f"    Failed: {e}")
        tc_extractor = None
        has_transcoder = False
    print()

    # Create baseline from system prompt
    SYSTEM_PROMPT = "You are a helpful AI assistant."
    print("Extracting baseline features (system prompt)...")
    sae_baseline = extract_features(SYSTEM_PROMPT, model, sae_extractor)
    tc_baseline = None
    if has_transcoder:
        tc_baseline = extract_features(SYSTEM_PROMPT, model, tc_extractor)
    print()

    # Test all samples
    results = {"sae": {"injection": [], "benign": []}}
    if has_transcoder:
        results["tc"] = {"injection": [], "benign": []}

    print("Processing INJECTION samples...")
    for i, text in enumerate(injections):
        # SAE
        features = extract_features(text, model, sae_extractor)
        metrics = compute_metrics(sae_baseline, features)
        results["sae"]["injection"].append(metrics)

        # Transcoder
        if has_transcoder:
            features = extract_features(text, model, tc_extractor)
            metrics = compute_metrics(tc_baseline, features)
            results["tc"]["injection"].append(metrics)

        print(f"  [{i+1:2}/{len(injections)}] SAE sim={results['sae']['injection'][-1]['cosine_similarity']:.3f}", end="")
        if has_transcoder:
            print(f" | TC sim={results['tc']['injection'][-1]['cosine_similarity']:.3f}", end="")
        print()

    print()
    print("Processing BENIGN samples...")
    for i, text in enumerate(benigns):
        # SAE
        features = extract_features(text, model, sae_extractor)
        metrics = compute_metrics(sae_baseline, features)
        results["sae"]["benign"].append(metrics)

        # Transcoder
        if has_transcoder:
            features = extract_features(text, model, tc_extractor)
            metrics = compute_metrics(tc_baseline, features)
            results["tc"]["benign"].append(metrics)

        print(f"  [{i+1:2}/{len(benigns)}] SAE sim={results['sae']['benign'][-1]['cosine_similarity']:.3f}", end="")
        if has_transcoder:
            print(f" | TC sim={results['tc']['benign'][-1]['cosine_similarity']:.3f}", end="")
        print()

    # Analysis
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    for metric in ["cosine_similarity", "l2_distance", "feature_overlap"]:
        print(f"--- {metric.upper()} ---")
        print()

        # SAE
        sae_inj = [r[metric] for r in results["sae"]["injection"]]
        sae_ben = [r[metric] for r in results["sae"]["benign"]]
        sae_disc = compute_discrimination(sae_inj, sae_ben)

        print(f"  SAE:")
        print(f"    Injection: {sae_disc['inj_mean']:.4f} (±{std(sae_inj):.4f})")
        print(f"    Benign:    {sae_disc['ben_mean']:.4f} (±{std(sae_ben):.4f})")
        print(f"    Cohen's d: {sae_disc['cohen_d']:.3f} ({sae_disc['effect']})")
        print()

        # Transcoder
        if has_transcoder:
            tc_inj = [r[metric] for r in results["tc"]["injection"]]
            tc_ben = [r[metric] for r in results["tc"]["benign"]]
            tc_disc = compute_discrimination(tc_inj, tc_ben)

            print(f"  TRANSCODER:")
            print(f"    Injection: {tc_disc['inj_mean']:.4f} (±{std(tc_inj):.4f})")
            print(f"    Benign:    {tc_disc['ben_mean']:.4f} (±{std(tc_ben):.4f})")
            print(f"    Cohen's d: {tc_disc['cohen_d']:.3f} ({tc_disc['effect']})")
            print()

            # Winner
            if sae_disc["cohen_d"] > tc_disc["cohen_d"]:
                print(f"  → SAE wins on {metric}")
            elif tc_disc["cohen_d"] > sae_disc["cohen_d"]:
                print(f"  → TRANSCODER wins on {metric}")
            else:
                print(f"  → TIE on {metric}")
        print()

    # Final verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    sae_cos_d = compute_discrimination(
        [r["cosine_similarity"] for r in results["sae"]["injection"]],
        [r["cosine_similarity"] for r in results["sae"]["benign"]]
    )["cohen_d"]

    if has_transcoder:
        tc_cos_d = compute_discrimination(
            [r["cosine_similarity"] for r in results["tc"]["injection"]],
            [r["cosine_similarity"] for r in results["tc"]["benign"]]
        )["cohen_d"]

        print(f"Cosine similarity discrimination:")
        print(f"  SAE:        Cohen's d = {sae_cos_d:.3f}")
        print(f"  Transcoder: Cohen's d = {tc_cos_d:.3f}")
        print()

        if sae_cos_d > 0.5 or tc_cos_d > 0.5:
            winner = "SAE" if sae_cos_d > tc_cos_d else "TRANSCODER"
            print(f"✅ {winner} shows better discrimination")
        else:
            print("⚠️ Neither shows strong discrimination (Cohen's d < 0.5)")
    else:
        print(f"SAE cosine similarity: Cohen's d = {sae_cos_d:.3f}")
        if sae_cos_d > 0.5:
            print("✅ SAE shows discrimination (medium+ effect)")
        else:
            print("⚠️ SAE shows weak discrimination")


if __name__ == "__main__":
    main()
