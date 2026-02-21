#!/usr/bin/env python3
"""
Deep Analysis: The Geometry of Injection

Explores the "diffuse graph = injection" hypothesis through:
1. Entropy analysis of influence distributions
2. Per-sample visualization
3. Feature relationship exploration
4. Gradient analysis (what makes something "more injection-like")

Usage:
    python experiments/analyze_injection_geometry.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = Path(__file__).parent.parent / "pint_metrics.json"
with open(data_path) as f:
    data = json.load(f)

samples = data["samples"]
print(f"Loaded {len(samples)} samples")

# Separate by label
injections = [s for s in samples if s["label"]]
benigns = [s for s in samples if not s["label"]]
print(f"Injection: {len(injections)}, Benign: {len(benigns)}")

# =============================================================================
# ANALYSIS 1: Distribution Deep Dive
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: DISTRIBUTION STATISTICS")
print("="*70)

metrics = ["n_active", "top_100_concentration", "mean_influence", "n_edges"]

for metric in metrics:
    inj_vals = np.array([s.get(metric, 0) for s in injections])
    ben_vals = np.array([s.get(metric, 0) for s in benigns])

    # Compute statistics
    inj_mean, inj_std = inj_vals.mean(), inj_vals.std()
    ben_mean, ben_std = ben_vals.mean(), ben_vals.std()

    # Effect size
    pooled_std = np.sqrt((inj_std**2 + ben_std**2) / 2)
    cohen_d = abs(inj_mean - ben_mean) / pooled_std if pooled_std > 0 else 0

    # Separation
    # What % of injections are above/below median benign?
    ben_median = np.median(ben_vals)
    if metric in ["top_100_concentration", "mean_influence"]:
        # Injection should be LOWER
        pct_separated = (inj_vals < ben_median).mean() * 100
        direction = "lower"
    else:
        # Injection should be HIGHER
        pct_separated = (inj_vals > ben_median).mean() * 100
        direction = "higher"

    print(f"\n{metric.upper()}:")
    print(f"  Injection: {inj_mean:.6f} ± {inj_std:.6f}")
    print(f"  Benign:    {ben_mean:.6f} ± {ben_std:.6f}")
    print(f"  Cohen's d: {cohen_d:.3f}")
    print(f"  Separation: {pct_separated:.1f}% of injections are {direction} than benign median")

# =============================================================================
# ANALYSIS 2: Derived Metrics (Entropy-like)
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: DERIVED METRICS")
print("="*70)

def compute_derived_metrics(sample):
    """Compute derived metrics that might be more discriminative."""
    n_active = sample.get("n_active", 1)
    n_edges = sample.get("n_edges", 1)
    mean_inf = sample.get("mean_influence", 0)
    top_100 = sample.get("top_100_concentration", 0)

    # Density: edges per feature pair
    density = n_edges / (n_active ** 2) if n_active > 0 else 0

    # Influence spread: inverse of concentration
    spread = 1 / (top_100 + 1e-10)

    # Total influence (proxy for "energy")
    total_influence = n_edges * mean_inf

    # Influence per feature
    influence_per_feature = total_influence / n_active if n_active > 0 else 0

    return {
        "density": density,
        "spread": spread,
        "total_influence": total_influence,
        "influence_per_feature": influence_per_feature,
    }

# Compute for all
for s in samples:
    s.update(compute_derived_metrics(s))

derived_metrics = ["density", "spread", "total_influence", "influence_per_feature"]

for metric in derived_metrics:
    inj_vals = np.array([s[metric] for s in injections])
    ben_vals = np.array([s[metric] for s in benigns])

    inj_mean, inj_std = inj_vals.mean(), inj_vals.std()
    ben_mean, ben_std = ben_vals.mean(), ben_vals.std()

    pooled_std = np.sqrt((inj_std**2 + ben_std**2) / 2)
    cohen_d = abs(inj_mean - ben_mean) / pooled_std if pooled_std > 0 else 0

    print(f"\n{metric.upper()}:")
    print(f"  Injection: {inj_mean:.6f} ± {inj_std:.6f}")
    print(f"  Benign:    {ben_mean:.6f} ± {ben_std:.6f}")
    print(f"  Cohen's d: {cohen_d:.3f}")

# =============================================================================
# ANALYSIS 3: Best/Worst Cases
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: EXTREME CASES")
print("="*70)

# Most "injection-like" benign (high n_active, low concentration)
benign_scores = []
for s in benigns:
    score = s["n_active"] / 10000 - s["top_100_concentration"] * 100
    benign_scores.append((score, s))

benign_scores.sort(reverse=True)
print("\nMost injection-like BENIGN samples:")
for score, s in benign_scores[:3]:
    print(f"  [{score:.2f}] {s['text'][:60]}...")
    print(f"       n_active={s['n_active']}, conc={s['top_100_concentration']:.4f}")

# Most "benign-like" injection (low n_active, high concentration)
injection_scores = []
for s in injections:
    score = s["n_active"] / 10000 - s["top_100_concentration"] * 100
    injection_scores.append((score, s))

injection_scores.sort()
print("\nMost benign-like INJECTION samples:")
for score, s in injection_scores[:3]:
    print(f"  [{score:.2f}] {s['text'][:60]}...")
    print(f"       n_active={s['n_active']}, conc={s['top_100_concentration']:.4f}")

# =============================================================================
# ANALYSIS 4: Correlation Structure
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 4: METRIC CORRELATIONS")
print("="*70)

all_metrics = metrics + derived_metrics
n_metrics = len(all_metrics)

# Compute correlation matrix
corr_matrix = np.zeros((n_metrics, n_metrics))
for i, m1 in enumerate(all_metrics):
    for j, m2 in enumerate(all_metrics):
        vals1 = np.array([s.get(m1, 0) for s in samples])
        vals2 = np.array([s.get(m2, 0) for s in samples])
        corr_matrix[i, j] = np.corrcoef(vals1, vals2)[0, 1]

print("\nCorrelation matrix:")
print("                    ", "  ".join([m[:8] for m in all_metrics]))
for i, m in enumerate(all_metrics):
    row = " ".join([f"{corr_matrix[i,j]:+.2f}" for j in range(n_metrics)])
    print(f"{m:20s} {row}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig_dir = Path(__file__).parent / "08_injection_detection" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: 2D scatter (n_active vs concentration)
fig, ax = plt.subplots(figsize=(10, 8))

inj_x = [s["n_active"] for s in injections]
inj_y = [s["top_100_concentration"] for s in injections]
ben_x = [s["n_active"] for s in benigns]
ben_y = [s["top_100_concentration"] for s in benigns]

ax.scatter(ben_x, ben_y, c='green', alpha=0.6, label='Benign', s=50)
ax.scatter(inj_x, inj_y, c='red', alpha=0.8, label='Injection', s=80, marker='x')

ax.set_xlabel('N Active Features', fontsize=12)
ax.set_ylabel('Top-100 Concentration', fontsize=12)
ax.set_title('Injection Geometry: Features vs Concentration', fontsize=14)
ax.legend()

# Add decision boundary approximation
ax.axhline(y=0.003, color='blue', linestyle='--', alpha=0.5, label='Approx boundary')
ax.axvline(x=20000, color='blue', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(fig_dir / "geometry_scatter.png", dpi=150)
plt.close()
print(f"✓ Saved: geometry_scatter.png")

# Figure 2: Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    inj_vals = [s.get(metric, 0) for s in injections]
    ben_vals = [s.get(metric, 0) for s in benigns]

    ax.hist(ben_vals, bins=20, alpha=0.6, label='Benign', color='green', density=True)
    ax.hist(inj_vals, bins=20, alpha=0.6, label='Injection', color='red', density=True)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Metric Distributions: Injection vs Benign', fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "distributions.png", dpi=150)
plt.close()
print(f"✓ Saved: distributions.png")

# Figure 3: Per-sample profile
fig, ax = plt.subplots(figsize=(14, 6))

# Normalize metrics to [0, 1] for comparison
def normalize(vals):
    vals = np.array(vals)
    return (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

sample_ids = list(range(len(samples)))
colors = ['red' if s['label'] else 'green' for s in samples]

n_active_norm = normalize([s['n_active'] for s in samples])
conc_norm = normalize([s['top_100_concentration'] for s in samples])
# Invert concentration so higher = more injection-like
conc_inv = 1 - conc_norm

# Combined score
combined = (n_active_norm + conc_inv) / 2

ax.bar(sample_ids, combined, color=colors, alpha=0.7)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Injection-likeness Score')
ax.set_title('Per-Sample Injection Score (Red=Injection, Green=Benign)')

plt.tight_layout()
plt.savefig(fig_dir / "per_sample_scores.png", dpi=150)
plt.close()
print(f"✓ Saved: per_sample_scores.png")

# Figure 4: 4-metric radar chart for average profiles
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Normalize metrics
def get_profile(samples_list):
    return [
        np.mean([s['n_active'] for s in samples_list]) / 30000,
        np.mean([s['n_edges'] for s in samples_list]) / 60000000,
        1 - np.mean([s['top_100_concentration'] for s in samples_list]) / 0.01,
        1 - np.mean([s['mean_influence'] for s in samples_list]) / 0.015,
    ]

inj_profile = get_profile(injections)
ben_profile = get_profile(benigns)

labels = ['N Active (norm)', 'N Edges (norm)', 'Diffusion (1-conc)', 'Weakness (1-inf)']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # complete the loop

inj_profile += inj_profile[:1]
ben_profile += ben_profile[:1]

ax.plot(angles, inj_profile, 'r-', linewidth=2, label='Injection')
ax.fill(angles, inj_profile, 'r', alpha=0.25)
ax.plot(angles, ben_profile, 'g-', linewidth=2, label='Benign')
ax.fill(angles, ben_profile, 'g', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title('Injection vs Benign: Average Profile', fontsize=14)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(fig_dir / "radar_profile.png", dpi=150)
plt.close()
print(f"✓ Saved: radar_profile.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nFigures saved to: {fig_dir}")
