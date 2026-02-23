"""
Domain Profile Similarity Analysis

Computes cosine similarity between domain metric profiles to identify
which domains share similar computational signatures. Uses standardized
metric vectors (z-scored) to ensure all metrics contribute equally.

Interpretation: High similarity indicates domains use features with similar
statistical properties, suggesting they operate in similar computational regimes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Style matching existing figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

LABELS = {
    'cola': 'CoLA\n(Grammar)',
    'winogrande': 'WinoGrande\n(Commonsense)',
    'snli': 'SNLI\n(Inference)',
    'hellaswag': 'HellaSwag\n(Situational)',
    'paws': 'PAWS\n(Paraphrase)',
}

METRICS = [
    'n_active',
    'n_edges',
    'mean_influence',
    'max_influence',
    'top_100_concentration',
    'mean_activation',
    'max_activation',
    'logit_entropy',
    'max_logit_prob'
]


def load_data():
    """Load domain attribution metrics."""
    with open('data/results/domain_attribution_metrics.json') as f:
        data = json.load(f)
    return data['samples']


def compute_domain_profiles(samples):
    """
    Compute mean metric vector for each domain.

    Returns:
        domain_profiles: dict mapping domain -> metric vector (9-dim)
        domain_order: list of domain names in consistent order
    """
    # Group by domain
    groups = {}
    for s in samples:
        domain = s.get('source', 'unknown')
        if domain not in groups:
            groups[domain] = []
        groups[domain].append(s)

    # Compute mean vector per domain
    domain_profiles = {}
    for domain, samps in groups.items():
        profile = np.array([
            np.mean([s[m] for s in samps]) for m in METRICS
        ])
        domain_profiles[domain] = profile

    # Consistent ordering
    domain_order = sorted(domain_profiles.keys())

    return domain_profiles, domain_order


def standardize_profiles(domain_profiles, domain_order):
    """
    Z-score standardization across domains for each metric.
    Ensures all metrics contribute equally to similarity.
    """
    # Stack profiles into matrix
    X = np.array([domain_profiles[d] for d in domain_order])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def compute_similarity_matrix(X_scaled):
    """
    Compute pairwise cosine similarity.

    Cosine similarity = dot(A, B) / (||A|| * ||B||)
    For standardized vectors, this measures alignment of metric patterns.
    """
    # Normalize rows to unit length
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    X_normalized = X_scaled / norms

    # Cosine similarity = dot product of normalized vectors
    similarity = X_normalized @ X_normalized.T

    return similarity


def plot_similarity_heatmap(similarity, domain_order, output_path):
    """Generate publication-quality similarity heatmap."""
    fig, ax = plt.subplots(figsize=(10, 9))

    # Labels with line breaks
    labels = [LABELS.get(d, d) for d in domain_order]

    # Heatmap
    sns.heatmap(
        similarity,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Cosine Similarity'},
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        'Domain Profile Similarity\n(Standardized Metric Vectors)',
        size=16,
        pad=20
    )

    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def print_similarity_analysis(similarity, domain_order):
    """Print similarity matrix with interpretation."""
    print("\n" + "="*70)
    print("DOMAIN PROFILE SIMILARITY MATRIX")
    print("="*70)
    print("\nCosine similarity of standardized metric vectors (9 metrics).")
    print("High similarity = domains use features with similar statistical properties.\n")

    # Print matrix
    header = "         " + "  ".join([f"{d[:4]:>6}" for d in domain_order])
    print(header)
    print("-" * len(header))

    for i, domain_i in enumerate(domain_order):
        row = f"{domain_i[:8]:>8} "
        row += "  ".join([f"{similarity[i, j]:>6.3f}" for j in range(len(domain_order))])
        print(row)

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Find most similar pairs (excluding diagonal)
    pairs = []
    n = len(domain_order)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((domain_order[i], domain_order[j], similarity[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost similar domain pairs:")
    for i, (d1, d2, sim) in enumerate(pairs[:3], 1):
        print(f"{i}. {d1} ↔ {d2}: {sim:.3f}")

    print("\nLeast similar domain pairs:")
    for i, (d1, d2, sim) in enumerate(pairs[-3:], 1):
        print(f"{i}. {d1} ↔ {d2}: {sim:.3f}")

    print("\n" + "="*70 + "\n")


def main():
    print("Computing domain profile similarity...")

    # Load and process data
    samples = load_data()
    print(f"Loaded {len(samples)} samples")

    domain_profiles, domain_order = compute_domain_profiles(samples)
    print(f"Computed profiles for {len(domain_order)} domains: {domain_order}")

    # Standardize
    X_scaled = standardize_profiles(domain_profiles, domain_order)
    print(f"Standardized metric vectors (shape: {X_scaled.shape})")

    # Compute similarity
    similarity = compute_similarity_matrix(X_scaled)

    # Output
    print_similarity_analysis(similarity, domain_order)

    output_path = Path('figures/domain_profile_similarity.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_similarity_heatmap(similarity, domain_order, output_path)

    print("Done.")


if __name__ == '__main__':
    main()
