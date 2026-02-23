"""
PCA Visualization on Attribution Vectors

Uses ALL numeric attribution metrics as features to visualize task manifolds
in latent space. Optionally includes UMAP for comparison.

Metrics used:
  - n_active, n_edges
  - mean_influence, max_influence, top_100_concentration
  - mean_activation, max_activation
  - logit_entropy, max_logit_prob
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Style matching existing figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'cola': '#2ecc71',
    'winogrande': '#3498db',
    'snli': '#9b59b6',
    'hellaswag': '#e74c3c',
    'paws': '#e67e22',
}

LABELS = {
    'cola': 'CoLA (Grammar)',
    'winogrande': 'WinoGrande',
    'snli': 'SNLI (Inference)',
    'hellaswag': 'HellaSwag',
    'paws': 'PAWS (Paraphrase)',
}

DATA_PATH = Path('/Users/ariaxhan/Downloads/Vaults/CodingVault/neural-polygraph/data/results/domain_attribution_metrics.json')
OUTPUT_DIR = Path('/Users/ariaxhan/Downloads/Vaults/CodingVault/neural-polygraph/figures')


def load_domain_data():
    """Load domain attribution metrics."""
    with open(DATA_PATH) as f:
        return json.load(f)['samples']


def extract_features(samples, metrics):
    """Extract feature matrix and labels from samples."""
    X = []
    labels = []

    for s in samples:
        # Stack all numeric metrics
        features = [s[m] for m in metrics]
        X.append(features)
        labels.append(s.get('source', 'unknown'))

    return np.array(X), labels


def run_pca_analysis():
    """Run PCA on all attribution metrics and generate visualization."""

    print("Loading domain attribution data...")
    samples = load_domain_data()
    print(f"  Loaded {len(samples)} samples")

    # All numeric metrics (9 features)
    metrics = [
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

    print(f"\nFeature extraction:")
    print(f"  Metrics: {len(metrics)} features")
    print(f"  Features: {', '.join(metrics)}")

    # Extract features
    X, labels = extract_features(samples, metrics)
    print(f"  Matrix shape: {X.shape}")

    # Standardize (z-score normalization)
    print("\nStandardizing features (z-score)...")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Run PCA
    print("\nRunning PCA...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_std)

    # Report variance explained
    print("\nVariance explained by principal components:")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {var:.1%}")
    print(f"  Total (PC1+PC2+PC3): {pca.explained_variance_ratio_.sum():.1%}")

    # Generate scatter plot
    print("\nGenerating PCA scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    sources = ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']

    for src in sources:
        mask = np.array(labels) == src
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            alpha=0.6,
            label=LABELS.get(src, src),
            color=COLORS.get(src, '#333'),
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: Domain Clustering on Attribution Features\n(9 metrics, standardized)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'pca_domain_clustering.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")

    # Also save PDF
    output_pdf = OUTPUT_DIR / 'pca_domain_clustering.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved: {output_pdf}")

    plt.close()

    return pca, X_pca, labels


def run_umap_analysis(X_std, labels):
    """Optional: Run UMAP for comparison."""
    try:
        from umap import UMAP
    except ImportError:
        print("\nUMAP not available (install with: pip install umap-learn)")
        print("Skipping UMAP visualization.")
        return

    print("\nRunning UMAP...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = umap.fit_transform(X_std)

    print("Generating UMAP scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    sources = ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']

    for src in sources:
        mask = np.array(labels) == src
        ax.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            alpha=0.6,
            label=LABELS.get(src, src),
            color=COLORS.get(src, '#333'),
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP: Domain Clustering on Attribution Features\n(9 metrics, standardized)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'umap_domain_clustering.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"UMAP figure saved: {output_path}")

    output_pdf = OUTPUT_DIR / 'umap_domain_clustering.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"UMAP PDF saved: {output_pdf}")

    plt.close()


def main():
    """Run PCA visualization on attribution vectors."""
    print("=" * 60)
    print("PCA VISUALIZATION ON ATTRIBUTION VECTORS")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run PCA
    pca, X_pca, labels = run_pca_analysis()

    # Optionally run UMAP
    samples = load_domain_data()
    metrics = [
        'n_active', 'n_edges', 'mean_influence', 'max_influence',
        'top_100_concentration', 'mean_activation', 'max_activation',
        'logit_entropy', 'max_logit_prob'
    ]
    X, _ = extract_features(samples, metrics)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    run_umap_analysis(X_std, labels)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
