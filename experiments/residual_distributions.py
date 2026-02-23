"""
Residual Distribution Visualization (KDE)

Visualizes distributions of length-controlled (residualized) metrics across domains.
Shows domain separation after regressing out text length confounds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Publication style
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

OUTPUT_DIR = Path('figures')


def load_data():
    """Load domain attribution metrics."""
    with open('data/results/domain_attribution_metrics.json') as f:
        return json.load(f)['samples']


def residualize(y, x):
    """Regress y on x, return residuals (length-controlled)."""
    coef = np.polyfit(x, y, 1)
    predicted = np.polyval(coef, x)
    return y - predicted


def add_residuals(samples):
    """Add length-controlled residual metrics to samples."""
    lengths = np.array([len(s.get('text', '')) for s in samples])

    metrics = ['mean_influence', 'top_100_concentration']

    for m in metrics:
        vals = np.array([s[m] for s in samples])
        resid = residualize(vals, lengths)
        for i, s in enumerate(samples):
            s[f'{m}_resid'] = resid[i]

    return samples


def group_by_source(samples):
    """Group samples by domain source."""
    groups = {}
    for s in samples:
        src = s.get('source', 'unknown')
        if src not in groups:
            groups[src] = []
        groups[src].append(s)
    return groups


def plot_kde_metric(groups, metric, title, output_path):
    """
    Plot KDE distributions for a single metric across all domains.

    Args:
        groups: Dict mapping source to list of samples
        metric: Metric name (e.g., 'mean_influence_resid')
        title: Plot title
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sources = ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']

    for src in sources:
        if src not in groups:
            continue

        vals = [s[metric] for s in groups[src]]

        # Plot KDE with seaborn
        sns.kdeplot(
            vals,
            ax=ax,
            color=COLORS[src],
            label=LABELS[src],
            linewidth=2.5,
            fill=True,
            alpha=0.3
        )

    ax.set_xlabel('Residual Value (length-controlled)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {output_path.name}")
    plt.close()


def main():
    print("Generating residual distribution KDE plots...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    samples = load_data()
    samples = add_residuals(samples)
    groups = group_by_source(samples)

    print(f"Loaded {len(samples)} samples across {len(groups)} domains")
    print(f"Domains: {list(groups.keys())}\n")

    # Generate KDE plots
    print("Generating figures:")

    plot_kde_metric(
        groups,
        'mean_influence_resid',
        'Influence: Domain Separation (Length-Controlled)',
        OUTPUT_DIR / 'residual_kde_influence.png'
    )

    plot_kde_metric(
        groups,
        'top_100_concentration_resid',
        'Concentration: Domain Separation (Length-Controlled)',
        OUTPUT_DIR / 'residual_kde_concentration.png'
    )

    print("\nDone. Figures show clear separation between domains after length control.")
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
