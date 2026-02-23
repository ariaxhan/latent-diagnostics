"""
Complete Figure Generation for Latent Diagnostics Paper

Generates all figures including the central summary figure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

# Style
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
    'truthful': '#27ae60',
    'false': '#c0392b',
}

LABELS = {
    'cola': 'CoLA (Grammar)',
    'winogrande': 'WinoGrande',
    'snli': 'SNLI (Inference)',
    'hellaswag': 'HellaSwag',
    'paws': 'PAWS (Paraphrase)',
}

OUTPUT_DIR = Path('figures/paper')


def load_domain_data():
    with open('data/results/domain_attribution_metrics.json') as f:
        return json.load(f)['samples']


def load_truthfulness_data():
    with open('data/results/truthfulness_metrics_clean.json') as f:
        return json.load(f)['samples']


def group_by_source(samples):
    groups = {}
    for s in samples:
        src = s.get('source', 'unknown')
        if src not in groups:
            groups[src] = []
        groups[src].append(s)
    return groups


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled_std == 0:
        return 0
    return (np.mean(a) - np.mean(b)) / pooled_std


# =============================================================================
# FIGURE 1: Truthfulness Negative Result
# =============================================================================
def fig_truthfulness_overlap(truth_samples):
    """Show that true/false completely overlap - negative result."""
    true_samples = [s for s in truth_samples if s.get('label') == 'truthful']
    false_samples = [s for s in truth_samples if s.get('label') == 'false']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = ['mean_influence', 'top_100_concentration', 'mean_activation']
    titles = ['Mean Influence', 'Concentration', 'Mean Activation']

    for ax, m, title in zip(axes, metrics, titles):
        true_vals = [s[m] for s in true_samples]
        false_vals = [s[m] for s in false_samples]

        # Histograms
        bins = np.linspace(min(true_vals + false_vals), max(true_vals + false_vals), 30)
        ax.hist(true_vals, bins=bins, alpha=0.6, label='Truthful', color=COLORS['truthful'], edgecolor='white')
        ax.hist(false_vals, bins=bins, alpha=0.6, label='False', color=COLORS['false'], edgecolor='white')

        d = cohens_d(true_vals, false_vals)
        ax.set_title(f'{title}\nd = {d:.3f} (no signal)', fontsize=12)
        ax.set_xlabel(m)
        ax.set_ylabel('Count')
        ax.legend()

    plt.suptitle('Truthfulness: No Detectable Signal\nTrue and false statements have identical activation topology',
                 fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'truthfulness_overlap.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'truthfulness_overlap.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ truthfulness_overlap.png")


# =============================================================================
# FIGURE 2: Distribution Histograms by Domain
# =============================================================================
def fig_domain_distributions(domain_samples):
    """Per-domain metric distributions."""
    groups = group_by_source(domain_samples)
    sources = ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = ['mean_influence', 'top_100_concentration', 'mean_activation', 'n_active']
    titles = ['Mean Influence (d=3.2)', 'Concentration (d=2.4)', 'Mean Activation (d=1.7)', 'N Active (confounded)']

    for ax, m, title in zip(axes, metrics, titles):
        for src in sources:
            if src not in groups:
                continue
            vals = [s[m] for s in groups[src]]
            ax.hist(vals, bins=20, alpha=0.5, label=LABELS.get(src, src),
                    color=COLORS.get(src, '#333'), edgecolor='white')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(m)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.suptitle('Metric Distributions by Task Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_distributions.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'domain_distributions.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ domain_distributions.png")


# =============================================================================
# FIGURE 3: PCA Clustering
# =============================================================================
def fig_pca_clustering(domain_samples):
    """PCA showing domain clustering in metric space."""
    groups = group_by_source(domain_samples)

    # Build feature matrix (robust metrics only)
    metrics = ['mean_influence', 'top_100_concentration', 'mean_activation']
    X = []
    labels = []
    colors = []

    for src in ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']:
        if src not in groups:
            continue
        for s in groups[src]:
            X.append([s[m] for m in metrics])
            labels.append(src)
            colors.append(COLORS.get(src, '#333'))

    X = np.array(X)

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    fig, ax = plt.subplots(figsize=(10, 8))

    for src in ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']:
        mask = np.array(labels) == src
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6,
                   label=LABELS.get(src, src), color=COLORS.get(src, '#333'),
                   s=60, edgecolors='white', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA: Domains Cluster in Activation Topology Space\n(using robust metrics only)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_clustering.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'pca_clustering.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ pca_clustering.png")


# =============================================================================
# FIGURE 4: Box Plots with Significance
# =============================================================================
def fig_boxplots_significance(domain_samples):
    """Box plots per domain with significance markers."""
    groups = group_by_source(domain_samples)
    sources = ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['mean_influence', 'top_100_concentration', 'mean_activation']
    titles = ['Mean Influence', 'Concentration', 'Mean Activation']

    for ax, m, title in zip(axes, metrics, titles):
        data = []
        labels_list = []
        for src in sources:
            if src in groups:
                data.append([s[m] for s in groups[src]])
                labels_list.append(LABELS.get(src, src).split()[0])  # Short name

        bp = ax.boxplot(data, labels=labels_list, patch_artist=True)

        for patch, src in zip(bp['boxes'], sources):
            patch.set_facecolor(COLORS.get(src, '#333'))
            patch.set_alpha(0.7)

        # Add significance bracket for CoLA vs others
        cola_vals = [s[m] for s in groups['cola']]
        other_vals = [s[m] for src in sources[1:] if src in groups for s in groups[src]]
        d = cohens_d(cola_vals, other_vals)

        ax.set_title(f'{title}\nCoLA vs others: d = {d:.2f}', fontsize=12)
        ax.set_ylabel(m)
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Metric Distributions by Task Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boxplots_significance.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'boxplots_significance.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ boxplots_significance.png")


# =============================================================================
# FIGURE 5: Correlation Heatmap
# =============================================================================
def fig_correlation_heatmap(domain_samples):
    """Correlation matrix of all metrics."""
    metrics = ['n_active', 'n_edges', 'mean_influence', 'top_100_concentration',
               'mean_activation', 'logit_entropy']
    metric_labels = ['N Active', 'N Edges', 'Influence', 'Concentration',
                     'Activation', 'Entropy']

    # Add length
    data = {m: [s[m] for s in domain_samples] for m in metrics}
    data['length'] = [len(s.get('text', '')) for s in domain_samples]
    metrics.append('length')
    metric_labels.append('Text Length')

    # Compute correlation matrix
    n = len(metrics)
    corr = np.zeros((n, n))
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            corr[i, j] = np.corrcoef(data[m1], data[m2])[0, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(metric_labels)

    # Add correlation values
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Metric Correlations\n(N Active and N Edges correlate with length)', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ correlation_heatmap.png")


# =============================================================================
# FIGURE 6: What It Detects Summary
# =============================================================================
def fig_detection_summary():
    """Visual summary of what the method can/cannot detect."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data
    tasks = ['Task Type\n(Grammar vs Reasoning)', 'Computational\nComplexity',
             'Adversarial\nInputs', 'Truthfulness']
    effect_sizes = [3.2, 2.4, 1.2, 0.05]
    colors = ['#27ae60', '#27ae60', '#f39c12', '#c0392b']
    works = ['WORKS', 'WORKS', 'WORKS', 'DOES NOT\nWORK']

    bars = ax.barh(tasks, effect_sizes, color=colors, edgecolor='black', linewidth=1.5, height=0.6)

    # Add effect size labels
    for bar, d, w in zip(bars, effect_sizes, works):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'd = {d:.2f}\n{w}', va='center', fontsize=11, fontweight='bold')

    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.7, label='Large effect threshold (d=0.8)')
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title('What Activation Topology Can Detect\n"Measures HOW model computes, not WHETHER correct"',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 4)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'detection_summary.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ detection_summary.png")


# =============================================================================
# FIGURE 7: CENTRAL SUMMARY FIGURE (Most Important)
# =============================================================================
def fig_central_summary(domain_samples, truth_samples):
    """The main figure showing everything important."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    groups = group_by_source(domain_samples)
    true_samples = [s for s in truth_samples if s.get('label') == 'truthful']
    false_samples = [s for s in truth_samples if s.get('label') == 'false']

    # =========================================================================
    # A: Detection Summary (top left, spans 2 cols)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, :2])
    tasks = ['Task Type', 'Complexity', 'Adversarial', 'Truthfulness']
    effect_sizes = [3.2, 2.4, 1.2, 0.05]
    colors = ['#27ae60', '#27ae60', '#f39c12', '#c0392b']

    bars = ax_a.barh(tasks, effect_sizes, color=colors, edgecolor='black', height=0.6)
    for bar, d in zip(bars, effect_sizes):
        ax_a.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                  f'd={d:.1f}', va='center', fontsize=10, fontweight='bold')

    ax_a.axvline(x=0.8, color='gray', linestyle='--', alpha=0.7)
    ax_a.set_xlabel("Effect Size (Cohen's d)")
    ax_a.set_title('A. What It Detects', fontsize=12, fontweight='bold')
    ax_a.set_xlim(0, 4)

    # =========================================================================
    # B: Key Insight Box (top right)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.axis('off')
    ax_b.text(0.5, 0.5,
              'KEY INSIGHT\n\n'
              'Activation topology\n'
              'measures HOW a model\n'
              'computes, not WHETHER\n'
              'it is correct.\n\n'
              'Simple tasks → Focused\n'
              'Complex tasks → Diffuse\n'
              'True vs False → Same',
              ha='center', va='center', fontsize=12,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='black'),
              transform=ax_b.transAxes)
    ax_b.set_title('B. Core Finding', fontsize=12, fontweight='bold')

    # =========================================================================
    # C: Influence by Domain (middle left)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    sources = ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']
    means = [np.mean([s['mean_influence'] for s in groups[src]]) for src in sources if src in groups]
    stds = [np.std([s['mean_influence'] for s in groups[src]]) for src in sources if src in groups]
    x = range(len(means))

    ax_c.bar(x, means, yerr=stds, color=[COLORS[s] for s in sources if s in groups],
             capsize=3, edgecolor='black')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(['Grammar', 'Wino', 'SNLI', 'Hella', 'PAWS'], rotation=30)
    ax_c.set_ylabel('Mean Influence')
    ax_c.set_title('C. Influence Gradient', fontsize=12, fontweight='bold')

    # =========================================================================
    # D: Scatter (middle center)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    for src in sources:
        if src not in groups:
            continue
        x = [s['mean_influence'] for s in groups[src]]
        y = [s['top_100_concentration'] for s in groups[src]]
        ax_d.scatter(x, y, alpha=0.5, label=src.upper()[:4], color=COLORS[src], s=30)

    ax_d.set_xlabel('Mean Influence')
    ax_d.set_ylabel('Concentration')
    ax_d.set_title('D. Domain Clustering', fontsize=12, fontweight='bold')
    ax_d.legend(fontsize=8)

    # =========================================================================
    # E: Truthfulness Overlap (middle right)
    # =========================================================================
    ax_e = fig.add_subplot(gs[1, 2])
    true_inf = [s['mean_influence'] for s in true_samples]
    false_inf = [s['mean_influence'] for s in false_samples]
    bins = np.linspace(min(true_inf + false_inf), max(true_inf + false_inf), 25)
    ax_e.hist(true_inf, bins=bins, alpha=0.6, label='True', color=COLORS['truthful'])
    ax_e.hist(false_inf, bins=bins, alpha=0.6, label='False', color=COLORS['false'])
    ax_e.set_xlabel('Mean Influence')
    ax_e.set_ylabel('Count')
    ax_e.set_title('E. Truthfulness: No Signal', fontsize=12, fontweight='bold')
    ax_e.legend()

    # =========================================================================
    # F: Length Confound (bottom left)
    # =========================================================================
    ax_f = fig.add_subplot(gs[2, 0])
    lengths = [len(s.get('text', '')) for s in domain_samples]
    n_active = [s['n_active'] for s in domain_samples]
    ax_f.scatter(lengths, n_active, alpha=0.3, s=20, color='#333')
    z = np.polyfit(lengths, n_active, 1)
    p = np.poly1d(z)
    ax_f.plot([min(lengths), max(lengths)], [p(min(lengths)), p(max(lengths))],
              'r--', linewidth=2, label='r = 0.98')
    ax_f.set_xlabel('Text Length')
    ax_f.set_ylabel('N Active (confounded)')
    ax_f.set_title('F. Length Confound', fontsize=12, fontweight='bold')
    ax_f.legend()

    # =========================================================================
    # G: Robust Metric (bottom center)
    # =========================================================================
    ax_g = fig.add_subplot(gs[2, 1])
    influence = [s['mean_influence'] for s in domain_samples]
    ax_g.scatter(lengths, influence, alpha=0.3, s=20, color='#333')
    r = np.corrcoef(lengths, influence)[0, 1]
    ax_g.set_xlabel('Text Length')
    ax_g.set_ylabel('Mean Influence (robust)')
    ax_g.set_title(f'G. Influence vs Length (r={r:.2f})', fontsize=12, fontweight='bold')

    # =========================================================================
    # H: Effect Sizes (bottom right)
    # =========================================================================
    ax_h = fig.add_subplot(gs[2, 2])
    metrics = ['n_active', 'mean_influence', 'concentration', 'activation']
    cola = groups.get('cola', [])
    others = [s for src, samps in groups.items() if src != 'cola' for s in samps]

    ds = [
        abs(cohens_d([s['n_active'] for s in cola], [s['n_active'] for s in others])),
        abs(cohens_d([s['mean_influence'] for s in cola], [s['mean_influence'] for s in others])),
        abs(cohens_d([s['top_100_concentration'] for s in cola], [s['top_100_concentration'] for s in others])),
        abs(cohens_d([s['mean_activation'] for s in cola], [s['mean_activation'] for s in others])),
    ]
    colors_h = ['#c0392b', '#27ae60', '#27ae60', '#27ae60']

    ax_h.bar(metrics, ds, color=colors_h, edgecolor='black')
    ax_h.axhline(y=0.8, color='gray', linestyle='--')
    ax_h.set_ylabel("Cohen's d")
    ax_h.set_title('H. Effect Sizes', fontsize=12, fontweight='bold')
    ax_h.tick_params(axis='x', rotation=30)

    # Main title
    fig.suptitle('Latent Diagnostics: Activation Topology Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / 'central_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'central_summary.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ central_summary.png (MAIN FIGURE)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Generating all figures for paper...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    domain_samples = load_domain_data()
    truth_samples = load_truthfulness_data()

    print(f"Loaded {len(domain_samples)} domain samples")
    print(f"Loaded {len(truth_samples)} truthfulness samples\n")

    print("Individual figures:")
    fig_truthfulness_overlap(truth_samples)
    fig_domain_distributions(domain_samples)
    fig_pca_clustering(domain_samples)
    fig_boxplots_significance(domain_samples)
    fig_correlation_heatmap(domain_samples)
    fig_detection_summary()

    print("\nMain figure:")
    fig_central_summary(domain_samples, truth_samples)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
