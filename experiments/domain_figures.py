"""
Domain Analysis Figures for Latent Diagnostics Paper

Generates publication-ready figures showing activation topology differences across domains.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

COLORS = {
    'cola': '#2ecc71',      # green - grammar (focused)
    'winogrande': '#3498db', # blue
    'snli': '#9b59b6',       # purple
    'hellaswag': '#e74c3c',  # red
    'paws': '#e67e22',       # orange - paraphrase (diffuse)
}

LABELS = {
    'cola': 'CoLA (Grammar)',
    'winogrande': 'WinoGrande (Commonsense)',
    'snli': 'SNLI (Inference)',
    'hellaswag': 'HellaSwag (Situational)',
    'paws': 'PAWS (Paraphrase)',
}


def load_data():
    with open('data/results/domain_attribution_metrics.json') as f:
        data = json.load(f)
    return data['samples']


def group_by_source(samples):
    groups = {}
    for s in samples:
        src = s.get('source', 'unknown')
        if src not in groups:
            groups[src] = []
        groups[src].append(s)
    return groups


def fig1_radar_chart(samples, output_dir):
    """
    Radar chart showing activation profile for each domain.
    Uses ROBUST metrics (influence, concentration, activation) not confounded ones.
    """
    groups = group_by_source(samples)

    # Metrics to show (robust ones only)
    metrics = ['mean_influence', 'top_100_concentration', 'mean_activation']
    metric_labels = ['Influence', 'Concentration', 'Activation']

    # Compute means per source
    source_means = {}
    for src, samps in groups.items():
        source_means[src] = {
            m: np.mean([s[m] for s in samps]) for m in metrics
        }

    # Normalize each metric to 0-1 range across sources
    normalized = {}
    for src in source_means:
        normalized[src] = {}

    for m in metrics:
        vals = [source_means[src][m] for src in source_means]
        min_v, max_v = min(vals), max(vals)
        for src in source_means:
            normalized[src][m] = (source_means[src][m] - min_v) / (max_v - min_v + 1e-10)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for src in ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']:
        if src not in normalized:
            continue
        values = [normalized[src][m] for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=LABELS.get(src, src),
                color=COLORS.get(src, '#333'))
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(src, '#333'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Activation Topology by Domain\n(Normalized Robust Metrics)', size=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_domain_radar.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_domain_radar.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_domain_radar.png")


def fig2_influence_concentration_scatter(samples, output_dir):
    """
    Scatter plot: mean_influence vs concentration, colored by domain.
    Shows clustering of domains in the diagnostic space.
    """
    groups = group_by_source(samples)

    fig, ax = plt.subplots(figsize=(10, 8))

    for src in ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']:
        if src not in groups:
            continue
        samps = groups[src]
        x = [s['mean_influence'] for s in samps]
        y = [s['top_100_concentration'] for s in samps]
        ax.scatter(x, y, alpha=0.6, label=LABELS.get(src, src),
                   color=COLORS.get(src, '#333'), s=60, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Mean Influence (edge strength)', size=14)
    ax.set_ylabel('Top-100 Concentration', size=14)
    ax.set_title('Domain Clustering in Influence-Concentration Space', size=16)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Grammar\n(focused)', xy=(0.0095, 0.006), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    ax.annotate('Reasoning\n(diffuse)', xy=(0.004, 0.0015), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#e67e22', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_influence_concentration.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_influence_concentration.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_influence_concentration.png")


def fig3_gradient_bar_chart(samples, output_dir):
    """
    Bar chart showing the gradient from focused (grammar) to diffuse (reasoning).
    Orders domains by mean_influence.
    """
    groups = group_by_source(samples)

    # Compute means and order by influence
    data = []
    for src, samps in groups.items():
        data.append({
            'source': src,
            'influence': np.mean([s['mean_influence'] for s in samps]),
            'concentration': np.mean([s['top_100_concentration'] for s in samps]),
            'influence_std': np.std([s['mean_influence'] for s in samps]),
        })

    data.sort(key=lambda x: x['influence'], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(data))
    bars = ax.bar(x, [d['influence'] for d in data],
                  yerr=[d['influence_std'] for d in data],
                  color=[COLORS.get(d['source'], '#333') for d in data],
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(d['source'], d['source']) for d in data], rotation=15, ha='right')
    ax.set_ylabel('Mean Influence', size=14)
    ax.set_title('Activation Influence Gradient: Focused → Diffuse', size=16)

    # Add arrow annotation
    ax.annotate('', xy=(len(data)-0.5, 0.002), xytext=(0.5, 0.002),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(len(data)/2, 0.0015, 'Increasing task complexity', ha='center', fontsize=11, color='gray')

    ax.set_ylim(0, max([d['influence'] for d in data]) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_influence_gradient.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_influence_gradient.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_influence_gradient.png")


def fig4_length_controlled_comparison(samples, output_dir):
    """
    Show that influence/concentration are NOT just length.
    Left: n_active vs length (confounded)
    Right: mean_influence vs length (independent)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    groups = group_by_source(samples)

    # Left: n_active vs length (CONFOUNDED)
    ax = axes[0]
    for src in ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']:
        if src not in groups:
            continue
        samps = groups[src]
        x = [len(s.get('text', '')) for s in samps]
        y = [s['n_active'] for s in samps]
        ax.scatter(x, y, alpha=0.5, label=LABELS.get(src, src),
                   color=COLORS.get(src, '#333'), s=40)

    # Fit line
    all_lens = [len(s.get('text', '')) for s in samples]
    all_n = [s['n_active'] for s in samples]
    z = np.polyfit(all_lens, all_n, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_lens), max(all_lens), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, label=f'r = 0.98')

    ax.set_xlabel('Text Length (chars)', size=12)
    ax.set_ylabel('N Active Features', size=12)
    ax.set_title('Feature Count: CONFOUNDED by Length', size=14, color='#c0392b')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: mean_influence vs length (INDEPENDENT)
    ax = axes[1]
    for src in ['cola', 'winogrande', 'snli', 'hellaswag', 'paws']:
        if src not in groups:
            continue
        samps = groups[src]
        x = [len(s.get('text', '')) for s in samps]
        y = [s['mean_influence'] for s in samps]
        ax.scatter(x, y, alpha=0.5, label=LABELS.get(src, src),
                   color=COLORS.get(src, '#333'), s=40)

    ax.set_xlabel('Text Length (chars)', size=12)
    ax.set_ylabel('Mean Influence', size=12)
    ax.set_title('Mean Influence: INDEPENDENT of Length', size=14, color='#27ae60')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_length_control.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_length_control.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_length_control.png")


def fig5_effect_sizes(samples, output_dir):
    """
    Bar chart of Cohen's d for grammar vs others across metrics.
    Shows which metrics have real signal.
    """
    groups = group_by_source(samples)
    cola = groups.get('cola', [])
    others = [s for src, samps in groups.items() if src != 'cola' for s in samps]

    metrics = ['n_active', 'mean_influence', 'top_100_concentration', 'mean_activation']
    metric_labels = ['N Active\n(confounded)', 'Mean Influence\n(robust)',
                     'Concentration\n(robust)', 'Mean Activation\n(robust)']

    def cohens_d(a, b):
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
        if pooled_std == 0:
            return 0
        return abs(np.mean(a) - np.mean(b)) / pooled_std

    effect_sizes = []
    for m in metrics:
        d = cohens_d([s[m] for s in cola], [s[m] for s in others])
        effect_sizes.append(d)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c', '#27ae60', '#27ae60', '#27ae60']  # red for confounded, green for robust
    bars = ax.bar(metric_labels, effect_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='Large effect (d=0.8)')
    ax.set_ylabel("Cohen's d (Grammar vs Others)", size=14)
    ax.set_title('Effect Sizes: Robust Metrics Show Strong Signal', size=16)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, effect_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'd={val:.2f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_effect_sizes.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_effect_sizes.png")


def main():
    print("Generating domain analysis figures...")

    samples = load_data()
    print(f"Loaded {len(samples)} samples")

    output_dir = Path('figures/domain_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1_radar_chart(samples, output_dir)
    fig2_influence_concentration_scatter(samples, output_dir)
    fig3_gradient_bar_chart(samples, output_dir)
    fig4_length_controlled_comparison(samples, output_dir)
    fig5_effect_sizes(samples, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
