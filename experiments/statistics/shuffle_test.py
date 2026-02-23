"""
Random-Label Shuffle Test for Falsification

Tests whether domain signal is genuine or accidental geometry by:
1. Computing observed effect size (grammar vs non-grammar)
2. Shuffling domain labels 1000x
3. Computing effect size for each shuffle
4. Building null distribution
5. Computing p-value: proportion of shuffled d >= observed d

If p < 0.05, signal is statistically significant (not accidental).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'


def load_domain_data():
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

    metrics = ['n_active', 'mean_influence', 'top_100_concentration', 'mean_activation']

    for m in metrics:
        vals = np.array([s[m] for s in samples])
        resid = residualize(vals, lengths)
        for i, s in enumerate(samples):
            s[f'{m}_resid'] = resid[i]

    return samples


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled_std == 0:
        return 0
    return (np.mean(a) - np.mean(b)) / pooled_std


def permutation_test(samples, metric='mean_influence_resid', n_permutations=1000):
    """
    Permutation test for grammar vs non-grammar.

    Returns:
        observed_d: Observed Cohen's d
        null_distribution: Array of shuffled Cohen's d values
        p_value: Proportion of shuffled d >= observed d
    """
    # Extract metric values and domain labels
    values = np.array([s[metric] for s in samples])
    domains = np.array([s['domain'] for s in samples])

    # Compute observed effect size
    grammar_mask = domains == 'grammar'
    grammar_vals = values[grammar_mask]
    non_grammar_vals = values[~grammar_mask]

    observed_d = abs(cohens_d(grammar_vals, non_grammar_vals))

    print(f"\nObserved:")
    print(f"  Grammar: n={len(grammar_vals)}, mean={np.mean(grammar_vals):.4f}, std={np.std(grammar_vals):.4f}")
    print(f"  Non-grammar: n={len(non_grammar_vals)}, mean={np.mean(non_grammar_vals):.4f}, std={np.std(non_grammar_vals):.4f}")
    print(f"  Cohen's d = {observed_d:.4f}")

    # Shuffle labels and compute null distribution
    null_distribution = []
    rng = np.random.RandomState(42)

    for i in range(n_permutations):
        # Shuffle domain labels
        shuffled_domains = rng.permutation(domains)

        # Compute Cohen's d for shuffled labels
        shuffled_grammar_mask = shuffled_domains == 'grammar'
        shuffled_grammar_vals = values[shuffled_grammar_mask]
        shuffled_non_grammar_vals = values[~shuffled_grammar_mask]

        d = abs(cohens_d(shuffled_grammar_vals, shuffled_non_grammar_vals))
        null_distribution.append(d)

    null_distribution = np.array(null_distribution)

    # Compute p-value (proportion of shuffled d >= observed d)
    p_value = np.mean(null_distribution >= observed_d)

    return observed_d, null_distribution, p_value


def plot_null_distribution(observed_d, null_distribution, p_value, metric_name, output_path):
    """Generate figure showing null distribution with observed d marked."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of null distribution
    ax.hist(null_distribution, bins=50, alpha=0.7, color='#3498db', edgecolor='white', label='Null distribution')

    # Mark observed d
    ax.axvline(observed_d, color='#e74c3c', linewidth=3, linestyle='--', label=f'Observed d = {observed_d:.3f}')

    # Statistics
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)

    # Add text box with statistics
    stats_text = (
        f'Null distribution:\n'
        f'  Mean: {null_mean:.3f}\n'
        f'  Std: {null_std:.3f}\n'
        f'\n'
        f'Observed: {observed_d:.3f}\n'
        f'p-value: {p_value:.4f}\n'
        f'\n'
        f'{"✓ SIGNIFICANT" if p_value < 0.05 else "✗ NOT SIGNIFICANT"}'
    )

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='gray', linewidth=1.5))

    ax.set_xlabel("Cohen's d", fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Shuffle Test: {metric_name}\nGrammar vs Non-Grammar', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {output_path}")


def main():
    print("=" * 70)
    print("SHUFFLE TEST: Random-Label Permutation for Falsification")
    print("=" * 70)

    # Load data
    samples = load_domain_data()
    print(f"\nLoaded {len(samples)} samples")

    # Add length-controlled residuals
    samples = add_residuals(samples)
    print("Added length-controlled residuals")

    # Count domains
    domain_counts = {}
    for s in samples:
        domain = s['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    # Test key metrics
    metrics = [
        ('mean_influence_resid', 'Influence (length-controlled)'),
        ('top_100_concentration_resid', 'Concentration (length-controlled)'),
        ('n_active_resid', 'N Active (length-controlled)'),
    ]

    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    for metric_key, metric_name in metrics:
        print("\n" + "=" * 70)
        print(f"Testing: {metric_name}")
        print("=" * 70)

        observed_d, null_dist, p_value = permutation_test(samples, metric=metric_key, n_permutations=1000)

        print(f"\nNull distribution:")
        print(f"  Mean: {np.mean(null_dist):.4f}")
        print(f"  Std: {np.std(null_dist):.4f}")
        print(f"  Min: {np.min(null_dist):.4f}")
        print(f"  Max: {np.max(null_dist):.4f}")

        print(f"\nResult:")
        print(f"  Observed Cohen's d: {observed_d:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT (p < 0.05) - Signal is REAL")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p >= 0.05) - Signal is accidental")

        # Generate figure
        output_name = f"shuffle_null_distribution_{metric_key.replace('_resid', '')}.png"
        output_path = output_dir / output_name
        plot_null_distribution(observed_d, null_dist, p_value, metric_name, output_path)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nIf influence and concentration show p < 0.05, the domain signal is:")
    print("  1. Statistically significant")
    print("  2. NOT due to accidental geometry")
    print("  3. Genuine computational regime difference")
    print("\nIf n_active shows p >= 0.05, it confirms n_active signal collapsed")
    print("after length control (as expected from previous analysis).")
    print("=" * 70)


if __name__ == '__main__':
    main()
