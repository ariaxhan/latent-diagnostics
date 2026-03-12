"""
Graph Metric Analysis from Existing Data
=========================================

Extracts insights from already-computed attribution metrics.
No new GPU runs required.

Metrics available:
- mean_influence, max_influence, top_100_concentration
- mean_activation, max_activation
- logit_entropy, max_logit_prob
- n_active, n_edges

Usage:
    python experiments/core/existing_data_analysis.py
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = Path("data/results/domain_attribution_metrics.json")
OUTPUT_DIR = Path("experiments/_runs/existing_data_analysis")

METRICS = [
    "mean_influence", "max_influence", "top_100_concentration",
    "mean_activation", "max_activation",
    "logit_entropy", "max_logit_prob",
    "n_active", "n_edges"
]

# Only analyze domains with enough samples
MIN_SAMPLES = 10


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def load_data() -> pd.DataFrame:
    """Load and filter attribution results."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    df = pd.DataFrame(data["samples"])

    # Filter to available metrics
    available = [m for m in METRICS if m in df.columns]

    # Filter to domains with enough samples
    domain_counts = df["domain"].value_counts()
    valid_domains = domain_counts[domain_counts >= MIN_SAMPLES].index
    df = df[df["domain"].isin(valid_domains)]

    return df, available


def compute_correlation_matrix(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Compute correlation matrix between metrics."""
    return df[metrics].corr()


def compute_effect_sizes(df: pd.DataFrame, metrics: List[str]) -> Dict:
    """Compute Cohen's d between all domain pairs."""
    domains = df["domain"].unique()
    results = {}

    for metric in metrics:
        results[metric] = {}
        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                x1 = df[df["domain"] == d1][metric].dropna()
                x2 = df[df["domain"] == d2][metric].dropna()

                if len(x1) < 5 or len(x2) < 5:
                    continue

                # Cohen's d
                pooled_std = np.sqrt(((len(x1)-1)*x1.std()**2 + (len(x2)-1)*x2.std()**2) / (len(x1)+len(x2)-2))
                d = (x1.mean() - x2.mean()) / (pooled_std + 1e-10)

                # t-test
                t_stat, p_val = stats.ttest_ind(x1, x2)

                results[metric][f"{d1}_vs_{d2}"] = {
                    "cohens_d": float(d),
                    "p_value": float(p_val),
                    "n1": len(x1),
                    "n2": len(x2),
                    "mean1": float(x1.mean()),
                    "mean2": float(x2.mean()),
                }

    return results


def compute_length_correlations(df: pd.DataFrame, metrics: List[str]) -> Dict:
    """Check correlation of each metric with text length."""
    if "text" not in df.columns:
        return {}

    df = df.copy()
    df["text_length"] = df["text"].str.len()

    results = {}
    for metric in metrics:
        r, p = stats.pearsonr(df["text_length"], df[metric].fillna(0))
        results[metric] = {"pearson_r": float(r), "p_value": float(p)}

    return results


def compute_pca_projection(df: pd.DataFrame, metrics: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """PCA projection of samples in metric space."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[metrics].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(3, len(metrics)))
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca.explained_variance_ratio_


def generate_visualizations(df: pd.DataFrame, metrics: List[str], output_dir: Path):
    """Generate analysis figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = compute_correlation_matrix(df, metrics)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(metrics, fontsize=8)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.title("Metric Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: correlation_matrix.png")

    # 2. Domain distributions for key metrics
    key_metrics = ["mean_influence", "top_100_concentration", "n_active"]
    key_metrics = [m for m in key_metrics if m in metrics]

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(5*len(key_metrics), 5))
    if len(key_metrics) == 1:
        axes = [axes]

    domains = df["domain"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))

    for ax, metric in zip(axes, key_metrics):
        for domain, color in zip(domains, colors):
            data = df[df["domain"] == domain][metric].dropna()
            ax.hist(data, bins=20, alpha=0.5, label=domain, color=color)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_title(f"Distribution: {metric}")

    plt.tight_layout()
    plt.savefig(output_dir / "domain_distributions.png", dpi=150)
    plt.close()
    print(f"Saved: domain_distributions.png")

    # 3. PCA projection
    X_pca, var_ratio = compute_pca_projection(df, metrics)

    fig, ax = plt.subplots(figsize=(10, 8))
    for domain, color in zip(domains, colors):
        mask = df["domain"] == domain
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=domain, alpha=0.6, c=[color])

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
    ax.set_title("Domain Clusters in Metric Space (PCA)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pca_projection.png", dpi=150)
    plt.close()
    print(f"Saved: pca_projection.png")

    # 4. Length correlation plot
    if "text" in df.columns:
        df_plot = df.copy()
        df_plot["text_length"] = df_plot["text"].str.len()

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for ax, metric in zip(axes, metrics[:6]):
            for domain, color in zip(domains, colors):
                mask = df_plot["domain"] == domain
                ax.scatter(df_plot.loc[mask, "text_length"],
                          df_plot.loc[mask, metric],
                          alpha=0.5, s=20, c=[color], label=domain)
            ax.set_xlabel("Text Length")
            ax.set_ylabel(metric)
            r, _ = stats.pearsonr(df_plot["text_length"], df_plot[metric].fillna(0))
            ax.set_title(f"{metric} (r={r:.2f})")

        axes[0].legend(fontsize=6, loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "length_correlations.png", dpi=150)
        plt.close()
        print(f"Saved: length_correlations.png")


def generate_report(df: pd.DataFrame, metrics: List[str], effect_sizes: Dict,
                   length_corrs: Dict, output_dir: Path) -> str:
    """Generate markdown analysis report."""

    lines = [
        "# Existing Data Analysis Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Dataset Summary",
        f"\n- Total samples: {len(df)}",
        f"- Metrics: {len(metrics)}",
        f"- Domains: {df['domain'].nunique()}",
        "",
    ]

    # Domain counts
    lines.append("### Samples per Domain\n")
    lines.append("| Domain | Count |")
    lines.append("|--------|-------|")
    for domain, count in df["domain"].value_counts().items():
        lines.append(f"| {domain} | {count} |")

    # Length correlations
    if length_corrs:
        lines.append("\n## Length Correlations (CONFOUND CHECK)")
        lines.append("\n| Metric | Pearson r | Interpretation |")
        lines.append("|--------|-----------|----------------|")
        for metric, stats_dict in sorted(length_corrs.items(), key=lambda x: -abs(x[1]["pearson_r"])):
            r = stats_dict["pearson_r"]
            interp = "CONFOUNDED" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "ok"
            lines.append(f"| {metric} | {r:.3f} | {interp} |")

    # Top effect sizes
    lines.append("\n## Top Effect Sizes (Cohen's d)")
    lines.append("\n| Comparison | Metric | Cohen's d | p-value |")
    lines.append("|------------|--------|-----------|---------|")

    all_effects = []
    for metric, comparisons in effect_sizes.items():
        for comp, stats_dict in comparisons.items():
            all_effects.append((metric, comp, stats_dict["cohens_d"], stats_dict["p_value"]))

    # Sort by absolute effect size
    all_effects.sort(key=lambda x: -abs(x[2]))

    for metric, comp, d, p in all_effects[:20]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"| {comp} | {metric} | {d:.2f} | {p:.4f}{sig} |")

    # Correlation highlights
    corr = compute_correlation_matrix(df, metrics)
    lines.append("\n## Metric Correlations (|r| > 0.5)")
    lines.append("\n| Metric 1 | Metric 2 | Pearson r |")
    lines.append("|----------|----------|-----------|")

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i < j and abs(corr.loc[m1, m2]) > 0.5:
                lines.append(f"| {m1} | {m2} | {corr.loc[m1, m2]:.3f} |")

    # Key insights
    lines.append("\n## Key Insights")

    # Find most discriminative metric
    if all_effects:
        top_metric = max(set(e[0] for e in all_effects[:10]),
                        key=lambda m: max(abs(e[2]) for e in all_effects if e[0] == m))
        lines.append(f"\n- **Most discriminative metric:** {top_metric}")

    # Find most confounded metric
    if length_corrs:
        worst = max(length_corrs.items(), key=lambda x: abs(x[1]["pearson_r"]))
        lines.append(f"- **Most length-confounded:** {worst[0]} (r={worst[1]['pearson_r']:.2f})")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Existing Data Analysis")
    print(f"=" * 50)
    print(f"Output: {run_dir}")

    # Load data
    df, available_metrics = load_data()
    print(f"Samples: {len(df)}")
    print(f"Metrics: {available_metrics}")
    print(f"Domains: {list(df['domain'].unique())}")

    # Compute statistics
    print("\nComputing statistics...")
    effect_sizes = compute_effect_sizes(df, available_metrics)
    length_corrs = compute_length_correlations(df, available_metrics)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(df, available_metrics, run_dir / "figures")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(df, available_metrics, effect_sizes, length_corrs, run_dir)

    with open(run_dir / "analysis_report.md", "w") as f:
        f.write(report)
    print(f"Saved: analysis_report.md")

    # Save raw results
    results = {
        "metadata": {
            "timestamp": timestamp,
            "n_samples": len(df),
            "metrics": available_metrics,
            "domains": list(df["domain"].unique()),
        },
        "effect_sizes": effect_sizes,
        "length_correlations": length_corrs,
        "correlation_matrix": compute_correlation_matrix(df, available_metrics).to_dict(),
    }

    with open(run_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: analysis_results.json")

    print(f"\nDone: {run_dir}")


if __name__ == "__main__":
    main()
