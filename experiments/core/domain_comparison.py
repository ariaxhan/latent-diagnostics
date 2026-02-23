"""
Domain Signature Analysis
=========================

Explores how different text domains (code, scientific, legal, poetry)
exhibit different activation patterns in SAE feature space.

This is mechanistic interpretability research - NOT classification.
Goal: Discover what SAE features encode domain-specific patterns.

Usage:
  # Prepare data (local)
  python experiments/domain_analysis.py --prepare

  # Run attribution on Modal (provides command)
  python experiments/domain_analysis.py --modal-command

  # Analyze results (local)
  python experiments/domain_analysis.py --analyze
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from neural_polygraph.datasets import DatasetLoader, Sample, create_domain_comparison_dataset
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False
    print("Warning: Dataset loader not available. Install: pip install datasets")


DATA_DIR = Path("data/domain_analysis")
RESULTS_DIR = Path("experiments/domain_analysis/runs")


def prepare_domain_dataset(n_per_domain: int = 100, output_path: Path = None):
    """
    Prepare domain comparison dataset for attribution analysis.

    Loads samples from diverse domains and saves to JSON for Modal processing.
    """
    if not LOADER_AVAILABLE:
        raise RuntimeError("Dataset loader not available")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or DATA_DIR / "domain_samples.json"

    print("Loading domain comparison dataset...")
    loader = DatasetLoader()

    # Load from available datasets
    all_samples = []

    # Define domain sources
    sources = {
        # Diverse text types
        "cola": ("grammar", n_per_domain),
        "winogrande": ("commonsense", n_per_domain),
        "hellaswag": ("situational", n_per_domain),
        "snli": ("inference", n_per_domain),
        "fever": ("factual", n_per_domain),
        "paws": ("paraphrase", n_per_domain),
        # Domain-specific
        "codesearchnet": ("code", n_per_domain),
        "pubmedqa": ("scientific", n_per_domain),
        "poem_sentiment": ("poetry", min(n_per_domain, 200)),  # Smaller dataset
        "truthfulqa": ("qa_truthfulness", n_per_domain),
    }

    for source_name, (domain, n) in sources.items():
        try:
            samples = loader.load(source_name, n_samples=n, split="validation")
            print(f"  {source_name}: loaded {len(samples)} samples -> domain '{domain}'")
            all_samples.extend(samples)
        except Exception as e:
            print(f"  {source_name}: FAILED - {e}")

    # Convert to serializable format
    serialized = []
    for i, sample in enumerate(all_samples):
        serialized.append({
            "idx": i,
            "text": sample.text[:1000],  # Truncate for attribution
            "source": sample.source,
            "domain": sample.domain,
            "label": sample.label,
        })

    # Save
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "n_samples": len(serialized),
                "sources": list(sources.keys()),
            },
            "samples": serialized,
        }, f, indent=2)

    print(f"\nSaved {len(serialized)} samples to {output_path}")

    # Domain distribution
    domain_counts = defaultdict(int)
    for s in serialized:
        domain_counts[s["domain"]] += 1

    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    return output_path


def print_modal_command():
    """Print the command to run attribution on Modal."""

    command = """
# ============================================================
# Modal Attribution Command for Domain Analysis
# ============================================================

# First ensure data is prepared:
python experiments/domain_analysis.py --prepare

# Then run on Modal (modify n-samples as needed):
modal run scripts/modal_domain_attribution.py --input-file data/domain_analysis/domain_samples.json --output-file data/results/domain_attribution_metrics.json

# Note: You need to create scripts/modal_domain_attribution.py
# See scripts/modal_pint_benchmark.py as template
# ============================================================
"""
    print(command)


def analyze_domain_results(results_path: Path = None):
    """
    Analyze domain attribution results.

    Explores:
    - Domain-specific activation patterns
    - Feature sparsity by domain
    - Cross-domain metric correlations
    """
    results_path = results_path or Path("data/results/domain_attribution_metrics.json")

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run attribution first. See: python experiments/domain_analysis.py --modal-command")
        return

    with open(results_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["samples"])

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    print(f"Analyzing {len(df)} samples...")
    print(f"Output: {run_dir}")

    # Metrics to analyze
    metrics = ["n_active", "n_edges", "mean_influence", "top_100_concentration",
               "mean_activation", "logit_entropy"]

    # Available metrics (some may be missing)
    available_metrics = [m for m in metrics if m in df.columns]

    results = {"domain_profiles": {}, "cross_domain_comparison": {}}

    # 1. Domain profiles
    print("\n--- Domain Profiles ---")

    domains = df["domain"].unique()

    for domain in domains:
        subset = df[df["domain"] == domain]
        profile = {"n_samples": len(subset)}

        for metric in available_metrics:
            profile[metric] = {
                "mean": float(subset[metric].mean()),
                "std": float(subset[metric].std()),
                "median": float(subset[metric].median()),
            }

        results["domain_profiles"][domain] = profile
        print(f"  {domain}: n={len(subset)}")

    # 2. Visualization: Domain comparison
    if len(available_metrics) >= 2 and len(domains) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, metric in enumerate(available_metrics[:6]):
            ax = axes[i]

            domain_data = [df[df["domain"] == d][metric].values for d in domains]
            bp = ax.boxplot(domain_data, labels=domains, patch_artist=True)

            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "domain_metric_distributions.png", dpi=150)
        plt.close()
        print(f"  Saved: domain_metric_distributions.png")

    # 3. Normalized metrics (per-character)
    df["n_chars"] = df["text"].str.len()

    for metric in ["n_active", "n_edges"]:
        if metric in df.columns:
            df[f"{metric}_per_char"] = df[metric] / df["n_chars"]

    # 4. Cross-domain effect sizes (which domains differ most?)
    from scipy import stats

    print("\n--- Cross-Domain Comparisons (Cohen's d) ---")

    if "n_active_per_char" in df.columns:
        metric = "n_active_per_char"
    elif "n_active" in available_metrics:
        metric = "n_active"
    else:
        metric = available_metrics[0]

    effect_matrix = np.zeros((len(domains), len(domains)))

    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i >= j:
                continue

            g1 = df[df["domain"] == d1][metric].values
            g2 = df[df["domain"] == d2][metric].values

            # Cohen's d
            pooled_std = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
            if pooled_std > 0:
                d_effect = (np.mean(g1) - np.mean(g2)) / pooled_std
            else:
                d_effect = 0

            effect_matrix[i, j] = d_effect
            effect_matrix[j, i] = -d_effect

            if abs(d_effect) > 0.5:
                print(f"  {d1} vs {d2}: d = {d_effect:.2f}")

    results["cross_domain_comparison"]["effect_matrix"] = effect_matrix.tolist()
    results["cross_domain_comparison"]["domains"] = list(domains)
    results["cross_domain_comparison"]["metric_used"] = metric

    # 5. Visualize effect matrix
    if len(domains) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(effect_matrix, cmap="RdBu_r", vmin=-2, vmax=2)

        ax.set_xticks(range(len(domains)))
        ax.set_yticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=45, ha="right")
        ax.set_yticklabels(domains)

        for i in range(len(domains)):
            for j in range(len(domains)):
                ax.text(j, i, f"{effect_matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, label="Cohen's d")
        plt.title(f"Domain Effect Sizes ({metric})")
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "domain_effect_matrix.png", dpi=150)
        plt.close()
        print(f"  Saved: domain_effect_matrix.png")

    # 6. PCA visualization (if enough data)
    from sklearn.decomposition import PCA

    feature_cols = [m for m in available_metrics if m in df.columns]
    if len(feature_cols) >= 2 and len(df) >= 50:
        X = df[feature_cols].fillna(0).values

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        df["pca_1"] = X_pca[:, 0]
        df["pca_2"] = X_pca[:, 1]

        fig, ax = plt.subplots(figsize=(10, 8))

        for domain in domains:
            subset = df[df["domain"] == domain]
            ax.scatter(subset["pca_1"], subset["pca_2"], label=domain, alpha=0.6, s=30)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("Domain Separation in Attribution Feature Space")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "domain_pca.png", dpi=150)
        plt.close()
        print(f"  Saved: domain_pca.png")

        results["pca"] = {
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "feature_names": feature_cols,
        }

    # 7. Save results
    with open(run_dir / "domain_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 8. Generate report
    report = generate_domain_report(df, results, available_metrics, domains)
    with open(run_dir / "domain_analysis_report.md", "w") as f:
        f.write(report)

    print(f"\nAnalysis complete: {run_dir}")


def generate_domain_report(df, results, metrics, domains):
    """Generate markdown report for domain analysis."""

    lines = []
    lines.append("# Domain Signature Analysis Report")
    lines.append("")
    lines.append(f"**Samples:** {len(df)}")
    lines.append(f"**Domains:** {len(domains)}")
    lines.append("")

    lines.append("## Domain Profiles")
    lines.append("")

    for domain, profile in results["domain_profiles"].items():
        lines.append(f"### {domain} (n={profile['n_samples']})")
        lines.append("")
        lines.append("| Metric | Mean | Std | Median |")
        lines.append("|--------|------|-----|--------|")

        for metric in metrics:
            if metric in profile:
                m = profile[metric]
                lines.append(f"| {metric} | {m['mean']:.4f} | {m['std']:.4f} | {m['median']:.4f} |")

        lines.append("")

    lines.append("## Key Observations")
    lines.append("")
    lines.append("*Patterns to investigate:*")
    lines.append("")
    lines.append("- Which domains have highest/lowest feature density?")
    lines.append("- Are there domain-specific feature circuits?")
    lines.append("- How does text length interact with domain effects?")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Domain Signature Analysis")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset for attribution")
    parser.add_argument("--modal-command", action="store_true", help="Print Modal command")
    parser.add_argument("--analyze", action="store_true", help="Analyze attribution results")
    parser.add_argument("--n-per-domain", type=int, default=100, help="Samples per domain")

    args = parser.parse_args()

    if args.prepare:
        prepare_domain_dataset(n_per_domain=args.n_per_domain)
    elif args.modal_command:
        print_modal_command()
    elif args.analyze:
        analyze_domain_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
