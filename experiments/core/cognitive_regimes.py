"""
Cognitive Regimes Analysis
==========================

Explores how different cognitive task families activate distinct
computational regimes in SAE feature space.

Task families (orthogonal to existing domain/linguistic analyses):
- Math reasoning (GSM8K): symbolic multi-step computation
- Code synthesis (HumanEval): program generation and algorithmic planning
- Factual retrieval (TruthfulQA): memory access with minimal reasoning
- Long-context abstraction (CNN/DailyMail): compression and salience selection

Hypothesis: Each task family occupies a distinct activation topology,
independent of text length. This extends the domain signature findings
into computational regimes.

Usage:
  # Prepare data (local)
  python experiments/cognitive_regimes_analysis.py --prepare

  # Run attribution on Modal
  modal run scripts/modal_general_attribution.py \
      --input-file data/cognitive_regimes/samples.json \
      --output-file data/results/cognitive_regimes_metrics.json

  # Analyze results (local)
  python experiments/cognitive_regimes_analysis.py --analyze
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
    from neural_polygraph.datasets import DatasetLoader, Sample
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False
    print("Warning: Dataset loader not available. Install: pip install datasets")


DATA_DIR = Path("data/cognitive_regimes")
RESULTS_DIR = Path("experiments/cognitive_regimes_analysis/runs")


# Task family definitions
TASK_FAMILIES = {
    "math": {
        "dataset": "gsm8k",
        "description": "Symbolic multi-step reasoning with arithmetic",
        "expected_topology": "structured_diffuse",  # wide but organized paths
    },
    "code": {
        "dataset": "humaneval",
        "description": "Program synthesis and algorithmic planning",
        "expected_topology": "focused_hierarchical",  # sharp API/syntax pathways
    },
    "retrieval": {
        "dataset": "truthfulqa",
        "description": "Factual recall with minimal synthesis",
        "expected_topology": "focused_sparse",  # memory access, less reasoning
    },
    "abstraction": {
        "dataset": "cnn_dailymail",
        "description": "Long-context compression and salience selection",
        "expected_topology": "diffuse_convergent",  # wide input, concentrated output
    },
}


def prepare_cognitive_regimes_dataset(n_per_family: int = 100, output_path: Path = None):
    """
    Prepare cognitive regimes dataset for attribution analysis.

    Loads samples from 4 task families that probe orthogonal cognitive capabilities.
    """
    if not LOADER_AVAILABLE:
        raise RuntimeError("Dataset loader not available")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or DATA_DIR / "samples.json"

    print("Loading cognitive regimes dataset...")
    loader = DatasetLoader()

    all_samples = []
    family_counts = {}

    for family_name, config in TASK_FAMILIES.items():
        dataset_name = config["dataset"]
        print(f"\n[{family_name}] Loading {dataset_name}...")

        try:
            # Use test split for gsm8k/humaneval, validation for others
            split = "test" if dataset_name in ["gsm8k", "humaneval"] else "validation"
            samples = loader.load(dataset_name, n_samples=n_per_family, split=split)
            print(f"  Loaded {len(samples)} samples")

            # Tag with family
            for s in samples:
                s.metadata = s.metadata or {}
                s.metadata["task_family"] = family_name
                s.metadata["expected_topology"] = config["expected_topology"]

            all_samples.extend(samples)
            family_counts[family_name] = len(samples)

        except Exception as e:
            print(f"  FAILED: {e}")
            family_counts[family_name] = 0

    # Convert to serializable format
    serialized = []
    for i, sample in enumerate(all_samples):
        serialized.append({
            "idx": i,
            "text": sample.text[:1000],  # Truncate for attribution
            "source": sample.source,
            "domain": sample.domain,
            "label": sample.label,
            "task_family": sample.metadata.get("task_family"),
            "expected_topology": sample.metadata.get("expected_topology"),
        })

    # Save
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "n_samples": len(serialized),
                "task_families": list(TASK_FAMILIES.keys()),
                "family_counts": family_counts,
                "purpose": "cognitive_regimes_analysis",
            },
            "samples": serialized,
        }, f, indent=2)

    print(f"\nSaved {len(serialized)} samples to {output_path}")
    print(f"\nFamily distribution:")
    for family, count in family_counts.items():
        print(f"  {family}: {count}")

    return output_path


def print_modal_command():
    """Print the command to run attribution on Modal."""

    command = """
# ============================================================
# Modal Attribution Command for Cognitive Regimes Analysis
# ============================================================

# First ensure data is prepared:
python experiments/cognitive_regimes_analysis.py --prepare

# Then run on Modal:
modal run scripts/modal_general_attribution.py \\
    --input-file data/cognitive_regimes/samples.json \\
    --output-file data/results/cognitive_regimes_metrics.json \\
    --n-workers 8

# Note: Uses parallel containers with incremental save.
# If interrupted, re-run the same command to resume.
# ============================================================
"""
    print(command)


def analyze_cognitive_regimes_results(results_path: Path = None):
    """
    Analyze cognitive regimes attribution results.

    Key questions:
    - Do different task families show distinct activation topologies?
    - Is the distinction independent of text length?
    - What metrics best separate the families?
    """
    results_path = results_path or Path("data/results/cognitive_regimes_metrics.json")

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run attribution first. See: python experiments/cognitive_regimes_analysis.py --modal-command")
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
    available_metrics = [m for m in metrics if m in df.columns]

    results = {"family_profiles": {}, "cross_family_comparison": {}, "length_independence": {}}

    # Add text length
    df["n_chars"] = df["text"].str.len()
    df["n_tokens_approx"] = df["n_chars"] / 4  # rough estimate

    # 1. Family profiles
    print("\n--- Task Family Profiles ---")

    families = df["task_family"].unique()

    for family in families:
        subset = df[df["task_family"] == family]
        profile = {
            "n_samples": len(subset),
            "expected_topology": subset["expected_topology"].iloc[0] if len(subset) > 0 else "unknown",
        }

        for metric in available_metrics:
            profile[metric] = {
                "mean": float(subset[metric].mean()),
                "std": float(subset[metric].std()),
                "median": float(subset[metric].median()),
            }

        # Length stats
        profile["text_length"] = {
            "mean": float(subset["n_chars"].mean()),
            "std": float(subset["n_chars"].std()),
        }

        results["family_profiles"][family] = profile
        print(f"  {family}: n={len(subset)}, expected={profile['expected_topology']}")

    # 2. Visualization: Family comparison
    if len(available_metrics) >= 2 and len(families) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        colors = plt.cm.Set2(np.linspace(0, 1, len(families)))

        for i, metric in enumerate(available_metrics[:6]):
            ax = axes[i]

            family_data = [df[df["task_family"] == f][metric].values for f in families]
            bp = ax.boxplot(family_data, labels=families, patch_artist=True)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "family_metric_distributions.png", dpi=150)
        plt.close()
        print(f"  Saved: family_metric_distributions.png")

    # 3. Length-controlled analysis
    print("\n--- Length Independence Check ---")

    # Normalize metrics by length
    for metric in ["n_active", "n_edges"]:
        if metric in df.columns:
            df[f"{metric}_per_char"] = df[metric] / df["n_chars"]
            df[f"{metric}_per_100_tokens"] = df[metric] / (df["n_tokens_approx"] / 100)

    # Correlation with length
    length_corrs = {}
    for metric in available_metrics:
        corr = df[metric].corr(df["n_chars"])
        length_corrs[metric] = float(corr)
        print(f"  {metric} ~ length: r={corr:.3f}")

    results["length_independence"]["raw_correlations"] = length_corrs

    # 4. Cross-family effect sizes (length-controlled)
    print("\n--- Cross-Family Comparisons (Cohen's d, length-controlled) ---")

    # Use normalized metric if available
    if "n_active_per_char" in df.columns:
        metric = "n_active_per_char"
    elif "mean_activation" in available_metrics:
        metric = "mean_activation"  # least confounded
    else:
        metric = available_metrics[0]

    print(f"  Using metric: {metric}")

    effect_matrix = np.zeros((len(families), len(families)))

    for i, f1 in enumerate(families):
        for j, f2 in enumerate(families):
            if i >= j:
                continue

            g1 = df[df["task_family"] == f1][metric].dropna().values
            g2 = df[df["task_family"] == f2][metric].dropna().values

            if len(g1) < 5 or len(g2) < 5:
                continue

            # Cohen's d
            pooled_std = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
            if pooled_std > 0:
                d_effect = (np.mean(g1) - np.mean(g2)) / pooled_std
            else:
                d_effect = 0

            effect_matrix[i, j] = d_effect
            effect_matrix[j, i] = -d_effect

            if abs(d_effect) > 0.5:
                print(f"  {f1} vs {f2}: d = {d_effect:.2f} {'***' if abs(d_effect) > 1.0 else '**' if abs(d_effect) > 0.8 else '*'}")

    results["cross_family_comparison"]["effect_matrix"] = effect_matrix.tolist()
    results["cross_family_comparison"]["families"] = list(families)
    results["cross_family_comparison"]["metric_used"] = metric

    # 5. Visualize effect matrix
    if len(families) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(effect_matrix, cmap="RdBu_r", vmin=-2, vmax=2)

        ax.set_xticks(range(len(families)))
        ax.set_yticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right")
        ax.set_yticklabels(families)

        for i in range(len(families)):
            for j in range(len(families)):
                ax.text(j, i, f"{effect_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)

        plt.colorbar(im, ax=ax, label="Cohen's d")
        plt.title(f"Cognitive Regime Separation\n({metric})")
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "family_effect_matrix.png", dpi=150)
        plt.close()
        print(f"  Saved: family_effect_matrix.png")

    # 6. PCA visualization
    from sklearn.decomposition import PCA

    feature_cols = [m for m in available_metrics if m in df.columns]
    if len(feature_cols) >= 2 and len(df) >= 20:
        X = df[feature_cols].fillna(0).values

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        df["pca_1"] = X_pca[:, 0]
        df["pca_2"] = X_pca[:, 1]

        fig, ax = plt.subplots(figsize=(10, 8))

        for family, color in zip(families, colors):
            subset = df[df["task_family"] == family]
            ax.scatter(subset["pca_1"], subset["pca_2"], label=family,
                      color=color, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("Cognitive Regime Separation in Attribution Space")
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "cognitive_regimes_pca.png", dpi=150)
        plt.close()
        print(f"  Saved: cognitive_regimes_pca.png")

        results["pca"] = {
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "feature_names": feature_cols,
        }

    # 7. Save results
    with open(run_dir / "cognitive_regimes_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 8. Generate report
    report = generate_cognitive_regimes_report(df, results, available_metrics, families)
    with open(run_dir / "cognitive_regimes_report.md", "w") as f:
        f.write(report)

    print(f"\nAnalysis complete: {run_dir}")


def generate_cognitive_regimes_report(df, results, metrics, families):
    """Generate markdown report for cognitive regimes analysis."""

    lines = []
    lines.append("# Cognitive Regimes Attribution Analysis")
    lines.append("")
    lines.append(f"**Samples:** {len(df)}")
    lines.append(f"**Task Families:** {', '.join(families)}")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append("")
    lines.append("Different cognitive task families (math reasoning, code synthesis, factual retrieval,")
    lines.append("long-context abstraction) occupy distinct activation topologies in SAE feature space,")
    lines.append("independent of text length.")
    lines.append("")

    lines.append("## Task Family Profiles")
    lines.append("")

    for family, profile in results.get("family_profiles", {}).items():
        lines.append(f"### {family.upper()} (n={profile['n_samples']})")
        lines.append(f"*Expected topology: {profile.get('expected_topology', 'unknown')}*")
        lines.append("")

        if profile.get("text_length"):
            lines.append(f"- **Text length:** {profile['text_length']['mean']:.0f} ± {profile['text_length']['std']:.0f} chars")

        lines.append("")
        lines.append("| Metric | Mean | Std | Median |")
        lines.append("|--------|------|-----|--------|")

        for metric in metrics:
            if metric in profile:
                m = profile[metric]
                lines.append(f"| {metric} | {m['mean']:.4f} | {m['std']:.4f} | {m['median']:.4f} |")

        lines.append("")

    lines.append("## Length Independence")
    lines.append("")
    lines.append("| Metric | Correlation with Length |")
    lines.append("|--------|------------------------|")

    for metric, corr in results.get("length_independence", {}).get("raw_correlations", {}).items():
        flag = "" if abs(corr) < 0.3 else " ⚠️" if abs(corr) < 0.6 else " ❌"
        lines.append(f"| {metric} | r = {corr:.3f}{flag} |")

    lines.append("")
    lines.append("*Note: High correlations (|r| > 0.5) indicate length confounding.*")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("*To be filled after analysis:*")
    lines.append("")
    lines.append("- [ ] Which families show the largest separation?")
    lines.append("- [ ] Are the differences robust to length control?")
    lines.append("- [ ] Do expected topologies match observed patterns?")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cognitive Regimes Analysis")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset for attribution")
    parser.add_argument("--modal-command", action="store_true", help="Print Modal command")
    parser.add_argument("--analyze", action="store_true", help="Analyze attribution results")
    parser.add_argument("--n-per-family", type=int, default=100, help="Samples per task family")

    args = parser.parse_args()

    if args.prepare:
        prepare_cognitive_regimes_dataset(n_per_family=args.n_per_family)
    elif args.modal_command:
        print_modal_command()
    elif args.analyze:
        analyze_cognitive_regimes_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
