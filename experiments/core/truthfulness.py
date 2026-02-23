"""
Truthfulness Analysis
=====================

Explores how truthful vs. false/hallucinated content activates
different feature patterns in SAE space.

This is directly relevant to hallucination detection research.

Usage:
  # Prepare data (local)
  python experiments/truthfulness_analysis.py --prepare

  # Run attribution on Modal
  # modal run scripts/modal_general_attribution.py \\
  #     --input-file data/truthfulness/samples.json \\
  #     --output-file data/results/truthfulness_metrics.json

  # Analyze results (local)
  python experiments/truthfulness_analysis.py --analyze
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


DATA_DIR = Path("data/truthfulness")
RESULTS_DIR = Path("experiments/truthfulness_analysis/runs")


def prepare_truthfulness_dataset(n_per_source: int = 200, output_path: Path = None):
    """
    Prepare truthfulness dataset for attribution analysis.

    Sources:
    - TruthfulQA: Questions with truthful vs false answers
    - HaluEval: Hallucinated vs faithful responses (if available)
    """
    if not LOADER_AVAILABLE:
        raise RuntimeError("Dataset loader not available")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or DATA_DIR / "truthfulness_samples.json"

    print("Loading truthfulness dataset...")
    loader = DatasetLoader()

    all_samples = []

    # TruthfulQA - primary source for truthful vs false
    try:
        samples = loader.load("truthfulqa", n_samples=n_per_source * 2, split="validation")
        print(f"  truthfulqa: loaded {len(samples)} samples")

        # Label distribution
        labels = defaultdict(int)
        for s in samples:
            labels[s.label] += 1
        print(f"    Distribution: {dict(labels)}")

        all_samples.extend(samples)
    except Exception as e:
        print(f"  truthfulqa: FAILED - {e}")

    # HaluEval QA - hallucinated vs faithful
    for task in ["qa", "dialogue"]:
        try:
            samples = loader.load(f"halueval_{task}", n_samples=n_per_source, split="data")
            print(f"  halueval_{task}: loaded {len(samples)} samples")
            all_samples.extend(samples)
        except Exception as e:
            print(f"  halueval_{task}: FAILED - {e}")

    # Convert to serializable format
    serialized = []
    for i, sample in enumerate(all_samples):
        serialized.append({
            "idx": i,
            "text": sample.text[:1000],  # Truncate
            "source": sample.source,
            "domain": sample.domain,
            "label": sample.label,
            "truthfulness_class": _classify_truthfulness(sample.label),
        })

    # Save
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "n_samples": len(serialized),
                "purpose": "truthfulness_analysis",
            },
            "samples": serialized,
        }, f, indent=2)

    print(f"\nSaved {len(serialized)} samples to {output_path}")

    # Truthfulness distribution
    truth_dist = defaultdict(int)
    for s in serialized:
        truth_dist[s["truthfulness_class"]] += 1
    print(f"\nTruthfulness distribution: {dict(truth_dist)}")

    return output_path


def _classify_truthfulness(label: str) -> str:
    """Map labels to truthfulness class."""
    truthful = {"truthful", "faithful", "correct", "supported"}
    false = {"false", "hallucinated", "incorrect", "refuted"}

    label_lower = label.lower() if label else ""

    if any(t in label_lower for t in truthful):
        return "truthful"
    elif any(f in label_lower for f in false):
        return "false"
    else:
        return "unknown"


def analyze_truthfulness_results(results_path: Path = None):
    """
    Analyze truthfulness attribution results.

    Key question: Do truthful vs false statements show different activation patterns?
    """
    results_path = results_path or Path("data/results/truthfulness_metrics.json")

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run attribution first.")
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

    # Split by truthfulness
    if "truthfulness_class" not in df.columns:
        df["truthfulness_class"] = df["label"].apply(_classify_truthfulness)

    truthful = df[df["truthfulness_class"] == "truthful"]
    false = df[df["truthfulness_class"] == "false"]

    print(f"\nTruthful: {len(truthful)}")
    print(f"False: {len(false)}")

    # Metrics
    metrics = ["n_active", "n_edges", "mean_influence", "top_100_concentration",
               "mean_activation", "logit_entropy"]
    available_metrics = [m for m in metrics if m in df.columns]

    results = {"truthful_profile": {}, "false_profile": {}, "comparison": {}}

    # Compare distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, metric in enumerate(available_metrics[:6]):
        ax = axes[i]

        # Distributions
        if len(truthful) > 0:
            ax.hist(truthful[metric], bins=20, alpha=0.6, label="truthful", density=True)
        if len(false) > 0:
            ax.hist(false[metric], bins=20, alpha=0.6, label="false", density=True)

        # Effect size
        if len(truthful) > 5 and len(false) > 5:
            t_vals = truthful[metric].values
            f_vals = false[metric].values
            pooled_std = np.sqrt((np.var(t_vals, ddof=1) + np.var(f_vals, ddof=1)) / 2)
            d = (np.mean(t_vals) - np.mean(f_vals)) / pooled_std if pooled_std > 0 else 0

            results["comparison"][metric] = {
                "truthful_mean": float(np.mean(t_vals)),
                "false_mean": float(np.mean(f_vals)),
                "cohens_d": float(d),
                "significant": abs(d) > 0.2,
            }

            ax.set_title(f"{metric}\nCohen's d = {d:.2f}")
        else:
            ax.set_title(metric)

        ax.legend()
        ax.set_xlabel(metric)

    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "truthfulness_comparison.png", dpi=150)
    plt.close()

    # Save results
    with open(run_dir / "truthfulness_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate report
    report = generate_truthfulness_report(df, results, available_metrics)
    with open(run_dir / "truthfulness_report.md", "w") as f:
        f.write(report)

    print(f"\nAnalysis complete: {run_dir}")


def generate_truthfulness_report(df, results, metrics):
    """Generate markdown report."""
    lines = []
    lines.append("# Truthfulness Attribution Analysis")
    lines.append("")
    lines.append(f"**Samples:** {len(df)}")
    lines.append("")

    lines.append("## Key Question")
    lines.append("")
    lines.append("Do truthful vs. false statements activate different feature patterns?")
    lines.append("")

    lines.append("## Comparison")
    lines.append("")
    lines.append("| Metric | Truthful Mean | False Mean | Cohen's d | Significant |")
    lines.append("|--------|---------------|------------|-----------|-------------|")

    for metric, comp in results.get("comparison", {}).items():
        sig = "yes" if comp.get("significant") else "no"
        lines.append(f"| {metric} | {comp['truthful_mean']:.4f} | {comp['false_mean']:.4f} | {comp['cohens_d']:.2f} | {sig} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("*Patterns to investigate:*")
    lines.append("")
    lines.append("- Do false statements have more diffuse activation patterns?")
    lines.append("- Is there a 'truth feature' that fires differently?")
    lines.append("- How does this relate to hallucination detection?")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Truthfulness Analysis")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--n-per-source", type=int, default=200, help="Samples per source")

    args = parser.parse_args()

    if args.prepare:
        prepare_truthfulness_dataset(n_per_source=args.n_per_source)
    elif args.analyze:
        analyze_truthfulness_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
