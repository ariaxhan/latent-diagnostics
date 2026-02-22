#!/usr/bin/env python3
"""
Visualization: Injection Detection Results

Generates:
1. ROC Curve with AUC score
2. Precision-Recall Curve
3. Confusion Matrix
4. Metric Distribution (injection vs benign)
5. Threshold Sensitivity Analysis
6. Per-Category Performance

Usage:
    python experiments/visualize.py
    python experiments/visualize.py --run-id 20261221_120000
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import ExperimentStorage


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def load_experiment_data(run_id: str = None):
    """Load experiment results."""
    experiment_path = Path(__file__).parent / "injection_detection"
    storage = ExperimentStorage(experiment_path, run_id=run_id or storage.get_latest_run())

    # If no run_id, get latest
    if run_id is None:
        latest = storage.get_latest_run()
        if latest is None:
            raise FileNotFoundError("No runs found. Run experiment first.")
        storage = ExperimentStorage(experiment_path, run_id=latest)

    df = storage.read_metrics()
    manifest = storage.read_manifest()

    return df, manifest, storage


def plot_roc_curve(df: pl.DataFrame, save_path: Path):
    """Plot ROC curve with AUC."""
    from sklearn.metrics import roc_curve, auc

    labels = df["label"].to_numpy()
    confidence = df["confidence"].to_numpy()

    fpr, tpr, thresholds = roc_curve(labels, confidence)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Attribution Detector (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Mark Lakera Guard performance
    lakera_recall = 0.95  # Approximate
    ax.axhline(y=lakera_recall, color='r', linestyle=':', alpha=0.7, label=f'Lakera Guard (~{lakera_recall*100:.0f}% recall)')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve: Injection Detection')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_precision_recall(df: pl.DataFrame, save_path: Path):
    """Plot Precision-Recall curve."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    labels = df["label"].to_numpy()
    confidence = df["confidence"].to_numpy()

    precision, recall, thresholds = precision_recall_curve(labels, confidence)
    ap = average_precision_score(labels, confidence)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(recall, precision, 'b-', linewidth=2, label=f'Attribution Detector (AP = {ap:.3f})')

    # Mark baseline (random classifier)
    baseline = labels.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.2f})')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve: Injection Detection')
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_confusion_matrix(df: pl.DataFrame, save_path: Path):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix

    labels = df["label"].to_numpy()
    predictions = df["prediction"].to_numpy()

    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Injection'],
        yticklabels=['Benign', 'Injection'],
        ax=ax,
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix: Injection Detection')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_metric_distributions(df: pl.DataFrame, save_path: Path):
    """Plot distribution of key metrics for injection vs benign."""
    metrics = ['n_active', 'top_100_concentration', 'mean_influence', 'n_edges']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        # Split by label
        injection = df.filter(pl.col("label") == True)[metric].to_numpy()
        benign = df.filter(pl.col("label") == False)[metric].to_numpy()

        # Plot histograms
        ax.hist(benign, bins=30, alpha=0.6, label='Benign', color='green', density=True)
        ax.hist(injection, bins=30, alpha=0.6, label='Injection', color='red', density=True)

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.legend()

        # Add effect size annotation
        if len(injection) > 0 and len(benign) > 0:
            inj_mean, ben_mean = injection.mean(), benign.mean()
            pooled_std = np.sqrt((injection.std()**2 + benign.std()**2) / 2)
            cohen_d = abs(inj_mean - ben_mean) / pooled_std if pooled_std > 0 else 0
            ax.annotate(f"Cohen's d = {cohen_d:.2f}", xy=(0.7, 0.9), xycoords='axes fraction')

    plt.suptitle('Metric Distributions: Injection vs Benign', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_per_category(df: pl.DataFrame, save_path: Path):
    """Plot accuracy by category."""
    categories = df["category"].unique().to_list()

    accuracies = []
    counts = []

    for cat in categories:
        cat_df = df.filter(pl.col("category") == cat)
        acc = cat_df["correct"].mean()
        n = len(cat_df)
        accuracies.append(acc * 100)
        counts.append(n)

    # Sort by accuracy
    sorted_idx = np.argsort(accuracies)[::-1]
    categories = [categories[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green' if a >= 95.22 else 'orange' if a >= 80 else 'red' for a in accuracies]

    bars = ax.barh(range(len(categories)), accuracies, color=colors, alpha=0.7)

    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + 1, i, f'n={count}', va='center', fontsize=10)

    # Add Lakera baseline
    ax.axvline(x=95.22, color='blue', linestyle='--', linewidth=2, label='Lakera Guard (95.22%)')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Category')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def plot_benchmark_comparison(manifest: dict, save_path: Path):
    """Plot comparison against baselines."""
    baselines = manifest.get("baselines", {})
    our_score = manifest.get("test_results", {}).get("pint_score", 0)

    # Add our result
    all_scores = {**baselines, "Attribution Detector (Ours)": our_score}

    # Sort by score
    sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if 'Ours' in n else 'steelblue' for n in names]
    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8)

    # Add score annotations
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.5, i, f'{score:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('PINT Score (%)')
    ax.set_title('PINT Benchmark Comparison')
    ax.set_xlim([0, 100])

    # Add 95% line
    ax.axvline(x=95, color='green', linestyle=':', alpha=0.5, linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def create_all_visualizations(run_id: str = None):
    """Generate all visualization figures."""
    print("=" * 80)
    print("VISUALIZING INJECTION DETECTION RESULTS")
    print("=" * 80)
    print()

    setup_style()

    # Load data
    experiment_path = Path(__file__).parent / "injection_detection"

    # Handle run_id
    if run_id is None:
        storage_temp = ExperimentStorage(experiment_path)
        run_id = storage_temp.get_latest_run()
        if run_id is None:
            print("ERROR: No runs found. Run experiment first:")
            print("  python experiments/08_injection_detection.py")
            return

    storage = ExperimentStorage(experiment_path, run_id=run_id)

    print(f"Loading run: {run_id}")
    df = storage.read_metrics()
    manifest = storage.read_manifest()
    print(f"✓ Loaded {len(df)} samples")
    print()

    # Create figures directory
    figures_dir = storage.run_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Generating figures...")
    print("-" * 80)

    # Generate all plots
    try:
        plot_roc_curve(df, figures_dir / "fig1_roc_curve.png")
    except Exception as e:
        print(f"  ⚠ ROC curve failed: {e}")

    try:
        plot_precision_recall(df, figures_dir / "fig2_precision_recall.png")
    except Exception as e:
        print(f"  ⚠ PR curve failed: {e}")

    try:
        plot_confusion_matrix(df, figures_dir / "fig3_confusion_matrix.png")
    except Exception as e:
        print(f"  ⚠ Confusion matrix failed: {e}")

    try:
        plot_metric_distributions(df, figures_dir / "fig4_metric_distributions.png")
    except Exception as e:
        print(f"  ⚠ Distributions failed: {e}")

    if "category" in df.columns:
        try:
            plot_per_category(df, figures_dir / "fig5_per_category.png")
        except Exception as e:
            print(f"  ⚠ Per-category failed: {e}")

    try:
        plot_benchmark_comparison(manifest, figures_dir / "fig6_benchmark_comparison.png")
    except Exception as e:
        print(f"  ⚠ Benchmark comparison failed: {e}")

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Figures saved to: {figures_dir}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None, help="Specific run ID to visualize")
    args = parser.parse_args()

    create_all_visualizations(run_id=args.run_id)
