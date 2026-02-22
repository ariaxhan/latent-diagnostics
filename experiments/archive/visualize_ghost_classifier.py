#!/usr/bin/env python3
"""
Visualization for Experiment 07: Ghost Feature Classifier

Creates comprehensive visualizations:
1. Universal Ghost Frequency Distribution
2. Semantic Word Clouds for Top Ghosts
3. Mechanistic Antagonism Heatmap
4. Classifier Performance Comparison
5. Feature Importance Rankings
6. ROC Curves
7. Confusion Matrices
"""

import sys
from pathlib import Path
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_experiment_results(run_path: Path) -> Dict:
    """Load all experiment results."""
    
    # Load manifest
    with open(run_path / "manifest.json") as f:
        manifest = json.load(f)
    
    # Load metrics
    metrics_file = list(run_path.glob("metrics_*.parquet"))[0]
    df = pl.read_parquet(metrics_file)
    
    # Load universal ghosts
    with open(run_path / "universal_ghosts.json") as f:
        universal_ghosts = json.load(f)
    
    # Load antagonism analysis
    with open(run_path / "antagonism_analysis.json") as f:
        antagonism = json.load(f)
    
    # Load classifier results
    with open(run_path / "classifier_results.json") as f:
        classifiers = json.load(f)
    
    return {
        "manifest": manifest,
        "metrics": df,
        "universal_ghosts": universal_ghosts,
        "antagonism": antagonism,
        "classifiers": classifiers,
    }


def plot_ghost_frequency_distribution(universal_ghosts: List[Dict], save_path: Path):
    """Plot frequency distribution of universal ghosts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    frequencies = [g["frequency"] for g in universal_ghosts]
    avg_magnitudes = [g["avg_magnitude"] for g in universal_ghosts]
    
    # 1. Frequency histogram
    ax = axes[0, 0]
    ax.hist(frequencies, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Frequency (# samples)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Universal Ghost Frequency Distribution", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 2. Magnitude histogram
    ax = axes[0, 1]
    ax.hist(avg_magnitudes, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Average Magnitude", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Universal Ghost Magnitude Distribution", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 3. Frequency vs Magnitude scatter
    ax = axes[1, 0]
    ax.scatter(frequencies, avg_magnitudes, alpha=0.6, s=50, c='purple')
    ax.set_xlabel("Frequency (# samples)", fontsize=12)
    ax.set_ylabel("Average Magnitude", fontsize=12)
    ax.set_title("Frequency vs Magnitude", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Top 20 ghosts by frequency
    ax = axes[1, 1]
    top_20 = sorted(universal_ghosts, key=lambda x: x["frequency"], reverse=True)[:20]
    feature_ids = [f"F{g['feature_id']}" for g in top_20]
    freqs = [g["frequency"] for g in top_20]
    
    y_pos = np.arange(len(feature_ids))
    ax.barh(y_pos, freqs, color='teal', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_ids, fontsize=8)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title("Top 20 Universal Ghosts", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_semantic_word_clouds(universal_ghosts: List[Dict], save_path: Path):
    """Plot semantic interpretation of top ghosts."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    top_10 = sorted(universal_ghosts, key=lambda x: x["frequency"], reverse=True)[:10]
    
    for idx, ghost in enumerate(top_10):
        ax = axes[idx]
        
        # Get top words and logits
        words = ghost["top_words"][:8]
        logits = ghost["top_logits"][:8]
        
        # Normalize logits for bar heights
        logits_norm = np.array(logits) - min(logits)
        if max(logits_norm) > 0:
            logits_norm = logits_norm / max(logits_norm)
        
        # Plot
        y_pos = np.arange(len(words))
        ax.barh(y_pos, logits_norm, color='mediumseagreen', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel("Normalized Logit", fontsize=10)
        ax.set_title(f"Feature #{ghost['feature_id']} (freq={ghost['frequency']})", 
                     fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_antagonism_heatmap(antagonism: List[Dict], universal_ghosts: List[Dict], save_path: Path):
    """Plot mechanistic antagonism scores."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sort by antagonism score
    sorted_ant = sorted(antagonism, key=lambda x: x["antagonism_score"], reverse=True)
    
    # Top 30 most antagonistic
    top_30 = sorted_ant[:30]
    feature_ids = [f"F{a['feature_id']}" for a in top_30]
    antagonism_scores = [a["antagonism_score"] for a in top_30]
    dot_products = [a["avg_dot_with_fact_token"] for a in top_30]
    
    # 1. Antagonism scores
    ax = axes[0]
    y_pos = np.arange(len(feature_ids))
    colors = ['red' if s > 0 else 'blue' for s in antagonism_scores]
    ax.barh(y_pos, antagonism_scores, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_ids, fontsize=8)
    ax.set_xlabel("Antagonism Score (higher = more antagonistic)", fontsize=12)
    ax.set_title("Top 30 Most Antagonistic Ghost Features", fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    # 2. Dot products with fact tokens
    ax = axes[1]
    colors = ['green' if d > 0 else 'red' for d in dot_products]
    ax.barh(y_pos, dot_products, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_ids, fontsize=8)
    ax.set_xlabel("Avg Dot Product with Fact Token", fontsize=12)
    ax.set_title("Alignment with Truth (negative = opposes)", fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_classifier_performance(classifiers: List[Dict], save_path: Path):
    """Plot classifier performance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [c["model_name"] for c in classifiers]
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    accuracies = [c["accuracy"] for c in classifiers]
    ax.bar(model_names, accuracies, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classifier Accuracy Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
    
    # 2. F1 Score comparison
    ax = axes[0, 1]
    f1_scores = [c["f1"] for c in classifiers]
    ax.bar(model_names, f1_scores, color='coral', alpha=0.7, edgecolor='black')
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
    
    # 3. ROC-AUC comparison
    ax = axes[1, 0]
    roc_aucs = [c["roc_auc"] for c in classifiers]
    ax.bar(model_names, roc_aucs, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("ROC-AUC Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(0.8, color='red', linestyle='--', linewidth=2, label='Strong Support Threshold')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    for i, v in enumerate(roc_aucs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
    
    # 4. Cross-validation scores
    ax = axes[1, 1]
    for i, classifier in enumerate(classifiers):
        cv_scores = classifier["cross_val_scores"]
        x = np.arange(len(cv_scores)) + i * 0.25
        ax.plot(x, cv_scores, marker='o', label=classifier["model_name"], linewidth=2)
    
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Scores", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_feature_importance(classifiers: List[Dict], universal_ghosts: List[Dict], save_path: Path):
    """Plot feature importance for each classifier."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]
        
        # Get feature importance
        feature_importance = classifier["feature_importance"]
        
        # Sort and get top 20
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        feature_ids = [f"F{int(f[0])}" for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Plot
        y_pos = np.arange(len(feature_ids))
        ax.barh(y_pos, importances, color='purple', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_ids, fontsize=8)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"{classifier['model_name']}\nTop 20 Features", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_confusion_matrices(classifiers: List[Dict], save_path: Path):
    """Plot confusion matrices for all classifiers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]
        
        cm = np.array(classifier["confusion_matrix"])
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=['Fact', 'Hallucination'],
                    yticklabels=['Fact', 'Hallucination'],
                    ax=ax, cbar_kws={'label': 'Proportion'})
        
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(f"{classifier['model_name']}\nConfusion Matrix", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def create_comprehensive_dashboard(results: Dict, save_path: Path):
    """Create a comprehensive dashboard with all key findings."""
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
    
    universal_ghosts = results["universal_ghosts"]
    antagonism = results["antagonism"]
    classifiers = results["classifiers"]
    df = results["metrics"]
    
    # Row 1: Ghost frequency and magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    frequencies = [g["frequency"] for g in universal_ghosts]
    ax1.hist(frequencies, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Frequency", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Ghost Frequency Distribution", fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    avg_magnitudes = [g["avg_magnitude"] for g in universal_ghosts]
    ax2.hist(avg_magnitudes, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Avg Magnitude", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Ghost Magnitude Distribution", fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ghost_counts = df["ghost_count"].to_list()
    ax3.hist(ghost_counts, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Ghost Count per Sample", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Ghost Count Distribution", fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Row 2: Top ghosts
    ax4 = fig.add_subplot(gs[1, :])
    top_15 = sorted(universal_ghosts, key=lambda x: x["frequency"], reverse=True)[:15]
    feature_ids = [f"F{g['feature_id']}" for g in top_15]
    freqs = [g["frequency"] for g in top_15]
    words = [", ".join(g["top_words"][:3]) for g in top_15]
    
    y_pos = np.arange(len(feature_ids))
    bars = ax4.barh(y_pos, freqs, color='teal', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{fid}\n{w}" for fid, w in zip(feature_ids, words)], fontsize=8)
    ax4.set_xlabel("Frequency", fontsize=11)
    ax4.set_title("Top 15 Universal Ghost Features (with semantic interpretation)", 
                  fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(alpha=0.3, axis='x')
    
    # Row 3: Antagonism
    ax5 = fig.add_subplot(gs[2, :])
    sorted_ant = sorted(antagonism, key=lambda x: x["antagonism_score"], reverse=True)[:20]
    feature_ids_ant = [f"F{a['feature_id']}" for a in sorted_ant]
    antagonism_scores = [a["antagonism_score"] for a in sorted_ant]
    
    y_pos = np.arange(len(feature_ids_ant))
    colors = ['red' if s > 0 else 'blue' for s in antagonism_scores]
    ax5.barh(y_pos, antagonism_scores, color=colors, alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(feature_ids_ant, fontsize=8)
    ax5.set_xlabel("Antagonism Score", fontsize=11)
    ax5.set_title("Top 20 Most Antagonistic Ghost Features (oppose truth)", 
                  fontsize=12, fontweight='bold')
    ax5.axvline(0, color='black', linestyle='--', linewidth=1)
    ax5.invert_yaxis()
    ax5.grid(alpha=0.3, axis='x')
    
    # Row 4: Classifier performance
    ax6 = fig.add_subplot(gs[3, 0])
    model_names = [c["model_name"] for c in classifiers]
    accuracies = [c["accuracy"] for c in classifiers]
    ax6.bar(model_names, accuracies, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.set_ylabel("Accuracy", fontsize=10)
    ax6.set_title("Accuracy", fontsize=11, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.grid(alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=8)
    
    ax7 = fig.add_subplot(gs[3, 1])
    f1_scores = [c["f1"] for c in classifiers]
    ax7.bar(model_names, f1_scores, color='coral', alpha=0.7, edgecolor='black')
    ax7.set_ylabel("F1 Score", fontsize=10)
    ax7.set_title("F1 Score", fontsize=11, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.grid(alpha=0.3, axis='y')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=8)
    
    ax8 = fig.add_subplot(gs[3, 2])
    roc_aucs = [c["roc_auc"] for c in classifiers]
    ax8.bar(model_names, roc_aucs, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax8.set_ylabel("ROC-AUC", fontsize=10)
    ax8.set_title("ROC-AUC", fontsize=11, fontweight='bold')
    ax8.set_ylim([0, 1])
    ax8.axhline(0.8, color='red', linestyle='--', linewidth=2, label='Strong Support')
    ax8.grid(alpha=0.3, axis='y')
    ax8.legend(fontsize=8)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=8)
    
    # Row 5: Feature importance (best model)
    best_model = max(classifiers, key=lambda x: x["roc_auc"])
    ax9 = fig.add_subplot(gs[4, :])
    feature_importance = best_model["feature_importance"]
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    feature_ids_imp = [f"F{int(f[0])}" for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    y_pos = np.arange(len(feature_ids_imp))
    ax9.barh(y_pos, importances, color='purple', alpha=0.7)
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels(feature_ids_imp, fontsize=8)
    ax9.set_xlabel("Importance", fontsize=11)
    ax9.set_title(f"Top 20 Most Important Features ({best_model['model_name']})", 
                  fontsize=12, fontweight='bold')
    ax9.invert_yaxis()
    ax9.grid(alpha=0.3, axis='x')
    
    # Row 6: Summary statistics
    ax10 = fig.add_subplot(gs[5, :])
    ax10.axis('off')
    
    summary_text = f"""
    EXPERIMENT 07: GHOST FEATURE CLASSIFIER - SUMMARY
    
    Dataset:
    • Total Samples: {len(df)}
    • Unique Ghost Features: {results['manifest']['unique_ghost_features']}
    • Universal Ghosts (freq ≥ 10): {len(universal_ghosts)}
    
    Best Classifier: {best_model['model_name']}
    • Accuracy: {best_model['accuracy']:.4f}
    • Precision: {best_model['precision']:.4f}
    • Recall: {best_model['recall']:.4f}
    • F1 Score: {best_model['f1']:.4f}
    • ROC-AUC: {best_model['roc_auc']:.4f}
    
    Thesis B Validation:
    """
    
    best_auc = best_model['roc_auc']
    if best_auc > 0.8:
        summary_text += f"    ✓ STRONG SUPPORT (AUC = {best_auc:.4f})\n"
        summary_text += "    → Ghost features are predictive biomarkers of hallucination\n"
    elif best_auc > 0.7:
        summary_text += f"    ○ MODERATE SUPPORT (AUC = {best_auc:.4f})\n"
        summary_text += "    → Ghost features show some predictive power\n"
    else:
        summary_text += f"    ✗ WEAK SUPPORT (AUC = {best_auc:.4f})\n"
        summary_text += "    → Ghost features may not be strongly predictive\n"
    
    ax10.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def visualize_ghost_classifier():
    """
    Main visualization function for Experiment 07.
    """
    print("=" * 80)
    print("VISUALIZATION: GHOST FEATURE CLASSIFIER")
    print("=" * 80)
    print()
    
    # Find latest run
    experiment_path = Path(__file__).parent / "07_ghost_classifier"
    runs_path = experiment_path / "runs"
    
    if not runs_path.exists():
        print("ERROR: No runs found. Please run the experiment first:")
        print("  python experiments/07_ghost_classifier.py")
        return
    
    # Get latest run
    run_dirs = sorted(runs_path.iterdir(), key=lambda x: x.name, reverse=True)
    if not run_dirs:
        print("ERROR: No run directories found.")
        return
    
    latest_run = run_dirs[0]
    print(f"Loading results from: {latest_run}")
    print()
    
    # Load results
    results = load_experiment_results(latest_run)
    
    # Create figures directory
    figures_path = latest_run / "figures"
    figures_path.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    print("-" * 80)
    
    # 1. Ghost frequency distribution
    plot_ghost_frequency_distribution(
        results["universal_ghosts"],
        figures_path / "01_ghost_frequency.png"
    )
    
    # 2. Semantic word clouds
    plot_semantic_word_clouds(
        results["universal_ghosts"],
        figures_path / "02_semantic_interpretation.png"
    )
    
    # 3. Antagonism heatmap
    plot_antagonism_heatmap(
        results["antagonism"],
        results["universal_ghosts"],
        figures_path / "03_antagonism_analysis.png"
    )
    
    # 4. Classifier performance
    plot_classifier_performance(
        results["classifiers"],
        figures_path / "04_classifier_performance.png"
    )
    
    # 5. Feature importance
    plot_feature_importance(
        results["classifiers"],
        results["universal_ghosts"],
        figures_path / "05_feature_importance.png"
    )
    
    # 6. Confusion matrices
    plot_confusion_matrices(
        results["classifiers"],
        figures_path / "06_confusion_matrices.png"
    )
    
    # 7. Comprehensive dashboard
    create_comprehensive_dashboard(
        results,
        figures_path / "00_comprehensive_dashboard.png"
    )
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"All figures saved to: {figures_path}")
    print()
    print("Key Visualizations:")
    print("  1. 00_comprehensive_dashboard.png - Complete overview")
    print("  2. 01_ghost_frequency.png - Distribution analysis")
    print("  3. 02_semantic_interpretation.png - What ghosts represent")
    print("  4. 03_antagonism_analysis.png - Mechanistic opposition to truth")
    print("  5. 04_classifier_performance.png - Predictive validity")
    print("  6. 05_feature_importance.png - Most diagnostic ghosts")
    print("  7. 06_confusion_matrices.png - Classification errors")
    print()


if __name__ == "__main__":
    visualize_ghost_classifier()



