#!/usr/bin/env python3
"""
Visualization for Experiment D: Layer Sensitivity

Creates Figure 4 for the paper: "Layer Sensitivity"
- Line chart showing detection accuracy across layers
- Compares spectroscopy vs geometry methods
"""

import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import ExperimentStorage


def calculate_effect_size(fact_values, hall_values):
    """Calculate Cohen's d effect size."""
    mean_diff = np.mean(hall_values) - np.mean(fact_values)
    pooled_std = np.sqrt((np.var(fact_values) + np.var(hall_values)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0


def create_layer_sensitivity_plot(df: pl.DataFrame, output_path: Path):
    """
    Create Figure 4: Layer Sensitivity Analysis.
    
    Line chart showing effect sizes across layers for different metrics.
    """
    df_pd = df.to_pandas()
    
    layers = sorted(df_pd['layer'].unique())
    
    # Calculate effect sizes for each metric at each layer
    metrics = {
        'Reconstruction Error': 'reconstruction_error',
        'Sphericity': 'sphericity',
        'Gini Coefficient': 'gini_coefficient',
        'L0 Norm': 'l0_norm'
    }
    
    effect_sizes = {metric: [] for metric in metrics}
    
    for layer in layers:
        layer_df = df_pd[df_pd['layer'] == layer]
        fact_df = layer_df[layer_df['condition'] == 'fact']
        hall_df = layer_df[layer_df['condition'] == 'hallucination']
        
        for metric_name, metric_col in metrics.items():
            fact_vals = fact_df[metric_col].values
            hall_vals = hall_df[metric_col].values
            
            effect_size = calculate_effect_size(fact_vals, hall_vals)
            effect_sizes[metric_name].append(effect_size)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        'Reconstruction Error': '#E63946',
        'Sphericity': '#457B9D',
        'Gini Coefficient': '#2A9D8F',
        'L0 Norm': '#F4A261'
    }
    
    markers = {
        'Reconstruction Error': 'o',
        'Sphericity': 's',
        'Gini Coefficient': '^',
        'L0 Norm': 'D'
    }
    
    for metric_name in metrics:
        ax.plot(
            layers,
            effect_sizes[metric_name],
            marker=markers[metric_name],
            color=colors[metric_name],
            linewidth=2.5,
            markersize=10,
            label=metric_name,
            alpha=0.8
        )
    
    # Add reference lines for effect size interpretation
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    ax.text(layers[-1] + 0.5, 0.2, 'Small', fontsize=9, color='gray', va='center')
    ax.text(layers[-1] + 0.5, 0.5, 'Medium', fontsize=9, color='gray', va='center')
    ax.text(layers[-1] + 0.5, 0.8, 'Large', fontsize=9, color='gray', va='center')
    
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel("Effect Size (Cohen's d)", fontsize=13, fontweight='bold')
    ax.set_title(
        "Figure 4: Layer Sensitivity Analysis\nDetection Accuracy Across Model Layers",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(layers)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_layer_comparison_heatmap(df: pl.DataFrame, output_path: Path):
    """
    Create heatmap showing mean values across layers and conditions.
    """
    df_pd = df.to_pandas()
    
    layers = sorted(df_pd['layer'].unique())
    metrics = ['reconstruction_error', 'sphericity', 'gini_coefficient', 'l0_norm']
    metric_labels = ['Reconstruction\nError', 'Sphericity', 'Gini\nCoefficient', 'L0 Norm']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, condition in enumerate(['fact', 'hallucination']):
        ax = axes[idx]
        
        # Build matrix
        matrix = []
        for layer in layers:
            row = []
            for metric in metrics:
                layer_cond_df = df_pd[
                    (df_pd['layer'] == layer) & 
                    (df_pd['condition'] == condition)
                ]
                mean_val = layer_cond_df[metric].mean()
                row.append(mean_val)
            matrix.append(row)
        
        matrix = np.array(matrix).T  # Transpose for better visualization
        
        # Normalize each row for better color scaling
        matrix_norm = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            row_min, row_max = matrix[i].min(), matrix[i].max()
            if row_max > row_min:
                matrix_norm[i] = (matrix[i] - row_min) / (row_max - row_min)
            else:
                matrix_norm[i] = 0.5
        
        im = ax.imshow(matrix_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metric_labels)
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_title(condition.capitalize(), fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(layers)):
                text = ax.text(
                    j, i, f'{matrix[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if matrix_norm[i, j] > 0.5 else "black",
                    fontsize=9
                )
    
    plt.suptitle(
        'Layer-wise Metric Comparison',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_domain_layer_interaction(df: pl.DataFrame, output_path: Path):
    """
    Create plot showing how different domains respond across layers.
    """
    df_pd = df.to_pandas()
    
    layers = sorted(df_pd['layer'].unique())
    domains = ["entity", "temporal", "logical", "adversarial"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    domain_titles = {
        "entity": "Entity Swaps",
        "temporal": "Temporal Shifts",
        "logical": "Logical Inversions",
        "adversarial": "Adversarial Traps"
    }
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        
        # Calculate effect sizes for reconstruction error
        effect_sizes = []
        for layer in layers:
            layer_domain_df = df_pd[(df_pd['layer'] == layer) & (df_pd['domain'] == domain)]
            fact_vals = layer_domain_df[layer_domain_df['condition'] == 'fact']['reconstruction_error'].values
            hall_vals = layer_domain_df[layer_domain_df['condition'] == 'hallucination']['reconstruction_error'].values
            
            if len(fact_vals) > 0 and len(hall_vals) > 0:
                effect_size = calculate_effect_size(fact_vals, hall_vals)
                effect_sizes.append(effect_size)
            else:
                effect_sizes.append(0)
        
        ax.plot(layers, effect_sizes, marker='o', linewidth=2.5, markersize=10, color='#457B9D')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel("Effect Size (Cohen's d)", fontsize=11)
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xticks(layers)
    
    plt.suptitle(
        'Domain-Specific Layer Sensitivity\n(Reconstruction Error)',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations for Experiment D."""
    print("=" * 80)
    print("VISUALIZING EXPERIMENT D: LAYER SENSITIVITY")
    print("=" * 80)
    print()
    
    # Load results
    experiment_path = Path(__file__).parent / "04_layer_sensitivity"
    
    # Check for existing runs first to avoid creating a new run directory
    runs_path = experiment_path / "runs"
    latest_run = None
    
    if runs_path.exists():
        runs = [
            d.name for d in runs_path.iterdir()
            if d.is_dir() and (d / "metrics.parquet").exists()
        ]
        runs = sorted(runs, reverse=True)
        latest_run = runs[0] if runs else None
    
    if not latest_run:
        print("Error: No experiment runs found. Run the experiment first:")
        print("  python experiments/04_layer_sensitivity.py")
        return
    
    # Initialize storage with latest run
    storage = ExperimentStorage(experiment_path, run_id=latest_run)
    
    print("Loading results...")
    print(f"  Using run: {latest_run}")
    df = storage.read_metrics()
    print(f"  ✓ Loaded {len(df)} records")
    print()
    
    # Create output directory
    output_dir = experiment_path / "figures"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating figures...")
    
    # Figure 1: Main layer sensitivity plot (KEY FIGURE)
    create_layer_sensitivity_plot(
        df,
        output_dir / "fig4_layer_sensitivity.png"
    )
    
    # Figure 2: Heatmap comparison
    create_layer_comparison_heatmap(
        df,
        output_dir / "layer_comparison_heatmap.png"
    )
    
    # Figure 3: Domain-layer interaction
    create_domain_layer_interaction(
        df,
        output_dir / "domain_layer_interaction.png"
    )
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Figures saved to: {output_dir}")
    print()
    print("Key Figure for Paper:")
    print("  - fig4_layer_sensitivity.png: Shows where hallucinations emerge in the network")
    print()


if __name__ == "__main__":
    main()




