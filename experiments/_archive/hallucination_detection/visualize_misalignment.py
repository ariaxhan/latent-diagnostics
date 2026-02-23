#!/usr/bin/env python3
"""
Visualization for Experiment E: Misalignment Results

Creates key figures demonstrating semantic drift:
- Fig 1: Drift Difference Distribution (Hall - Fact)
- Fig 2: Twist Difference Distribution (Hall - Fact)
- Fig 3: Scatter Plot (Drift vs Twist) showing misalignment signature
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph.storage import ExperimentStorage
import polars as pl


def create_misalignment_figures(run_id=None):
    """
    Create visualization figures for misalignment experiment.
    
    Args:
        run_id: Optional specific run ID. If None, uses latest run.
    """
    
    print("=" * 80)
    print("MISALIGNMENT VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load data
    experiment_path = Path(__file__).parent / "05_misalignment"
    
    # Check for existing runs first to avoid creating a new run directory
    runs_path = experiment_path / "runs"
    if run_id is None:
        if runs_path.exists():
            runs = [
                d.name for d in runs_path.iterdir()
                if d.is_dir() and (d / "metrics.parquet").exists()
            ]
            runs = sorted(runs, reverse=True)
            run_id = runs[0] if runs else None
    
    if not run_id:
        print("Error: No experiment runs found. Run the experiment first:")
        print("  python experiments/05_misalignment.py")
        return
    
    # Initialize storage with the run_id
    storage = ExperimentStorage(experiment_path, run_id=run_id)
    print(f"Using run: {run_id}")
    
    df = storage.read_metrics()
    
    # Convert to pandas for seaborn
    df_pd = df.to_pandas()
    
    print(f"Loaded {len(df)} rows from {run_id}")
    print()
    
    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Create figures directory
    figures_dir = storage.run_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Figure 1: Drift Difference Distribution
    print("Creating Figure 1: Drift Difference Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 1: Semantic Drift - Centroid Angular Deviation", 
                 fontsize=16, fontweight='bold')
    
    domains = ["adversarial", "temporal"]
    domain_titles = {
        "adversarial": "Adversarial Traps",
        "temporal": "Temporal Shifts"
    }
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        
        domain_data = df_pd[df_pd['domain'] == domain]
        
        # Histogram of drift differences
        sns.histplot(
            data=domain_data,
            x='drift_diff',
            bins=30,
            alpha=0.7,
            ax=ax,
            stat='density',
            color='coral',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Drift')
        
        # Add mean line
        mean_drift = domain_data['drift_diff'].mean()
        ax.axvline(x=mean_drift, color='red', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_drift:.2f}°')
        
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('Drift Difference (Hallucination - Fact) [degrees]', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = figures_dir / "fig1_drift_difference.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close()
    
    # Figure 2: Twist Difference Distribution
    print("Creating Figure 2: Twist Difference Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 2: Axis Twist - Principal Axis Rotation", 
                 fontsize=16, fontweight='bold')
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        
        domain_data = df_pd[df_pd['domain'] == domain]
        
        # Histogram of twist differences
        sns.histplot(
            data=domain_data,
            x='twist_diff',
            bins=30,
            alpha=0.7,
            ax=ax,
            stat='density',
            color='skyblue',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Twist')
        
        # Add mean line
        mean_twist = domain_data['twist_diff'].mean()
        ax.axvline(x=mean_twist, color='blue', linestyle='-', linewidth=2,
                  label=f'Mean: {mean_twist:.2f}°')
        
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('Twist Difference (Hallucination - Fact) [degrees]', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = figures_dir / "fig2_twist_difference.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close()
    
    # Figure 3: Scatter Plot (Drift vs Twist) - The Misalignment Signature
    print("Creating Figure 3: Misalignment Signature (Drift vs Twist)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 3: Misalignment Signature - Drift vs Twist", 
                 fontsize=16, fontweight='bold')
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        
        domain_data = df_pd[df_pd['domain'] == domain]
        
        # Scatter plot
        scatter = ax.scatter(
            domain_data['drift_diff'],
            domain_data['twist_diff'],
            alpha=0.6,
            s=50,
            c=domain_data['complexity'],
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add quadrant lines
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.05, 0.95, 'High Drift\nHigh Twist', 
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='coral', alpha=0.3))
        
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('Drift Difference [degrees]', fontsize=10)
        ax.set_ylabel('Twist Difference [degrees]', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Complexity', fontsize=9)
    
    plt.tight_layout()
    fig_path = figures_dir / "fig3_misalignment_signature.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close()
    
    # Figure 4: Comparison Box Plot
    print("Creating Figure 4: Comparison Box Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 4: Misalignment Metrics Comparison", 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for box plot
    plot_data = []
    for domain in domains:
        domain_data = df_pd[df_pd['domain'] == domain]
        for _, row in domain_data.iterrows():
            plot_data.append({
                'domain': domain_titles[domain],
                'metric': 'Drift Difference',
                'value': row['drift_diff']
            })
            plot_data.append({
                'domain': domain_titles[domain],
                'metric': 'Twist Difference',
                'value': row['twist_diff']
            })
    
    plot_df = pl.DataFrame(plot_data).to_pandas()
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        
        domain_plot_data = plot_df[plot_df['domain'] == domain_titles[domain]]
        
        sns.boxplot(
            data=domain_plot_data,
            x='metric',
            y='value',
            ax=ax,
            palette=['coral', 'skyblue']
        )
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=10)
        ax.set_ylabel('Difference [degrees]', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = figures_dir / "fig4_comparison_boxplot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close()
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All figures saved to: {figures_dir}")
    print()


if __name__ == "__main__":
    create_misalignment_figures()

