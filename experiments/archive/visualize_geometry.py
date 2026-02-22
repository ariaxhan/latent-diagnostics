#!/usr/bin/env python3
"""
Visualization for Experiment B: Geometric Topology

Creates the key figure for the paper:
- Scatter plot: Sphericity vs L0 Norm
- Color-coded by Fact/Hallucination
- Faceted by domain

This is Figure 2 in the paper: "Topological Phase Plot"
"""

import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import ExperimentStorage


def create_topology_phase_plot(df: pl.DataFrame, output_path: Path):
    """
    Create the main topology phase plot (Figure 2).
    
    Scatter plot showing Sphericity vs Sparsity (L0 Norm).
    Facts should cluster in "Structured/Sparse" region.
    Hallucinations should cluster in "Isotropic/Diffuse" region.
    """
    # Convert to pandas for seaborn
    df_pd = df.to_pandas()
    
    # Create figure with subplots for each domain
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    domains = ["entity", "temporal", "logical", "adversarial"]
    domain_titles = {
        "entity": "Entity Swaps",
        "temporal": "Temporal Shifts",
        "logical": "Logical Inversions",
        "adversarial": "Adversarial Traps"
    }
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        domain_df = df_pd[df_pd['domain'] == domain]
        
        # Plot facts and hallucinations
        for condition, color, marker in [
            ('fact', '#2E86AB', 'o'),
            ('hallucination', '#A23B72', '^')
        ]:
            cond_df = domain_df[domain_df['condition'] == condition]
            ax.scatter(
                cond_df['l0_norm'],
                cond_df['sphericity'],
                c=color,
                marker=marker,
                alpha=0.6,
                s=50,
                label=condition.capitalize(),
                edgecolors='white',
                linewidth=0.5
            )
        
        # Add reference lines
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, 0.92, 'Spherical', 
                ha='right', va='bottom', fontsize=9, color='gray')
        
        # Styling
        ax.set_xlabel('L0 Norm (Sparsity)', fontsize=11)
        ax.set_ylabel('Sphericity (c/a)', fontsize=11)
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.05)
    
    plt.suptitle(
        'Figure 2: Topological Phase Plot\nSphericity vs Sparsity Across Domains',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_shape_distribution(df: pl.DataFrame, output_path: Path):
    """
    Create bar chart showing distribution of shape classes.
    """
    # Convert to pandas
    df_pd = df.to_pandas()
    
    # Count shape classes by condition
    shape_counts = df_pd.groupby(['condition', 'shape_class']).size().reset_index(name='count')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot grouped bar chart
    shapes = ["Spherical", "Oblate", "Prolate", "Triaxial"]
    x = range(len(shapes))
    width = 0.35
    
    fact_counts = [
        shape_counts[(shape_counts['condition'] == 'fact') & 
                     (shape_counts['shape_class'] == shape)]['count'].sum()
        for shape in shapes
    ]
    hall_counts = [
        shape_counts[(shape_counts['condition'] == 'hallucination') & 
                     (shape_counts['shape_class'] == shape)]['count'].sum()
        for shape in shapes
    ]
    
    ax.bar([i - width/2 for i in x], fact_counts, width, label='Fact', color='#2E86AB', alpha=0.8)
    ax.bar([i + width/2 for i in x], hall_counts, width, label='Hallucination', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Shape Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Shape Distribution: Facts vs Hallucinations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shapes)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_dimensionality_comparison(df: pl.DataFrame, output_path: Path):
    """
    Create violin plot comparing effective dimensionality.
    """
    df_pd = df.to_pandas()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(
        data=df_pd,
        x='domain',
        y='dimensionality',
        hue='condition',
        split=True,
        palette={'fact': '#2E86AB', 'hallucination': '#A23B72'},
        ax=ax
    )
    
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Effective Dimensionality', fontsize=12)
    ax.set_title('Effective Dimensionality: Facts vs Hallucinations', fontsize=14, fontweight='bold')
    ax.legend(title='Condition')
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations for Experiment B."""
    print("=" * 80)
    print("VISUALIZING EXPERIMENT B: GEOMETRIC TOPOLOGY")
    print("=" * 80)
    print()
    
    # Load results
    experiment_path = Path(__file__).parent / "02_geometry"
    
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
        print("  python experiments/02_geometry.py")
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
    
    # Figure 1: Main topology phase plot (THE KEY FIGURE)
    create_topology_phase_plot(
        df,
        output_dir / "fig2_topology_phase_plot.png"
    )
    
    # Figure 2: Shape distribution
    create_shape_distribution(
        df,
        output_dir / "shape_distribution.png"
    )
    
    # Figure 3: Dimensionality comparison
    create_dimensionality_comparison(
        df,
        output_dir / "dimensionality_comparison.png"
    )
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Figures saved to: {output_dir}")
    print()
    print("Key Figure for Paper:")
    print("  - fig2_topology_phase_plot.png: Main result showing geometric degradation")
    print()


if __name__ == "__main__":
    main()




