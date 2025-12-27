#!/usr/bin/env python3
"""
Visualization for Experiment A: Spectroscopy Results

Creates the key figures for the paper:
- Fig 1: Reconstruction Error Histogram (Fact vs Hallucination)
- Faceted by domain
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hallucination_detector.storage import ExperimentStorage
import polars as pl


def create_spectroscopy_figures(run_id=None):
    """
    Create visualization figures for spectroscopy experiment.
    
    Args:
        run_id: Optional specific run ID. If None, uses latest run.
    """
    
    print("=" * 80)
    print("SPECTROSCOPY VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load data
    experiment_path = Path(__file__).parent / "01_spectroscopy"
    storage = ExperimentStorage(experiment_path, run_id=run_id)
    
    if run_id is None:
        run_id = storage.get_latest_run()
        print(f"Using latest run: {run_id}")
    
    df = storage.read_metrics(run_id)
    
    # Convert to pandas for seaborn
    df_pd = df.to_pandas()
    
    print(f"Loaded {len(df)} rows from {run_id}")
    print()
    
    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Figure 1: Reconstruction Error by Domain
    print("Creating Figure 1: Reconstruction Error Distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 1: The Spectral Shift - Reconstruction Error by Domain", 
                 fontsize=16, fontweight='bold')
    
    domains = ["entity", "temporal", "logical", "adversarial"]
    domain_titles = {
        "entity": "Entity Swaps",
        "temporal": "Temporal Shifts",
        "logical": "Logical Inversions",
        "adversarial": "Adversarial Traps"
    }
    
    for idx, domain in enumerate(domains):
        ax = axes[idx // 2, idx % 2]
        
        domain_data = df_pd[df_pd['domain'] == domain]
        
        sns.histplot(
            data=domain_data,
            x='reconstruction_error',
            hue='condition',
            bins=30,
            alpha=0.6,
            ax=ax,
            stat='density',
            common_norm=False
        )
        
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('Reconstruction Error (L2 Norm)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(title='Condition', labels=['Fact', 'Hallucination'])
    
    plt.tight_layout()
    
    output_path = experiment_path / "runs" / run_id / "fig1_reconstruction_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()
    
    # Figure 2: L0 Norm (Sparsity) by Domain
    print("Creating Figure 2: L0 Norm (Sparsity) Distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 2: Feature Sparsity - L0 Norm by Domain", 
                 fontsize=16, fontweight='bold')
    
    for idx, domain in enumerate(domains):
        ax = axes[idx // 2, idx % 2]
        
        domain_data = df_pd[df_pd['domain'] == domain]
        
        sns.histplot(
            data=domain_data,
            x='l0_norm',
            hue='condition',
            bins=30,
            alpha=0.6,
            ax=ax,
            stat='density',
            common_norm=False
        )
        
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('L0 Norm (Number of Active Features)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(title='Condition', labels=['Fact', 'Hallucination'])
    
    plt.tight_layout()
    
    output_path = experiment_path / "runs" / run_id / "fig2_l0_norm.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()
    
    # Figure 3: Gini Coefficient (Focus) by Domain
    print("Creating Figure 3: Gini Coefficient (Focus) Distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 3: Activation Focus - Gini Coefficient by Domain", 
                 fontsize=16, fontweight='bold')
    
    for idx, domain in enumerate(domains):
        ax = axes[idx // 2, idx % 2]
        
        domain_data = df_pd[df_pd['domain'] == domain]
        
        sns.histplot(
            data=domain_data,
            x='gini_coefficient',
            hue='condition',
            bins=30,
            alpha=0.6,
            ax=ax,
            stat='density',
            common_norm=False
        )
        
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.set_xlabel('Gini Coefficient (0=Diffuse, 1=Focused)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(title='Condition', labels=['Fact', 'Hallucination'])
    
    plt.tight_layout()
    
    output_path = experiment_path / "runs" / run_id / "fig3_gini_coefficient.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Figures saved to: {experiment_path / 'runs' / run_id}")
    print()


if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    create_spectroscopy_figures(run_id)

