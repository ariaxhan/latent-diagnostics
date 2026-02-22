#!/usr/bin/env python3
"""
Visualization for Experiment F: Comprehensive Analysis

✅ WORKS WITH BOTH SEQUENTIAL AND PARALLEL VERSIONS!
   - Both save to: 06_comprehensive_analysis/runs/
   - Both use identical metric names
   - Both produce same Parquet format

Creates publication-ready figures showing all 5 tests:
1. Layer Sensitivity heatmap
2. Semantic Misalignment scatter plot
3. Stability comparison
4. Entropy distribution
5. Cross-layer consistency
"""

import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neural_polygraph import ExperimentStorage


def create_comprehensive_visualization(storage: ExperimentStorage):
    """
    Create comprehensive visualization dashboard for all 5 tests.
    """
    
    # Load data
    df = storage.read_all_runs()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # PLOT 1: Layer Sensitivity - Reconstruction Error Heatmap
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    layers = [5, 12, 20]
    domains = ["entity", "temporal", "logical", "adversarial"]
    
    # Calculate deltas for each domain and layer
    heatmap_data = np.zeros((len(domains), len(layers)))
    
    for i, domain in enumerate(domains):
        domain_df = df.filter(pl.col("domain") == domain)
        for j, layer in enumerate(layers):
            fact_col = f"l{layer}_fact_reconstruction_error"
            hall_col = f"l{layer}_hall_reconstruction_error"
            delta = domain_df[hall_col].mean() - domain_df[fact_col].mean()
            heatmap_data[i, j] = delta
    
    im1 = ax1.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"L{l}" for l in layers])
    ax1.set_yticks(range(len(domains)))
    ax1.set_yticklabels([d.capitalize() for d in domains])
    ax1.set_title("Test 1: Layer Sensitivity\n(Reconstruction Error Δ)", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Domain")
    
    # Add values to heatmap
    for i in range(len(domains)):
        for j in range(len(layers)):
            text = ax1.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Δ (Hall - Fact)')
    
    # =========================================================================
    # PLOT 2: Layer Sensitivity - Sphericity Heatmap
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    heatmap_data2 = np.zeros((len(domains), len(layers)))
    
    for i, domain in enumerate(domains):
        domain_df = df.filter(pl.col("domain") == domain)
        for j, layer in enumerate(layers):
            fact_col = f"l{layer}_fact_sphericity"
            hall_col = f"l{layer}_hall_sphericity"
            delta = domain_df[hall_col].mean() - domain_df[fact_col].mean()
            heatmap_data2[i, j] = delta
    
    im2 = ax2.imshow(heatmap_data2, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"L{l}" for l in layers])
    ax2.set_yticks(range(len(domains)))
    ax2.set_yticklabels([d.capitalize() for d in domains])
    ax2.set_title("Test 1: Layer Sensitivity\n(Sphericity Δ)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Domain")
    
    # Add values
    for i in range(len(domains)):
        for j in range(len(layers)):
            text = ax2.text(j, i, f'{heatmap_data2[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Δ (Hall - Fact)')
    
    # =========================================================================
    # PLOT 3: Semantic Misalignment - Centroid Drift
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    domain_colors = {'entity': '#1f77b4', 'temporal': '#ff7f0e', 
                     'logical': '#2ca02c', 'adversarial': '#d62728'}
    
    for domain in domains:
        domain_df = df.filter(pl.col("domain") == domain)
        ax3.scatter(domain_df['centroid_drift_fact'], 
                   domain_df['centroid_drift_hall'],
                   alpha=0.6, label=domain.capitalize(),
                   color=domain_colors[domain], s=30)
    
    # Add diagonal line (y=x)
    max_val = max(df['centroid_drift_fact'].max(), df['centroid_drift_hall'].max())
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    
    ax3.set_xlabel("Fact Centroid Drift (°)")
    ax3.set_ylabel("Hallucination Centroid Drift (°)")
    ax3.set_title("Test 2: Semantic Misalignment\n(Centroid Drift)", fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 4: Semantic Misalignment - Axis Twist
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    for domain in domains:
        domain_df = df.filter(pl.col("domain") == domain)
        ax4.scatter(domain_df['axis_twist_fact'], 
                   domain_df['axis_twist_hall'],
                   alpha=0.6, label=domain.capitalize(),
                   color=domain_colors[domain], s=30)
    
    # Add diagonal line
    max_val = max(df['axis_twist_fact'].max(), df['axis_twist_hall'].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    
    ax4.set_xlabel("Fact Axis Twist (°)")
    ax4.set_ylabel("Hallucination Axis Twist (°)")
    ax4.set_title("Test 2: Semantic Misalignment\n(Axis Twist)", fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 5: Stability Test (Earthquake)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    x_pos = np.arange(len(domains))
    fact_stability = [df.filter(pl.col("domain") == d)['stability_fact'].mean() for d in domains]
    hall_stability = [df.filter(pl.col("domain") == d)['stability_hall'].mean() for d in domains]
    
    width = 0.35
    ax5.bar(x_pos - width/2, fact_stability, width, label='Fact', color='#2ca02c', alpha=0.8)
    ax5.bar(x_pos + width/2, hall_stability, width, label='Hallucination', color='#d62728', alpha=0.8)
    
    ax5.set_xlabel("Domain")
    ax5.set_ylabel("Jaccard Similarity (Stability)")
    ax5.set_title("Test 3: Stability (Earthquake)\n(Higher = More Stable)", fontsize=10, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([d.capitalize() for d in domains], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1])
    
    # =========================================================================
    # PLOT 6: Entropy Test (Fat Tail)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    fact_entropy = [df.filter(pl.col("domain") == d)['entropy_fact'].mean() for d in domains]
    hall_entropy = [df.filter(pl.col("domain") == d)['entropy_hall'].mean() for d in domains]
    
    ax6.bar(x_pos - width/2, fact_entropy, width, label='Fact', color='#2ca02c', alpha=0.8)
    ax6.bar(x_pos + width/2, hall_entropy, width, label='Hallucination', color='#d62728', alpha=0.8)
    
    ax6.set_xlabel("Domain")
    ax6.set_ylabel("Shannon Entropy")
    ax6.set_title("Test 4: Entropy (Fat Tail)\n(Higher = More Dispersed)", fontsize=10, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([d.capitalize() for d in domains], rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # PLOT 7: Cross-Layer Consistency (Schizo Test)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    fact_consistency = [df.filter(pl.col("domain") == d)['cross_layer_consistency_fact'].mean() for d in domains]
    hall_consistency = [df.filter(pl.col("domain") == d)['cross_layer_consistency_hall'].mean() for d in domains]
    
    ax7.bar(x_pos - width/2, fact_consistency, width, label='Fact', color='#2ca02c', alpha=0.8)
    ax7.bar(x_pos + width/2, hall_consistency, width, label='Hallucination', color='#d62728', alpha=0.8)
    
    ax7.set_xlabel("Domain")
    ax7.set_ylabel("Cosine Similarity (L12 vs L20)")
    ax7.set_title("Test 5: Cross-Layer Consistency (Schizo)\n(Higher = More Consistent)", fontsize=10, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([d.capitalize() for d in domains], rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # PLOT 8: Effect Sizes Summary
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 1:])
    
    # Calculate effect sizes (Cohen's d) for each test
    tests = ['Recon Error\n(L5)', 'Sphericity\n(L5)', 'Centroid\nDrift', 
             'Axis\nTwist', 'Stability', 'Entropy', 'Cross-Layer\nConsistency']
    
    effect_sizes = []
    
    for domain in domains:
        domain_df = df.filter(pl.col("domain") == domain)
        
        # Test 1: Reconstruction Error (L5)
        fact_mean = domain_df['l5_fact_reconstruction_error'].mean()
        fact_std = domain_df['l5_fact_reconstruction_error'].std()
        hall_mean = domain_df['l5_hall_reconstruction_error'].mean()
        hall_std = domain_df['l5_hall_reconstruction_error'].std()
        pooled_std = np.sqrt((fact_std**2 + hall_std**2) / 2)
        d1 = (hall_mean - fact_mean) / pooled_std if pooled_std > 0 else 0
        
        # Test 1: Sphericity (L5)
        fact_mean = domain_df['l5_fact_sphericity'].mean()
        fact_std = domain_df['l5_fact_sphericity'].std()
        hall_mean = domain_df['l5_hall_sphericity'].mean()
        hall_std = domain_df['l5_hall_sphericity'].std()
        pooled_std = np.sqrt((fact_std**2 + hall_std**2) / 2)
        d2 = (hall_mean - fact_mean) / pooled_std if pooled_std > 0 else 0
        
        # Test 2: Centroid Drift
        d3 = domain_df['centroid_drift_diff'].mean() / domain_df['centroid_drift_diff'].std() if domain_df['centroid_drift_diff'].std() > 0 else 0
        
        # Test 2: Axis Twist
        d4 = domain_df['axis_twist_diff'].mean() / domain_df['axis_twist_diff'].std() if domain_df['axis_twist_diff'].std() > 0 else 0
        
        # Test 3: Stability
        d5 = domain_df['stability_diff'].mean() / domain_df['stability_diff'].std() if domain_df['stability_diff'].std() > 0 else 0
        
        # Test 4: Entropy
        d6 = domain_df['entropy_diff'].mean() / domain_df['entropy_diff'].std() if domain_df['entropy_diff'].std() > 0 else 0
        
        # Test 5: Cross-Layer
        d7 = domain_df['cross_layer_consistency_diff'].mean() / domain_df['cross_layer_consistency_diff'].std() if domain_df['cross_layer_consistency_diff'].std() > 0 else 0
        
        effect_sizes.append([d1, d2, d3, d4, d5, d6, d7])
    
    effect_sizes = np.array(effect_sizes)
    
    # Plot grouped bar chart
    x_pos = np.arange(len(tests))
    width = 0.2
    
    for i, domain in enumerate(domains):
        offset = (i - 1.5) * width
        ax8.bar(x_pos + offset, effect_sizes[i], width, 
               label=domain.capitalize(), color=domain_colors[domain], alpha=0.8)
    
    ax8.set_xlabel("Test")
    ax8.set_ylabel("Effect Size (Cohen's d)")
    ax8.set_title("Effect Sizes Across All Tests\n(|d| > 0.5 = medium, |d| > 0.8 = large)", 
                 fontsize=10, fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(tests, fontsize=8)
    ax8.legend(fontsize=8, ncol=4)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax8.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax8.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax8.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax8.axhline(y=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Overall title
    fig.suptitle("Comprehensive Hallucination Analysis: All 5 Tests", 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    figures_dir = storage.run_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    output_path = figures_dir / "comprehensive_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive dashboard: {output_path}")
    
    plt.close()


def print_summary_table(storage: ExperimentStorage):
    """
    Print a summary table of all metrics by domain.
    """
    df = storage.read_all_runs()
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 100)
    print()
    
    domains = ["entity", "temporal", "logical", "adversarial"]
    
    for domain in domains:
        domain_df = df.filter(pl.col("domain") == domain)
        if len(domain_df) == 0:
            continue
        
        print(f"Domain: {domain.upper()}")
        print(f"  Samples: {len(domain_df)}")
        print()
        
        print("  Test 1: Layer Sensitivity")
        for layer in [5, 12, 20]:
            recon_delta = domain_df[f'l{layer}_hall_reconstruction_error'].mean() - domain_df[f'l{layer}_fact_reconstruction_error'].mean()
            spher_delta = domain_df[f'l{layer}_hall_sphericity'].mean() - domain_df[f'l{layer}_fact_sphericity'].mean()
            print(f"    Layer {layer}: Recon Δ = {recon_delta:.4f}, Sphericity Δ = {spher_delta:.4f}")
        print()
        
        print("  Test 2: Semantic Misalignment")
        print(f"    Centroid Drift Δ: {domain_df['centroid_drift_diff'].mean():.2f}° ± {domain_df['centroid_drift_diff'].std():.2f}°")
        print(f"    Axis Twist Δ: {domain_df['axis_twist_diff'].mean():.2f}° ± {domain_df['axis_twist_diff'].std():.2f}°")
        print()
        
        print("  Test 3: Stability (Earthquake)")
        print(f"    Fact Stability: {domain_df['stability_fact'].mean():.4f} ± {domain_df['stability_fact'].std():.4f}")
        print(f"    Hall Stability: {domain_df['stability_hall'].mean():.4f} ± {domain_df['stability_hall'].std():.4f}")
        print(f"    Δ (Fact - Hall): {domain_df['stability_diff'].mean():.4f}")
        print()
        
        print("  Test 4: Entropy (Fat Tail)")
        print(f"    Fact Entropy: {domain_df['entropy_fact'].mean():.4f} ± {domain_df['entropy_fact'].std():.4f}")
        print(f"    Hall Entropy: {domain_df['entropy_hall'].mean():.4f} ± {domain_df['entropy_hall'].std():.4f}")
        print(f"    Δ (Hall - Fact): {domain_df['entropy_diff'].mean():.4f}")
        print()
        
        print("  Test 5: Cross-Layer Consistency (Schizo)")
        print(f"    Fact Consistency: {domain_df['cross_layer_consistency_fact'].mean():.4f} ± {domain_df['cross_layer_consistency_fact'].std():.4f}")
        print(f"    Hall Consistency: {domain_df['cross_layer_consistency_hall'].mean():.4f} ± {domain_df['cross_layer_consistency_hall'].std():.4f}")
        print(f"    Δ (Fact - Hall): {domain_df['cross_layer_consistency_diff'].mean():.4f}")
        print()
        print("-" * 100)
        print()


def main():
    """
    Main visualization function.
    """
    print("=" * 80)
    print("VISUALIZING COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load experiment
    experiment_path = Path(__file__).parent / "06_comprehensive_analysis"
    storage = ExperimentStorage(experiment_path)
    
    # Check if data exists
    latest_run = storage.get_latest_run()
    if latest_run is None:
        print("Error: No experiment runs found!")
        print("Please run the experiment first: python experiments/06_comprehensive_analysis.py")
        return
    
    print(f"Loading data from run: {latest_run}")
    print()
    
    # Create visualizations
    print("Creating comprehensive dashboard...")
    create_comprehensive_visualization(storage)
    print()
    
    # Print summary table
    print_summary_table(storage)
    
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

