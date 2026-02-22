#!/usr/bin/env python3
"""
Visualization for Experiment C: Ghost Features

Creates Figure 3 for the paper: "Feature Prism"
- Case study showing fact vs hallucination feature activations
- Semantic interpretation of ghost features
"""

import sys
from pathlib import Path
import json
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import ExperimentStorage


def create_ghost_count_distribution(df: pl.DataFrame, output_path: Path):
    """
    Create histogram showing distribution of ghost feature counts.
    """
    df_pd = df.to_pandas()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
        
        ax.hist(
            domain_df['ghost_count'],
            bins=20,
            color='#A23B72',
            alpha=0.7,
            edgecolor='white',
            linewidth=1
        )
        
        # Add mean line
        mean_val = domain_df['ghost_count'].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        ax.set_xlabel('Number of Ghost Features', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(domain_titles[domain], fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
    
    plt.suptitle(
        'Ghost Feature Distribution Across Domains',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_feature_prism_case_study(
    df: pl.DataFrame,
    feature_decodings: dict,
    output_path: Path
):
    """
    Create Figure 3: Feature Prism case study.
    
    Shows one example with fact features on left, hallucination features on right,
    with ghost features highlighted.
    """
    df_pd = df.to_pandas()
    
    # Find an interesting example (entity domain, high ghost count)
    entity_df = df_pd[df_pd['domain'] == 'entity']
    entity_df = entity_df.sort_values('ghost_count', ascending=False)
    
    if len(entity_df) == 0:
        print("  ! No entity samples found, skipping case study")
        return
    
    example = entity_df.iloc[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Fact (no ghost features to show, just text)
    ax1.text(
        0.5, 0.9,
        "FACT",
        ha='center',
        va='top',
        fontsize=16,
        fontweight='bold',
        color='#2E86AB'
    )
    ax1.text(
        0.5, 0.8,
        f"Prompt: {example['prompt']}",
        ha='center',
        va='top',
        fontsize=12,
        wrap=True
    )
    ax1.text(
        0.5, 0.7,
        f"Completion: {example['fact']}",
        ha='center',
        va='top',
        fontsize=12,
        fontweight='bold',
        color='#2E86AB'
    )
    ax1.text(
        0.5, 0.5,
        "No unique features\n(baseline activation)",
        ha='center',
        va='top',
        fontsize=11,
        style='italic',
        color='gray'
    )
    ax1.axis('off')
    
    # Right: Hallucination with ghost features
    ax2.text(
        0.5, 0.9,
        "HALLUCINATION",
        ha='center',
        va='top',
        fontsize=16,
        fontweight='bold',
        color='#A23B72'
    )
    ax2.text(
        0.5, 0.8,
        f"Prompt: {example['prompt']}",
        ha='center',
        va='top',
        fontsize=12,
        wrap=True
    )
    ax2.text(
        0.5, 0.7,
        f"Completion: {example['hallucination']}",
        ha='center',
        va='top',
        fontsize=12,
        fontweight='bold',
        color='#A23B72'
    )
    
    # Show top ghost features
    ghost_features = example['top_ghost_features'][:5]
    ghost_mags = example['ghost_magnitudes'][:5]
    
    y_pos = 0.55
    ax2.text(
        0.5, y_pos,
        f"Ghost Features Detected: {example['ghost_count']}",
        ha='center',
        va='top',
        fontsize=11,
        fontweight='bold'
    )
    
    y_pos -= 0.08
    for feat_idx, magnitude in zip(ghost_features, ghost_mags):
        if str(feat_idx) in feature_decodings:
            decoding = feature_decodings[str(feat_idx)]
            words = ", ".join(decoding['top_words'][:5])
            
            ax2.text(
                0.1, y_pos,
                f"Feature #{feat_idx}:",
                ha='left',
                va='top',
                fontsize=10,
                fontweight='bold',
                color='#A23B72'
            )
            ax2.text(
                0.1, y_pos - 0.03,
                f"  Magnitude: {magnitude:.3f}",
                ha='left',
                va='top',
                fontsize=9
            )
            ax2.text(
                0.1, y_pos - 0.06,
                f"  Words: {words}",
                ha='left',
                va='top',
                fontsize=9,
                style='italic'
            )
            y_pos -= 0.10
    
    ax2.axis('off')
    
    plt.suptitle(
        'Figure 3: Feature Prism - Case Study\nSemantic Biomarkers of Hallucination',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_top_ghost_features_table(
    feature_decodings: dict,
    output_path: Path,
    top_k: int = 20
):
    """
    Create a text report of the most common ghost features.
    """
    # Count frequency of each feature across all samples
    # (This would require tracking in the experiment, for now just list decoded features)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOP GHOST FEATURES - SEMANTIC INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total unique ghost features decoded: {len(feature_decodings)}\n\n")
        
        for idx, (feat_id, decoding) in enumerate(list(feature_decodings.items())[:top_k], 1):
            f.write(f"{idx}. Feature #{feat_id}\n")
            f.write(f"   Top words: {', '.join(decoding['top_words'][:10])}\n")
            f.write(f"   Logits: {', '.join([f'{x:.2f}' for x in decoding['top_logits'][:5]])}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION NOTES:\n")
        f.write("=" * 80 + "\n\n")
        f.write("Ghost features represent semantic concepts that appear during hallucinations\n")
        f.write("but are absent in factual completions. These features often correspond to:\n")
        f.write("  - Conflicting semantic categories (e.g., wrong location names)\n")
        f.write("  - Tangential associations (e.g., related but incorrect concepts)\n")
        f.write("  - Weak or spurious correlations learned during training\n\n")
    
    print(f"  ✓ Saved: {output_path}")


def main():
    """Generate all visualizations for Experiment C."""
    print("=" * 80)
    print("VISUALIZING EXPERIMENT C: GHOST FEATURES")
    print("=" * 80)
    print()
    
    # Load results
    experiment_path = Path(__file__).parent / "03_ghost_features"
    
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
        print("  python experiments/03_ghost_features.py")
        return
    
    # Initialize storage with latest run
    storage = ExperimentStorage(experiment_path, run_id=latest_run)
    
    print("Loading results...")
    print(f"  Using run: {latest_run}")
    df = storage.read_metrics()
    print(f"  ✓ Loaded {len(df)} records")
    
    # Load feature decodings
    decodings_path = storage.run_path / "feature_decodings.json"
    with open(decodings_path, 'r') as f:
        feature_decodings = json.load(f)
    print(f"  ✓ Loaded {len(feature_decodings)} feature decodings")
    print()
    
    # Create output directory
    output_dir = experiment_path / "figures"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating figures...")
    
    # Figure 1: Ghost count distribution
    create_ghost_count_distribution(
        df,
        output_dir / "ghost_count_distribution.png"
    )
    
    # Figure 2: Feature prism case study (KEY FIGURE)
    create_feature_prism_case_study(
        df,
        feature_decodings,
        output_dir / "fig3_feature_prism.png"
    )
    
    # Figure 3: Top ghost features table
    create_top_ghost_features_table(
        feature_decodings,
        output_dir / "top_ghost_features.txt"
    )
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Figures saved to: {output_dir}")
    print()
    print("Key Figure for Paper:")
    print("  - fig3_feature_prism.png: Case study showing ghost features")
    print()


if __name__ == "__main__":
    main()




