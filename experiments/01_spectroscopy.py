#!/usr/bin/env python3
"""
Experiment A: Pure Spectroscopy (The "Prism" Effect)

Goal: Demonstrate that hallucinations have a distinct spectral signature (entropy/sparsity).

Metrics:
1. L0 Norm (Sparsity) - Do hallucinations trigger more/fewer features?
2. Reconstruction Error - Does the SAE fail to reconstruct hallucinations (off-manifold)?
3. Gini Coefficient - Measure of "focus" in activation magnitudes

Hypothesis: Hallucinations activate "loose association" features that lack inhibition.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hallucination_detector import (
    HB_Benchmark,
    ExperimentStorage,
)


def calculate_gini_coefficient(magnitudes: list) -> float:
    """
    Calculate Gini coefficient of activation magnitudes.
    
    Gini = 0: Perfect equality (all features equally active)
    Gini = 1: Perfect inequality (one feature dominates)
    
    High Gini = Focused/specific representation
    Low Gini = Diffuse/confused representation
    """
    if len(magnitudes) == 0:
        return 0.0
    
    # Sort magnitudes
    sorted_mags = np.sort(magnitudes)
    n = len(sorted_mags)
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_mags)) / (n * np.sum(sorted_mags)) - (n + 1) / n
    
    return float(gini)


def run_spectroscopy_experiment():
    """
    Run Experiment A: Spectroscopy Analysis
    
    Analyzes spectral signatures across all 4 benchmark domains:
    - Entity Swaps
    - Temporal Shifts
    - Logical Inversions
    - Adversarial Traps
    """
    
    print("=" * 80)
    print("EXPERIMENT A: PURE SPECTROSCOPY")
    print("The 'Prism' Effect - Spectral Signatures of Hallucinations")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "01_spectroscopy"
    storage = ExperimentStorage(experiment_path)
    
    # Initialize benchmark loader
    print("STEP 1: Loading Benchmark and Model")
    print("-" * 80)
    benchmark = HB_Benchmark(data_dir="experiments/data")
    
    # Load all 4 datasets
    benchmark.load_datasets(domains=["entity", "temporal", "logical", "adversarial"])
    print()
    
    # Load model and SAE
    benchmark.load_model_and_sae(layer=5, width="16k")
    print()
    
    # Prepare results storage
    results = {
        "sample_id": [],
        "domain": [],
        "complexity": [],
        "condition": [],  # "fact" or "hallucination"
        "l0_norm": [],
        "l2_norm": [],
        "reconstruction_error": [],
        "gini_coefficient": [],
        "total_energy": [],
    }
    
    print("STEP 2: Running Spectroscopy Analysis")
    print("-" * 80)
    
    all_samples = benchmark.get_all_samples()
    total_samples = len(all_samples)
    
    print(f"Processing {total_samples} samples across 4 domains...")
    print()
    
    for idx, (domain, sample) in enumerate(all_samples, 1):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total_samples} samples processed...")
        
        # Process FACT
        fact_text = sample.get_fact_text()
        fact_act = benchmark.get_activations(fact_text)
        
        results["sample_id"].append(sample.id)
        results["domain"].append(domain)
        results["complexity"].append(sample.complexity)
        results["condition"].append("fact")
        results["l0_norm"].append(fact_act.l0_norm)
        results["l2_norm"].append(fact_act.l2_norm)
        results["reconstruction_error"].append(fact_act.reconstruction_error)
        results["gini_coefficient"].append(
            calculate_gini_coefficient(fact_act.feature_magnitudes)
        )
        results["total_energy"].append(sum(fact_act.feature_magnitudes))
        
        # Process HALLUCINATION
        hall_text = sample.get_hallucination_text()
        hall_act = benchmark.get_activations(hall_text)
        
        results["sample_id"].append(sample.id)
        results["domain"].append(domain)
        results["complexity"].append(sample.complexity)
        results["condition"].append("hallucination")
        results["l0_norm"].append(hall_act.l0_norm)
        results["l2_norm"].append(hall_act.l2_norm)
        results["reconstruction_error"].append(hall_act.reconstruction_error)
        results["gini_coefficient"].append(
            calculate_gini_coefficient(hall_act.feature_magnitudes)
        )
        results["total_energy"].append(sum(hall_act.feature_magnitudes))
    
    print(f"✓ All {total_samples} samples processed!")
    print()
    
    # Save results
    print("STEP 3: Saving Results")
    print("-" * 80)
    
    # Save manifest
    manifest = {
        "experiment_type": "spectroscopy",
        "experiment_name": "01_spectroscopy",
        "description": "Pure spectroscopy analysis of hallucination signatures",
        "model": "gemma-2-2b",
        "sae_layer": 5,
        "sae_width": "16k",
        "total_samples": total_samples,
        "domains": ["entity", "temporal", "logical", "adversarial"],
        "metrics": ["l0_norm", "l2_norm", "reconstruction_error", "gini_coefficient", "total_energy"],
        "timestamp": datetime.now().isoformat(),
    }
    storage.write_manifest(manifest)
    
    # Save metrics as Parquet
    storage.write_metrics(results)
    
    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    
    # Print summary statistics
    import polars as pl
    df = pl.DataFrame(results)
    
    print("Summary Statistics:")
    print()
    
    for domain in ["entity", "temporal", "logical", "adversarial"]:
        domain_df = df.filter(pl.col("domain") == domain)
        
        fact_df = domain_df.filter(pl.col("condition") == "fact")
        hall_df = domain_df.filter(pl.col("condition") == "hallucination")
        
        print(f"Domain: {domain.upper()}")
        print(f"  Samples: {len(fact_df)}")
        print(f"  Reconstruction Error (Fact):          {fact_df['reconstruction_error'].mean():.4f} ± {fact_df['reconstruction_error'].std():.4f}")
        print(f"  Reconstruction Error (Hallucination): {hall_df['reconstruction_error'].mean():.4f} ± {hall_df['reconstruction_error'].std():.4f}")
        print(f"  L0 Norm (Fact):                       {fact_df['l0_norm'].mean():.2f} ± {fact_df['l0_norm'].std():.2f}")
        print(f"  L0 Norm (Hallucination):              {hall_df['l0_norm'].mean():.2f} ± {hall_df['l0_norm'].std():.2f}")
        print(f"  Gini Coefficient (Fact):              {fact_df['gini_coefficient'].mean():.4f} ± {fact_df['gini_coefficient'].std():.4f}")
        print(f"  Gini Coefficient (Hallucination):     {hall_df['gini_coefficient'].mean():.4f} ± {hall_df['gini_coefficient'].std():.4f}")
        print()
    
    print("Next Steps:")
    print("  1. Visualize results: python experiments/visualize_spectroscopy.py")
    print("  2. Run Experiment B: python experiments/02_geometry.py")
    print(f"  3. Results saved to: {storage.run_path}")
    print()
    
    return storage


if __name__ == "__main__":
    storage = run_spectroscopy_experiment()

