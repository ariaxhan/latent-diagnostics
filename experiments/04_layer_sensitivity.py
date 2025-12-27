#!/usr/bin/env python3
"""
Experiment D: Layer Sensitivity Analysis

Goal: Show *where* in the model the thought becomes "untrue" by analyzing
      detection accuracy across different layers.

Method:
1. Run spectroscopy and geometry analysis across multiple layers (5, 12, 20)
2. Compare detection accuracy using both methods at each layer
3. Identify which layer shows the strongest hallucination signature

Hypothesis: Mid-to-late layers (12-20) will show stronger hallucination signatures
            as semantic processing deepens and the model commits to its output.
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
    compute_inertia_tensor,
)
import torch


def calculate_gini_coefficient(magnitudes: list) -> float:
    """Calculate Gini coefficient of activation magnitudes."""
    if len(magnitudes) == 0:
        return 0.0
    
    sorted_mags = np.sort(magnitudes)
    n = len(sorted_mags)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_mags)) / (n * np.sum(sorted_mags)) - (n + 1) / n
    
    return float(gini)


def analyze_layer(
    benchmark: HB_Benchmark,
    layer: int,
    width: str = "16k",
    sample_limit: int = None
):
    """
    Analyze hallucination signatures at a specific layer.
    
    Args:
        benchmark: Benchmark instance with loaded data
        layer: Layer number to analyze
        width: SAE width
        sample_limit: Limit number of samples (for faster testing)
        
    Returns:
        Dictionary with metrics for this layer
    """
    print(f"  Analyzing Layer {layer}...")
    
    # Load SAE for this layer
    benchmark.load_model_and_sae(layer=layer, width=width)
    
    results = {
        "sample_id": [],
        "domain": [],
        "condition": [],
        "layer": [],
        # Spectroscopy metrics
        "l0_norm": [],
        "reconstruction_error": [],
        "gini_coefficient": [],
        # Geometry metrics
        "sphericity": [],
        "shape_class": [],
        "dimensionality": [],
    }
    
    all_samples = benchmark.get_all_samples()
    if sample_limit:
        all_samples = all_samples[:sample_limit]
    
    for domain, sample in all_samples:
        # Process FACT
        fact_text = sample.get_fact_text()
        fact_act = benchmark.get_activations(fact_text)
        
        # Spectroscopy metrics
        results["sample_id"].append(sample.id)
        results["domain"].append(domain)
        results["condition"].append("fact")
        results["layer"].append(layer)
        results["l0_norm"].append(fact_act.l0_norm)
        results["reconstruction_error"].append(fact_act.reconstruction_error)
        results["gini_coefficient"].append(
            calculate_gini_coefficient(fact_act.feature_magnitudes)
        )
        
        # Geometry metrics
        if fact_act.l0_norm > 0:
            active_indices = fact_act.feature_indices
            decoder_weights = benchmark.sae.W_dec[active_indices]
            magnitudes = torch.tensor(fact_act.feature_magnitudes, device=benchmark.device)
            geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
            
            results["sphericity"].append(geom.c_over_a)
            results["shape_class"].append(geom.shape_class)
            results["dimensionality"].append(geom.dimensionality)
        else:
            results["sphericity"].append(0.0)
            results["shape_class"].append("None")
            results["dimensionality"].append(0.0)
        
        # Process HALLUCINATION
        hall_text = sample.get_hallucination_text()
        hall_act = benchmark.get_activations(hall_text)
        
        # Spectroscopy metrics
        results["sample_id"].append(sample.id)
        results["domain"].append(domain)
        results["condition"].append("hallucination")
        results["layer"].append(layer)
        results["l0_norm"].append(hall_act.l0_norm)
        results["reconstruction_error"].append(hall_act.reconstruction_error)
        results["gini_coefficient"].append(
            calculate_gini_coefficient(hall_act.feature_magnitudes)
        )
        
        # Geometry metrics
        if hall_act.l0_norm > 0:
            active_indices = hall_act.feature_indices
            decoder_weights = benchmark.sae.W_dec[active_indices]
            magnitudes = torch.tensor(hall_act.feature_magnitudes, device=benchmark.device)
            geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
            
            results["sphericity"].append(geom.c_over_a)
            results["shape_class"].append(geom.shape_class)
            results["dimensionality"].append(geom.dimensionality)
        else:
            results["sphericity"].append(0.0)
            results["shape_class"].append("None")
            results["dimensionality"].append(0.0)
    
    print(f"    ✓ Layer {layer} complete ({len(all_samples)} samples)")
    
    return results


def run_layer_sensitivity_analysis():
    """
    Run Experiment D: Layer Sensitivity Analysis
    
    Analyzes hallucination detection across multiple layers to identify
    where in the network the hallucination signature emerges most strongly.
    """
    
    print("=" * 80)
    print("EXPERIMENT D: LAYER SENSITIVITY ANALYSIS")
    print("Where in the Model Does the Thought Become Untrue?")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "04_layer_sensitivity"
    storage = ExperimentStorage(experiment_path)
    
    # Initialize benchmark loader
    print("STEP 1: Loading Benchmark")
    print("-" * 80)
    benchmark = HB_Benchmark(data_dir="experiments/data")
    
    # Load all 4 datasets
    benchmark.load_datasets(domains=["entity", "temporal", "logical", "adversarial"])
    print()
    
    # Define layers to analyze
    layers = [5, 12, 20]
    print(f"STEP 2: Analyzing Layers {layers}")
    print("-" * 80)
    print(f"Will analyze {len(layers)} layers across all domains")
    print()
    
    # Collect results across all layers
    all_results = {
        "sample_id": [],
        "domain": [],
        "condition": [],
        "layer": [],
        "l0_norm": [],
        "reconstruction_error": [],
        "gini_coefficient": [],
        "sphericity": [],
        "shape_class": [],
        "dimensionality": [],
    }
    
    for layer in layers:
        layer_results = analyze_layer(benchmark, layer, width="16k")
        
        # Merge into all_results
        for key in all_results.keys():
            all_results[key].extend(layer_results[key])
    
    print()
    print("✓ All layers analyzed!")
    print()
    
    # Save results
    print("STEP 3: Saving Results")
    print("-" * 80)
    
    # Save manifest
    manifest = {
        "experiment_type": "layer_sensitivity",
        "experiment_name": "04_layer_sensitivity",
        "description": "Analysis of hallucination signatures across model layers",
        "model": "gemma-2-2b",
        "layers_analyzed": layers,
        "sae_width": "16k",
        "domains": ["entity", "temporal", "logical", "adversarial"],
        "metrics": [
            "l0_norm", "reconstruction_error", "gini_coefficient",
            "sphericity", "shape_class", "dimensionality"
        ],
        "timestamp": datetime.now().isoformat(),
    }
    storage.write_manifest(manifest)
    
    # Save metrics as Parquet
    storage.write_metrics(all_results)
    
    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    
    # Print summary statistics
    import polars as pl
    df = pl.DataFrame(all_results)
    
    print("Summary Statistics by Layer:")
    print()
    
    for layer in layers:
        layer_df = df.filter(pl.col("layer") == layer)
        fact_df = layer_df.filter(pl.col("condition") == "fact")
        hall_df = layer_df.filter(pl.col("condition") == "hallucination")
        
        print(f"Layer {layer}:")
        print(f"  Samples: {len(fact_df)}")
        
        # Spectroscopy metrics
        print(f"  Reconstruction Error:")
        print(f"    Fact:          {fact_df['reconstruction_error'].mean():.4f} ± {fact_df['reconstruction_error'].std():.4f}")
        print(f"    Hallucination: {hall_df['reconstruction_error'].mean():.4f} ± {hall_df['reconstruction_error'].std():.4f}")
        print(f"    Δ (Hall - Fact): {hall_df['reconstruction_error'].mean() - fact_df['reconstruction_error'].mean():.4f}")
        
        # Geometry metrics
        print(f"  Sphericity:")
        print(f"    Fact:          {fact_df['sphericity'].mean():.4f} ± {fact_df['sphericity'].std():.4f}")
        print(f"    Hallucination: {hall_df['sphericity'].mean():.4f} ± {hall_df['sphericity'].std():.4f}")
        print(f"    Δ (Hall - Fact): {hall_df['sphericity'].mean() - fact_df['sphericity'].mean():.4f}")
        
        # Shape distribution
        print(f"  Shape Distribution (Hallucination):")
        for shape in ["Spherical", "Oblate", "Prolate", "Triaxial"]:
            count = len(hall_df.filter(pl.col("shape_class") == shape))
            pct = 100 * count / len(hall_df) if len(hall_df) > 0 else 0
            print(f"    {shape:12s}: {pct:5.1f}%")
        
        print()
    
    # Calculate effect sizes
    print("=" * 80)
    print("EFFECT SIZES (Cohen's d)")
    print("=" * 80)
    print()
    
    for layer in layers:
        layer_df = df.filter(pl.col("layer") == layer)
        fact_df = layer_df.filter(pl.col("condition") == "fact")
        hall_df = layer_df.filter(pl.col("condition") == "hallucination")
        
        # Cohen's d for reconstruction error
        mean_diff = hall_df['reconstruction_error'].mean() - fact_df['reconstruction_error'].mean()
        pooled_std = np.sqrt(
            (fact_df['reconstruction_error'].std()**2 + hall_df['reconstruction_error'].std()**2) / 2
        )
        cohens_d_recon = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Cohen's d for sphericity
        mean_diff = hall_df['sphericity'].mean() - fact_df['sphericity'].mean()
        pooled_std = np.sqrt(
            (fact_df['sphericity'].std()**2 + hall_df['sphericity'].std()**2) / 2
        )
        cohens_d_spher = mean_diff / pooled_std if pooled_std > 0 else 0
        
        print(f"Layer {layer}:")
        print(f"  Reconstruction Error: d = {cohens_d_recon:.3f}")
        print(f"  Sphericity:           d = {cohens_d_spher:.3f}")
        print()
    
    print("Key Findings:")
    print("  - Larger effect sizes indicate stronger hallucination signatures")
    print("  - Layer with highest effect size is optimal for detection")
    print("  - Both spectroscopy and geometry provide complementary signals")
    print()
    
    print("Next Steps:")
    print("  1. Visualize results: python experiments/visualize_layer_sensitivity.py")
    print("  2. Generate paper figures from all experiments")
    print(f"  3. Results saved to: {storage.run_path}")
    print()
    
    return storage


if __name__ == "__main__":
    storage = run_layer_sensitivity_analysis()

