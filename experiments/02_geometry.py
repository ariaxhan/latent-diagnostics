#!/usr/bin/env python3
"""
Experiment B: Geometric Topology (The "AIDA-TNG" Angle)

Goal: Treat active features as point masses in high-dimensional space and measure 
      the "shape" of the thought.

Method: Calculate the Inertia Tensor of active features weighted by activation strength.

Metrics:
1. Principal Axis Ratios (Sphericity) - Are facts triaxial and hallucinations spherical?
2. Substructure/Clumping - Do features form tight clusters or disconnected islands?

Hypothesis: Facts are TRIAXIAL (highly structured, specific direction).
            Hallucinations are SPHERICAL (isotropic confusion) or OBLATE (flat, lacking depth).
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


def run_geometric_analysis():
    """
    Run Experiment B: Geometric Topology Analysis
    
    Analyzes the geometric shape of feature activations in latent space.
    This is the core novelty: treating SAE features as a physical distribution
    and computing its inertia tensor to reveal structural properties.
    """
    
    print("=" * 80)
    print("EXPERIMENT B: GEOMETRIC TOPOLOGY")
    print("The 'AIDA-TNG' Angle - Shape Analysis of Thoughts")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "02_geometry"
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
        "sphericity": [],  # c/a ratio
        "elongation": [],  # b/a ratio
        "shape_class": [],  # Spherical, Oblate, Prolate, Triaxial
        "eigenvalue_entropy": [],
        "dimensionality": [],  # Effective dimensionality (participation ratio)
        "misalignment_angle": [],  # Angle between centroid and major axis
        "lambda_1": [],  # Largest eigenvalue
        "lambda_2": [],  # Second eigenvalue
        "lambda_3": [],  # Third eigenvalue
    }
    
    print("STEP 2: Running Geometric Analysis")
    print("-" * 80)
    
    all_samples = benchmark.get_all_samples()
    total_samples = len(all_samples)
    
    print(f"Processing {total_samples} samples across 4 domains...")
    print("Computing inertia tensors for each activation pattern...")
    print()
    
    for idx, (domain, sample) in enumerate(all_samples, 1):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total_samples} samples processed...")
        
        # Process FACT
        fact_text = sample.get_fact_text()
        fact_act = benchmark.get_activations(fact_text)
        
        # Compute geometric metrics if there are active features
        if fact_act.l0_norm > 0:
            # Get decoder weights for active features
            active_indices = fact_act.feature_indices
            decoder_weights = benchmark.sae.W_dec[active_indices]
            
            # Convert magnitudes to tensor
            import torch
            magnitudes = torch.tensor(fact_act.feature_magnitudes, device=benchmark.device)
            
            # Compute inertia tensor
            geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
            
            results["sample_id"].append(sample.id)
            results["domain"].append(domain)
            results["complexity"].append(sample.complexity)
            results["condition"].append("fact")
            results["l0_norm"].append(fact_act.l0_norm)
            results["sphericity"].append(geom.c_over_a)
            results["elongation"].append(geom.b_over_a)
            results["shape_class"].append(geom.shape_class)
            results["eigenvalue_entropy"].append(geom.eigenvalue_entropy)
            results["dimensionality"].append(geom.dimensionality)
            results["misalignment_angle"].append(geom.misalignment_angle)
            results["lambda_1"].append(float(geom.eigenvalues[0]))
            results["lambda_2"].append(float(geom.eigenvalues[1]))
            results["lambda_3"].append(float(geom.eigenvalues[2]))
        
        # Process HALLUCINATION
        hall_text = sample.get_hallucination_text()
        hall_act = benchmark.get_activations(hall_text)
        
        # Compute geometric metrics if there are active features
        if hall_act.l0_norm > 0:
            # Get decoder weights for active features
            active_indices = hall_act.feature_indices
            decoder_weights = benchmark.sae.W_dec[active_indices]
            
            # Convert magnitudes to tensor
            import torch
            magnitudes = torch.tensor(hall_act.feature_magnitudes, device=benchmark.device)
            
            # Compute inertia tensor
            geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
            
            results["sample_id"].append(sample.id)
            results["domain"].append(domain)
            results["complexity"].append(sample.complexity)
            results["condition"].append("hallucination")
            results["l0_norm"].append(hall_act.l0_norm)
            results["sphericity"].append(geom.c_over_a)
            results["elongation"].append(geom.b_over_a)
            results["shape_class"].append(geom.shape_class)
            results["eigenvalue_entropy"].append(geom.eigenvalue_entropy)
            results["dimensionality"].append(geom.dimensionality)
            results["misalignment_angle"].append(geom.misalignment_angle)
            results["lambda_1"].append(float(geom.eigenvalues[0]))
            results["lambda_2"].append(float(geom.eigenvalues[1]))
            results["lambda_3"].append(float(geom.eigenvalues[2]))
    
    print(f"✓ All {total_samples} samples processed!")
    print()
    
    # Save results
    print("STEP 3: Saving Results")
    print("-" * 80)
    
    # Save manifest
    manifest = {
        "experiment_type": "geometry",
        "experiment_name": "02_geometry",
        "description": "Geometric topology analysis using inertia tensor methodology",
        "model": "gemma-2-2b",
        "sae_layer": 5,
        "sae_width": "16k",
        "total_samples": total_samples,
        "domains": ["entity", "temporal", "logical", "adversarial"],
        "metrics": [
            "sphericity", "elongation", "shape_class", "eigenvalue_entropy",
            "dimensionality", "misalignment_angle", "lambda_1", "lambda_2", "lambda_3"
        ],
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
        print(f"  Sphericity (Fact):          {fact_df['sphericity'].mean():.4f} ± {fact_df['sphericity'].std():.4f}")
        print(f"  Sphericity (Hallucination): {hall_df['sphericity'].mean():.4f} ± {hall_df['sphericity'].std():.4f}")
        print(f"  Shape Distribution (Fact):")
        for shape in ["Spherical", "Oblate", "Prolate", "Triaxial"]:
            count = len(fact_df.filter(pl.col("shape_class") == shape))
            pct = 100 * count / len(fact_df) if len(fact_df) > 0 else 0
            print(f"    {shape:12s}: {count:3d} ({pct:5.1f}%)")
        print(f"  Shape Distribution (Hallucination):")
        for shape in ["Spherical", "Oblate", "Prolate", "Triaxial"]:
            count = len(hall_df.filter(pl.col("shape_class") == shape))
            pct = 100 * count / len(hall_df) if len(hall_df) > 0 else 0
            print(f"    {shape:12s}: {count:3d} ({pct:5.1f}%)")
        print()
    
    print("Key Findings:")
    print("  - Higher sphericity in hallucinations suggests isotropic confusion")
    print("  - Triaxial shapes in facts indicate structured, directional representations")
    print("  - Eigenvalue entropy measures energy dispersion across dimensions")
    print()
    
    print("Next Steps:")
    print("  1. Visualize results: python experiments/visualize_geometry.py")
    print("  2. Run Experiment C: python experiments/03_ghost_features.py")
    print(f"  3. Results saved to: {storage.run_path}")
    print()
    
    return storage


if __name__ == "__main__":
    storage = run_geometric_analysis()

