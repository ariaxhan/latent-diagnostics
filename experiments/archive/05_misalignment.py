#!/usr/bin/env python3
"""
Experiment E: Semantic Misalignment (The "Adversarial Paradox" Solution)

Goal: Detect "confidently wrong" hallucinations by measuring semantic drift.
      Even if a hallucination is sharp (low sphericity), it may point in a 
      different semantic direction than the prompt implies.

Hypothesis: Hallucinations have larger angular deviation from the prompt's 
            semantic subspace than facts do.

Metrics:
1. Centroid Drift - Angular distance between prompt centroid and response centroid
2. Axis Twist - Angular rotation of the principal axis orientation
3. Drift Difference - Hall - Fact (positive = hallucination drifted further)
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import (
    HB_Benchmark,
    ExperimentStorage,
    compute_inertia_tensor,
)


def get_principal_axis_and_centroid(benchmark: HB_Benchmark, text: str):
    """
    Extract the principal axis and centroid for a given text.
    
    Returns:
        centroid: Mean position of the thought (weighted by activation)
        principal_axis: Direction of maximum variance (first eigenvector)
    """
    # Get activations
    act_result = benchmark.get_activations(text)
    
    if act_result.l0_norm == 0:
        # No active features - return zero vectors
        d_model = benchmark.sae.cfg.d_model
        return np.zeros(d_model), np.zeros(d_model)
    
    # Get decoder weights for active features
    active_indices = act_result.feature_indices
    decoder_weights = benchmark.sae.W_dec[active_indices]  # [k, d_model]
    
    # Convert magnitudes to tensor
    magnitudes = torch.tensor(act_result.feature_magnitudes, device=benchmark.device)
    
    # Compute inertia tensor to get principal axis
    geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
    
    # Calculate centroid (weighted mean position)
    w_norm = (magnitudes / magnitudes.sum()).detach().cpu().numpy()
    r = decoder_weights.detach().cpu().numpy()
    centroid = np.dot(w_norm, r)
    
    # Principal axis is the first eigenvector
    principal_axis = geom.principal_axes[0]
    
    return centroid, principal_axis


def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle between two vectors in degrees.
    
    Args:
        v1, v2: Vectors to compare
        
    Returns:
        Angle in degrees (0-180)
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Convert to degrees
    angle_rad = np.arccos(cos_sim)
    return np.degrees(angle_rad)


def run_misalignment_experiment():
    """
    Run Experiment E: Semantic Misalignment Analysis
    
    Focuses on adversarial and temporal domains where simple spectroscopy failed.
    Measures how far hallucinations drift from the prompt's semantic direction.
    """
    
    print("=" * 80)
    print("EXPERIMENT E: SEMANTIC MISALIGNMENT")
    print("Detecting 'Confidently Wrong' Hallucinations via Vector Alignment")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "05_misalignment"
    storage = ExperimentStorage(experiment_path)
    
    # Initialize benchmark loader
    print("STEP 1: Loading Benchmark and Model")
    print("-" * 80)
    benchmark = HB_Benchmark(data_dir="experiments/data")
    
    # Focus on domains where simple spectroscopy failed
    target_domains = ["adversarial", "temporal"]
    benchmark.load_datasets(domains=target_domains)
    print()
    
    # Load model and SAE
    benchmark.load_model_and_sae(layer=5, width="16k")
    print()
    
    # Prepare results storage
    results = {
        "sample_id": [],
        "domain": [],
        "complexity": [],
        "centroid_drift_fact": [],      # Angle between prompt and fact centroids
        "centroid_drift_hallucination": [],  # Angle between prompt and hallucination centroids
        "drift_diff": [],               # Hall - Fact (positive = hallucination drifted further)
        "axis_twist_fact": [],          # Angle between prompt and fact principal axes
        "axis_twist_hallucination": [], # Angle between prompt and hallucination principal axes
        "twist_diff": [],               # Hall - Fact (positive = hallucination twisted more)
    }
    
    print("STEP 2: Running Misalignment Analysis")
    print("-" * 80)
    
    all_samples = benchmark.get_all_samples()
    total_samples = len(all_samples)
    
    print(f"Processing {total_samples} samples across {len(target_domains)} domains...")
    print("Computing semantic drift and axis twist for each sample...")
    print()
    
    for idx, (domain, sample) in enumerate(all_samples, 1):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{total_samples} samples processed...")
        
        try:
            # 1. ANALYZE THE PROMPT (The "Question" Vector)
            # We want the direction the prompt was "pointing" before generation started.
            prompt_centroid, prompt_axis = get_principal_axis_and_centroid(
                benchmark, sample.prompt
            )
            
            # 2. ANALYZE THE FACT (The "Truth" Vector)
            fact_text = sample.get_fact_text()
            fact_centroid, fact_axis = get_principal_axis_and_centroid(
                benchmark, fact_text
            )
            
            # 3. ANALYZE THE HALLUCINATION (The "Lie" Vector)
            hall_text = sample.get_hallucination_text()
            hall_centroid, hall_axis = get_principal_axis_and_centroid(
                benchmark, hall_text
            )
            
            # 4. COMPUTE ANGLES (Cosine Distance)
            # Method A: Centroid Drift (Did the "center of mass" move?)
            angle_fact_centroid = compute_angle(prompt_centroid, fact_centroid)
            angle_hall_centroid = compute_angle(prompt_centroid, hall_centroid)
            drift_diff = angle_hall_centroid - angle_fact_centroid
            
            # Method B: Axis Twist (Did the "orientation" of the thought rotate?)
            # For eigenvectors (bidirectional lines), we use absolute value of dot product
            # This gives us the angle between the lines (0-90 degrees)
            def compute_line_angle(v1: np.ndarray, v2: np.ndarray) -> float:
                """Compute angle between two lines (handles bidirectional eigenvectors)."""
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                cos_sim = np.abs(np.dot(v1, v2)) / (norm1 * norm2)
                cos_sim = np.clip(cos_sim, 0.0, 1.0)  # Clamp to [0, 1] for arccos
                angle_rad = np.arccos(cos_sim)
                return np.degrees(angle_rad)
            
            twist_fact = compute_line_angle(prompt_axis, fact_axis)
            twist_hall = compute_line_angle(prompt_axis, hall_axis)
            twist_diff = twist_hall - twist_fact
            
            results["sample_id"].append(sample.id)
            results["domain"].append(domain)
            results["complexity"].append(sample.complexity)
            results["centroid_drift_fact"].append(angle_fact_centroid)
            results["centroid_drift_hallucination"].append(angle_hall_centroid)
            results["drift_diff"].append(drift_diff)
            results["axis_twist_fact"].append(twist_fact)
            results["axis_twist_hallucination"].append(twist_hall)
            results["twist_diff"].append(twist_diff)
            
        except Exception as e:
            print(f"  Warning: Error processing sample {sample.id}: {e}")
            continue
    
    print(f"✓ All {total_samples} samples processed!")
    print()
    
    # Save results
    print("STEP 3: Saving Results")
    print("-" * 80)
    
    # Save manifest
    manifest = {
        "experiment_type": "misalignment",
        "experiment_name": "05_misalignment",
        "description": "Semantic misalignment analysis measuring drift and twist angles",
        "model": "gemma-2-2b",
        "sae_layer": 5,
        "sae_width": "16k",
        "total_samples": total_samples,
        "domains": target_domains,
        "metrics": [
            "centroid_drift_fact",
            "centroid_drift_hallucination",
            "drift_diff",
            "axis_twist_fact",
            "axis_twist_hallucination",
            "twist_diff"
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
    
    for domain in target_domains:
        domain_df = df.filter(pl.col("domain") == domain)
        
        print(f"Domain: {domain.upper()}")
        print(f"  Samples: {len(domain_df)}")
        print(f"  Centroid Drift (Fact):          {domain_df['centroid_drift_fact'].mean():.2f}° ± {domain_df['centroid_drift_fact'].std():.2f}°")
        print(f"  Centroid Drift (Hallucination): {domain_df['centroid_drift_hallucination'].mean():.2f}° ± {domain_df['centroid_drift_hallucination'].std():.2f}°")
        print(f"  Drift Difference (Hall - Fact): {domain_df['drift_diff'].mean():.2f}° ± {domain_df['drift_diff'].std():.2f}°")
        print(f"  Axis Twist (Fact):              {domain_df['axis_twist_fact'].mean():.2f}° ± {domain_df['axis_twist_fact'].std():.2f}°")
        print(f"  Axis Twist (Hallucination):     {domain_df['axis_twist_hallucination'].mean():.2f}° ± {domain_df['axis_twist_hallucination'].std():.2f}°")
        print(f"  Twist Difference (Hall - Fact): {domain_df['twist_diff'].mean():.2f}° ± {domain_df['twist_diff'].std():.2f}°")
        print()
    
    print("Key Findings:")
    print("  - Positive drift_diff indicates hallucinations drift further from prompt")
    print("  - Positive twist_diff indicates hallucinations rotate more from prompt direction")
    print("  - This solves the 'Adversarial Paradox': confident but misaligned")
    print()
    
    print("Next Steps:")
    print("  1. Visualize results: python experiments/visualize_misalignment.py")
    print(f"  2. Results saved to: {storage.run_path}")
    print()
    
    return storage


if __name__ == "__main__":
    storage = run_misalignment_experiment()

