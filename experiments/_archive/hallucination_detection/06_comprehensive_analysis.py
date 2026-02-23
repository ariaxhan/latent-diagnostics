#!/usr/bin/env python3
"""
Experiment F: Comprehensive Hallucination Analysis (MEGA EXPERIMENT)

Goal: Combine all detection methods into one efficient experiment.
      Single pass through data with multi-layer SAE analysis.

Tests Included:
1. Layer Sensitivity (from Exp 4) - Spectroscopy + Geometry across layers 5, 12, 20
2. Semantic Misalignment (from Exp 5) - Centroid drift and axis twist
3. Stability Test (NEW) - Noise robustness via Jaccard similarity
4. Entropy Test (NEW) - Shannon entropy of top-50 features
5. Cross-Layer Consistency (NEW) - Cosine similarity between layers 12 and 20

Optimization: Load all SAEs once, process each sample through all tests in sequence.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from typing import Dict, List, Tuple

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import (
    HB_Benchmark,
    ExperimentStorage,
    compute_inertia_tensor,
)


def calculate_gini_coefficient(magnitudes: list) -> float:
    """Calculate Gini coefficient of activation magnitudes."""
    if len(magnitudes) == 0:
        return 0.0
    
    sorted_mags = np.sort(magnitudes)
    n = len(sorted_mags)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_mags)) / (n * np.sum(sorted_mags)) - (n + 1) / n
    
    return float(gini)


def calculate_shannon_entropy(magnitudes: np.ndarray) -> float:
    """
    Calculate Shannon entropy of feature activation distribution.
    
    Args:
        magnitudes: Array of feature activation magnitudes
        
    Returns:
        Shannon entropy (higher = more dispersed/uncertain)
    """
    if len(magnitudes) == 0:
        return 0.0
    
    # Normalize to probability distribution
    total = np.sum(magnitudes)
    if total == 0:
        return 0.0
    
    probs = magnitudes / total
    # Filter zeros for log
    probs = probs[probs > 0]
    
    return float(-np.sum(probs * np.log(probs)))


def calculate_jaccard_similarity(indices1: List[int], indices2: List[int]) -> float:
    """
    Calculate Jaccard similarity between two sets of feature indices.
    
    Args:
        indices1: First set of feature indices
        indices2: Second set of feature indices
        
    Returns:
        Jaccard similarity (0 to 1)
    """
    set1 = set(indices1)
    set2 = set(indices2)
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle between two vectors in degrees.
    
    Args:
        v1, v2: Vectors to compare
        
    Returns:
        Angle in degrees (0-180)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_sim)
    return np.degrees(angle_rad)


def get_principal_axis_and_centroid(
    feature_indices: List[int],
    feature_magnitudes: List[float],
    sae,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the principal axis and centroid for given activations.
    
    Returns:
        centroid: Mean position of the thought (weighted by activation)
        principal_axis: Direction of maximum variance (first eigenvector)
    """
    if len(feature_indices) == 0:
        d_model = sae.cfg.d_model
        return np.zeros(d_model), np.zeros(d_model)
    
    # Get decoder weights for active features
    decoder_weights = sae.W_dec[feature_indices]
    magnitudes = torch.tensor(feature_magnitudes, device=device)
    
    # Compute inertia tensor to get principal axis
    geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
    
    # Calculate centroid (weighted mean position)
    w_norm = (magnitudes / magnitudes.sum()).detach().cpu().numpy()
    r = decoder_weights.detach().cpu().numpy()
    centroid = np.dot(w_norm, r)
    
    # Principal axis is the first eigenvector
    principal_axis = geom.principal_axes[0]
    
    return centroid, principal_axis


def analyze_sample_comprehensive(
    benchmark: HB_Benchmark,
    sample,
    domain: str,
    sae_layers: Dict[int, object],
    device: str
) -> Dict:
    """
    Run all 5 tests on a single sample.
    
    Args:
        benchmark: Benchmark instance
        sample: Sample to analyze
        domain: Domain name
        sae_layers: Dictionary mapping layer number to loaded SAE
        device: Device to use
        
    Returns:
        Dictionary with all metrics for this sample
    """
    results = {
        "sample_id": sample.id,
        "domain": domain,
        "complexity": sample.complexity,
    }
    
    # Get text for all conditions
    prompt_text = sample.prompt
    fact_text = sample.get_fact_text()
    hall_text = sample.get_hallucination_text()
    
    # =========================================================================
    # TEST 1: LAYER SENSITIVITY (Spectroscopy + Geometry across layers)
    # =========================================================================
    
    for layer in [5, 12, 20]:
        # Switch to this layer's SAE
        benchmark.sae = sae_layers[layer]
        
        # FACT analysis
        fact_act = benchmark.get_activations(fact_text)
        
        # Spectroscopy metrics
        results[f"l{layer}_fact_l0_norm"] = fact_act.l0_norm
        results[f"l{layer}_fact_reconstruction_error"] = fact_act.reconstruction_error
        results[f"l{layer}_fact_gini"] = calculate_gini_coefficient(fact_act.feature_magnitudes)
        
        # Geometry metrics
        if fact_act.l0_norm > 0:
            active_indices = fact_act.feature_indices
            decoder_weights = benchmark.sae.W_dec[active_indices]
            magnitudes = torch.tensor(fact_act.feature_magnitudes, device=device)
            geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
            
            results[f"l{layer}_fact_sphericity"] = geom.c_over_a
            results[f"l{layer}_fact_shape_class"] = geom.shape_class
            results[f"l{layer}_fact_dimensionality"] = geom.dimensionality
        else:
            results[f"l{layer}_fact_sphericity"] = 0.0
            results[f"l{layer}_fact_shape_class"] = "None"
            results[f"l{layer}_fact_dimensionality"] = 0.0
        
        # HALLUCINATION analysis
        hall_act = benchmark.get_activations(hall_text)
        
        # Spectroscopy metrics
        results[f"l{layer}_hall_l0_norm"] = hall_act.l0_norm
        results[f"l{layer}_hall_reconstruction_error"] = hall_act.reconstruction_error
        results[f"l{layer}_hall_gini"] = calculate_gini_coefficient(hall_act.feature_magnitudes)
        
        # Geometry metrics
        if hall_act.l0_norm > 0:
            active_indices = hall_act.feature_indices
            decoder_weights = benchmark.sae.W_dec[active_indices]
            magnitudes = torch.tensor(hall_act.feature_magnitudes, device=device)
            geom = compute_inertia_tensor(magnitudes, decoder_weights, top_k_components=3)
            
            results[f"l{layer}_hall_sphericity"] = geom.c_over_a
            results[f"l{layer}_hall_shape_class"] = geom.shape_class
            results[f"l{layer}_hall_dimensionality"] = geom.dimensionality
        else:
            results[f"l{layer}_hall_sphericity"] = 0.0
            results[f"l{layer}_hall_shape_class"] = "None"
            results[f"l{layer}_hall_dimensionality"] = 0.0
    
    # =========================================================================
    # TEST 2: SEMANTIC MISALIGNMENT (using layer 5)
    # =========================================================================
    
    benchmark.sae = sae_layers[5]
    
    # Get activations for all three texts
    prompt_act = benchmark.get_activations(prompt_text)
    fact_act = benchmark.get_activations(fact_text)
    hall_act = benchmark.get_activations(hall_text)
    
    # Get centroids and principal axes
    prompt_centroid, prompt_axis = get_principal_axis_and_centroid(
        prompt_act.feature_indices, prompt_act.feature_magnitudes, benchmark.sae, device
    )
    fact_centroid, fact_axis = get_principal_axis_and_centroid(
        fact_act.feature_indices, fact_act.feature_magnitudes, benchmark.sae, device
    )
    hall_centroid, hall_axis = get_principal_axis_and_centroid(
        hall_act.feature_indices, hall_act.feature_magnitudes, benchmark.sae, device
    )
    
    # Centroid drift
    results["centroid_drift_fact"] = compute_angle(prompt_centroid, fact_centroid)
    results["centroid_drift_hall"] = compute_angle(prompt_centroid, hall_centroid)
    results["centroid_drift_diff"] = results["centroid_drift_hall"] - results["centroid_drift_fact"]
    
    # Axis twist (use absolute value for bidirectional eigenvectors)
    def compute_line_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_sim = np.abs(np.dot(v1, v2)) / (norm1 * norm2)
        cos_sim = np.clip(cos_sim, 0.0, 1.0)
        angle_rad = np.arccos(cos_sim)
        return np.degrees(angle_rad)
    
    results["axis_twist_fact"] = compute_line_angle(prompt_axis, fact_axis)
    results["axis_twist_hall"] = compute_line_angle(prompt_axis, hall_axis)
    results["axis_twist_diff"] = results["axis_twist_hall"] - results["axis_twist_fact"]
    
    # =========================================================================
    # TEST 3: STABILITY (The Earthquake Test)
    # =========================================================================
    
    # Get residual activations for fact and hallucination
    tokens_fact = benchmark.model.to_tokens(fact_text)
    _, cache_fact = benchmark.model.run_with_cache(tokens_fact)
    x_fact = cache_fact["blocks.5.hook_resid_post"][0, -1, :]
    
    tokens_hall = benchmark.model.to_tokens(hall_text)
    _, cache_hall = benchmark.model.run_with_cache(tokens_hall)
    x_hall = cache_hall["blocks.5.hook_resid_post"][0, -1, :]
    
    # Add 10% noise
    x_fact_noise = x_fact + torch.randn_like(x_fact) * 0.1
    x_hall_noise = x_hall + torch.randn_like(x_hall) * 0.1
    
    # Get SAE features for clean and noisy versions
    feat_fact_clean = benchmark.sae.encode(x_fact.unsqueeze(0)).squeeze()
    feat_fact_noise = benchmark.sae.encode(x_fact_noise.unsqueeze(0)).squeeze()
    feat_hall_clean = benchmark.sae.encode(x_hall.unsqueeze(0)).squeeze()
    feat_hall_noise = benchmark.sae.encode(x_hall_noise.unsqueeze(0)).squeeze()
    
    # Get top-10 feature indices
    def get_top_k_indices(features: torch.Tensor, k: int = 10) -> List[int]:
        topk_values, topk_indices = torch.topk(features, k=min(k, len(features)))
        return topk_indices[topk_values > 0].tolist()
    
    fact_clean_top10 = get_top_k_indices(feat_fact_clean)
    fact_noise_top10 = get_top_k_indices(feat_fact_noise)
    hall_clean_top10 = get_top_k_indices(feat_hall_clean)
    hall_noise_top10 = get_top_k_indices(feat_hall_noise)
    
    # Calculate Jaccard similarity (stability score)
    results["stability_fact"] = calculate_jaccard_similarity(fact_clean_top10, fact_noise_top10)
    results["stability_hall"] = calculate_jaccard_similarity(hall_clean_top10, hall_noise_top10)
    results["stability_diff"] = results["stability_fact"] - results["stability_hall"]
    
    # =========================================================================
    # TEST 4: ENTROPY (The Fat Tail Test)
    # =========================================================================
    
    # Get top-50 feature magnitudes
    def get_top_k_magnitudes(features: torch.Tensor, k: int = 50) -> np.ndarray:
        topk_values, _ = torch.topk(features, k=min(k, len(features)))
        return topk_values[topk_values > 0].detach().cpu().numpy()
    
    fact_top50 = get_top_k_magnitudes(feat_fact_clean)
    hall_top50 = get_top_k_magnitudes(feat_hall_clean)
    
    results["entropy_fact"] = calculate_shannon_entropy(fact_top50)
    results["entropy_hall"] = calculate_shannon_entropy(hall_top50)
    results["entropy_diff"] = results["entropy_hall"] - results["entropy_fact"]
    
    # =========================================================================
    # TEST 5: CROSS-LAYER CONSISTENCY (The Schizo Test)
    # =========================================================================
    
    # Get SAE activations for layers 12 and 20
    benchmark.sae = sae_layers[12]
    tokens = benchmark.model.to_tokens(fact_text)
    _, cache = benchmark.model.run_with_cache(tokens)
    x_l12_fact = cache["blocks.12.hook_resid_post"][0, -1, :]
    feat_l12_fact = benchmark.sae.encode(x_l12_fact.unsqueeze(0)).squeeze()
    
    benchmark.sae = sae_layers[20]
    x_l20_fact = cache["blocks.20.hook_resid_post"][0, -1, :]
    feat_l20_fact = benchmark.sae.encode(x_l20_fact.unsqueeze(0)).squeeze()
    
    # Same for hallucination
    benchmark.sae = sae_layers[12]
    tokens = benchmark.model.to_tokens(hall_text)
    _, cache = benchmark.model.run_with_cache(tokens)
    x_l12_hall = cache["blocks.12.hook_resid_post"][0, -1, :]
    feat_l12_hall = benchmark.sae.encode(x_l12_hall.unsqueeze(0)).squeeze()
    
    benchmark.sae = sae_layers[20]
    x_l20_hall = cache["blocks.20.hook_resid_post"][0, -1, :]
    feat_l20_hall = benchmark.sae.encode(x_l20_hall.unsqueeze(0)).squeeze()
    
    # Compute cosine similarity between layer 12 and layer 20 features
    def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
        # Normalize to handle different dimensionalities
        v1_norm = v1 / (v1.norm() + 1e-9)
        v2_norm = v2 / (v2.norm() + 1e-9)
        # Use only the minimum dimension
        min_dim = min(len(v1_norm), len(v2_norm))
        return float(torch.dot(v1_norm[:min_dim], v2_norm[:min_dim]))
    
    results["cross_layer_consistency_fact"] = cosine_similarity(feat_l12_fact, feat_l20_fact)
    results["cross_layer_consistency_hall"] = cosine_similarity(feat_l12_hall, feat_l20_hall)
    results["cross_layer_consistency_diff"] = results["cross_layer_consistency_fact"] - results["cross_layer_consistency_hall"]
    
    return results


def run_comprehensive_analysis():
    """
    Run Experiment F: Comprehensive Hallucination Analysis
    
    Combines all detection methods into one efficient experiment:
    - Layer Sensitivity (Exp 4)
    - Semantic Misalignment (Exp 5)
    - Stability Test (NEW)
    - Entropy Test (NEW)
    - Cross-Layer Consistency (NEW)
    """
    
    print("=" * 80)
    print("EXPERIMENT F: COMPREHENSIVE HALLUCINATION ANALYSIS")
    print("The MEGA Experiment - All 5 Tests in One Pass")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "06_comprehensive_analysis"
    storage = ExperimentStorage(experiment_path)
    
    # Initialize benchmark loader
    print("STEP 1: Loading Benchmark and Model")
    print("-" * 80)
    benchmark = HB_Benchmark(data_dir="experiments/data")
    
    # Load all 4 datasets
    benchmark.load_datasets(domains=["entity", "temporal", "logical", "adversarial"])
    print()
    
    # Load model once
    print("STEP 2: Loading Model and Multiple SAEs")
    print("-" * 80)
    print("Loading Gemma-2-2b model...")
    benchmark.load_model_and_sae(layer=5, width="16k")
    device = benchmark.device
    print(f"✓ Model loaded on device: {device}")
    
    # Load SAEs for all layers
    print("Loading SAEs for layers 5, 12, 20...")
    from sae_lens import SAE
    
    sae_layers = {}
    for layer in [5, 12, 20]:
        print(f"  Loading SAE for layer {layer}...")
        sae_release = "gemma-scope-2b-pt-res-canonical"
        sae_id = f"layer_{layer}/width_16k/canonical"
        sae_layers[layer] = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
    
    print("✓ All SAEs loaded!")
    print()
    
    # Prepare results storage
    print("STEP 3: Running Comprehensive Analysis")
    print("-" * 80)
    
    all_samples = benchmark.get_all_samples()
    total_samples = len(all_samples)
    
    print(f"Processing {total_samples} samples across 4 domains...")
    print("Running all 5 tests per sample...")
    print()
    
    results_list = []
    
    for idx, (domain, sample) in enumerate(all_samples, 1):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{total_samples} samples processed...")
        
        try:
            sample_results = analyze_sample_comprehensive(
                benchmark, sample, domain, sae_layers, device
            )
            results_list.append(sample_results)
        except Exception as e:
            print(f"  Warning: Error processing sample {sample.id}: {e}")
            continue
    
    print(f"✓ All {total_samples} samples processed!")
    print()
    
    # Convert results list to columnar format
    results = {}
    if results_list:
        for key in results_list[0].keys():
            results[key] = [r[key] for r in results_list]
    
    # Save results
    print("STEP 4: Saving Results")
    print("-" * 80)
    
    # Save manifest
    manifest = {
        "experiment_type": "comprehensive_analysis",
        "experiment_name": "06_comprehensive_analysis",
        "description": "Combined analysis with all 5 hallucination detection tests",
        "model": "gemma-2-2b",
        "sae_layers": [5, 12, 20],
        "sae_width": "16k",
        "total_samples": len(results_list),
        "domains": ["entity", "temporal", "logical", "adversarial"],
        "tests": [
            "layer_sensitivity",
            "semantic_misalignment",
            "stability_earthquake",
            "entropy_fat_tail",
            "cross_layer_consistency_schizo"
        ],
        "metrics": list(results.keys()) if results else [],
        "timestamp": datetime.now().isoformat(),
    }
    storage.write_manifest(manifest)
    
    # Save metrics as Parquet
    if results:
        storage.write_metrics(results)
    
    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    
    # Print summary statistics
    if results:
        import polars as pl
        df = pl.DataFrame(results)
        
        print("Summary Statistics by Domain:")
        print()
        
        for domain in ["entity", "temporal", "logical", "adversarial"]:
            domain_df = df.filter(pl.col("domain") == domain)
            if len(domain_df) == 0:
                continue
            
            print(f"Domain: {domain.upper()}")
            print(f"  Samples: {len(domain_df)}")
            
            # Layer 5 metrics
            print(f"  Layer 5 Reconstruction Error Δ: {(domain_df['l5_hall_reconstruction_error'].mean() - domain_df['l5_fact_reconstruction_error'].mean()):.4f}")
            print(f"  Layer 5 Sphericity Δ: {(domain_df['l5_hall_sphericity'].mean() - domain_df['l5_fact_sphericity'].mean()):.4f}")
            
            # Misalignment
            print(f"  Centroid Drift Δ: {domain_df['centroid_drift_diff'].mean():.2f}°")
            print(f"  Axis Twist Δ: {domain_df['axis_twist_diff'].mean():.2f}°")
            
            # Stability
            print(f"  Stability Δ (Fact - Hall): {domain_df['stability_diff'].mean():.4f}")
            
            # Entropy
            print(f"  Entropy Δ (Hall - Fact): {domain_df['entropy_diff'].mean():.4f}")
            
            # Cross-layer
            print(f"  Cross-Layer Consistency Δ (Fact - Hall): {domain_df['cross_layer_consistency_diff'].mean():.4f}")
            
            print()
    
    print("Key Findings:")
    print("  - Combined all 5 tests in single efficient pass")
    print("  - Loaded SAEs once, processed all samples through all tests")
    print("  - Comprehensive metrics for multi-dimensional hallucination detection")
    print()
    
    print("Next Steps:")
    print("  1. Visualize results: python experiments/visualize_comprehensive.py")
    print(f"  2. Results saved to: {storage.run_path}")
    print()
    
    return storage


if __name__ == "__main__":
    storage = run_comprehensive_analysis()

