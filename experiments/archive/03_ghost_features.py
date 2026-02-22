#!/usr/bin/env python3
"""
Experiment C: The "Ghost Feature" Finder

Goal: Identify features that are uniquely active in hallucinations but silent in facts.

Method:
1. Calculate the "Differential Spectrum": Act_diff = Act_hallucination - Act_fact
2. Identify the "Top-K" features with highest positive value in Act_diff
3. Decode these features to understand what semantic concepts they represent

Hypothesis: Hallucinations activate "loose association" features that represent
            conflicting or tangential semantic concepts not present in factual text.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import (
    HB_Benchmark,
    ExperimentStorage,
)
import torch


def find_ghost_features(
    fact_activations,
    hall_activations,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find features that are uniquely active in hallucination.
    
    Args:
        fact_activations: ActivationResult for factual text
        hall_activations: ActivationResult for hallucinated text
        top_k: Number of top ghost features to return
        
    Returns:
        List of (feature_index, magnitude_difference) tuples
    """
    # Create sets of active features
    fact_set = set(fact_activations.feature_indices)
    hall_set = set(hall_activations.feature_indices)
    
    # Find features unique to hallucination
    unique_to_hall = hall_set - fact_set
    
    # Get magnitudes for unique features
    hall_idx_to_mag = {
        idx: mag 
        for idx, mag in zip(hall_activations.feature_indices, hall_activations.feature_magnitudes)
    }
    
    ghost_features = [
        (idx, hall_idx_to_mag[idx])
        for idx in unique_to_hall
    ]
    
    # Sort by magnitude (loudest ghosts first)
    ghost_features.sort(key=lambda x: x[1], reverse=True)
    
    return ghost_features[:top_k]


def decode_feature_to_vocab(
    feature_idx: int,
    sae,
    model,
    top_k_words: int = 10
) -> Dict:
    """
    Decode a feature by projecting it to vocabulary space.
    
    Args:
        feature_idx: SAE feature index
        sae: SAE model
        model: Language model
        top_k_words: Number of top words to return
        
    Returns:
        Dictionary with feature interpretation
    """
    # Get feature direction in model space
    feature_direction = sae.W_dec[feature_idx]
    
    # Project to vocabulary
    logits = model.unembed(feature_direction)
    
    # Get top words
    top_token_ids = logits.argsort(descending=True)[:top_k_words]
    top_words = model.to_str_tokens(top_token_ids)
    top_logits = logits[top_token_ids].tolist()
    
    return {
        "feature_id": int(feature_idx),
        "top_words": top_words,
        "top_logits": top_logits,
    }


def run_ghost_feature_analysis():
    """
    Run Experiment C: Ghost Feature Finder
    
    Identifies and decodes features that are uniquely active during hallucinations.
    These "ghost features" represent semantic concepts that appear in lies but not truths.
    """
    
    print("=" * 80)
    print("EXPERIMENT C: GHOST FEATURE FINDER")
    print("Identifying Semantic Biomarkers of Hallucination")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "03_ghost_features"
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
        "prompt": [],
        "fact": [],
        "hallucination": [],
        "ghost_count": [],
        "top_ghost_features": [],
        "ghost_magnitudes": [],
    }
    
    # Store detailed feature decodings
    feature_decodings = {}
    
    print("STEP 2: Finding Ghost Features")
    print("-" * 80)
    
    all_samples = benchmark.get_all_samples()
    total_samples = len(all_samples)
    
    print(f"Processing {total_samples} samples across 4 domains...")
    print("Identifying features unique to hallucinations...")
    print()
    
    for idx, (domain, sample) in enumerate(all_samples, 1):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total_samples} samples processed...")
        
        # Get activations for both conditions
        fact_text = sample.get_fact_text()
        hall_text = sample.get_hallucination_text()
        
        fact_act = benchmark.get_activations(fact_text)
        hall_act = benchmark.get_activations(hall_text)
        
        # Find ghost features
        ghosts = find_ghost_features(fact_act, hall_act, top_k=10)
        
        # Store results
        results["sample_id"].append(sample.id)
        results["domain"].append(domain)
        results["complexity"].append(sample.complexity)
        results["prompt"].append(sample.prompt)
        results["fact"].append(sample.fact)
        results["hallucination"].append(sample.hallucination)
        results["ghost_count"].append(len(ghosts))
        results["top_ghost_features"].append([g[0] for g in ghosts])
        results["ghost_magnitudes"].append([g[1] for g in ghosts])
        
        # Decode top ghost features (if we haven't seen them before)
        for feature_idx, magnitude in ghosts[:5]:  # Decode top 5
            if feature_idx not in feature_decodings:
                decoding = decode_feature_to_vocab(
                    feature_idx,
                    benchmark.sae,
                    benchmark.model,
                    top_k_words=10
                )
                feature_decodings[feature_idx] = decoding
    
    print(f"✓ All {total_samples} samples processed!")
    print(f"✓ Decoded {len(feature_decodings)} unique ghost features")
    print()
    
    # Save results
    print("STEP 3: Saving Results")
    print("-" * 80)
    
    # Save manifest
    manifest = {
        "experiment_type": "ghost_features",
        "experiment_name": "03_ghost_features",
        "description": "Identification and decoding of features unique to hallucinations",
        "model": "gemma-2-2b",
        "sae_layer": 5,
        "sae_width": "16k",
        "total_samples": total_samples,
        "domains": ["entity", "temporal", "logical", "adversarial"],
        "unique_features_decoded": len(feature_decodings),
        "timestamp": datetime.now().isoformat(),
    }
    storage.write_manifest(manifest)
    
    # Save metrics as Parquet
    storage.write_metrics(results)
    
    # Save feature decodings as JSON
    decodings_path = storage.run_path / "feature_decodings.json"
    with open(decodings_path, 'w') as f:
        json.dump(feature_decodings, f, indent=2)
    print(f"  ✓ Feature decodings saved to: {decodings_path}")
    
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
        
        print(f"Domain: {domain.upper()}")
        print(f"  Samples: {len(domain_df)}")
        print(f"  Avg Ghost Features per Sample: {domain_df['ghost_count'].mean():.2f} ± {domain_df['ghost_count'].std():.2f}")
        print(f"  Max Ghost Features: {domain_df['ghost_count'].max()}")
        print(f"  Min Ghost Features: {domain_df['ghost_count'].min()}")
        print()
    
    # Print example case studies
    print("=" * 80)
    print("CASE STUDIES: Example Ghost Features")
    print("=" * 80)
    print()
    
    for domain in ["entity", "temporal", "logical", "adversarial"]:
        domain_df = df.filter(pl.col("domain") == domain)
        if len(domain_df) == 0:
            continue
        
        # Get sample with most ghost features
        max_idx = domain_df['ghost_count'].arg_max()
        sample_row = domain_df[max_idx]
        
        print(f"Domain: {domain.upper()}")
        print(f"  Prompt: {sample_row['prompt'][0]}")
        print(f"  Fact: {sample_row['fact'][0]}")
        print(f"  Hallucination: {sample_row['hallucination'][0]}")
        print(f"  Ghost Features Found: {sample_row['ghost_count'][0]}")
        print()
        
        # Show top 3 ghost features with their decodings
        top_features = sample_row['top_ghost_features'][0][:3]
        top_mags = sample_row['ghost_magnitudes'][0][:3]
        
        for feat_idx, magnitude in zip(top_features, top_mags):
            if feat_idx in feature_decodings:
                decoding = feature_decodings[feat_idx]
                words = ", ".join(decoding['top_words'][:5])
                print(f"    Feature #{feat_idx} (magnitude: {magnitude:.3f})")
                print(f"      Top words: {words}")
        print()
    
    print("Key Findings:")
    print("  - Ghost features represent semantic concepts unique to hallucinations")
    print("  - Different domains show different types of ghost features")
    print("  - Feature decodings reveal what the model 'thinks' during hallucination")
    print()
    
    print("Next Steps:")
    print("  1. Visualize results: python experiments/visualize_ghost_features.py")
    print("  2. Run Experiment D: python experiments/04_layer_sensitivity.py")
    print(f"  3. Results saved to: {storage.run_path}")
    print()
    
    return storage


if __name__ == "__main__":
    storage = run_ghost_feature_analysis()




