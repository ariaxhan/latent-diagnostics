#!/usr/bin/env python3
"""
Hallucination Biopsy Experiment

This script demonstrates the core methodology for detecting hallucination
signatures using Sparse Autoencoder (SAE) analysis. It compares feature
activations between factual and hallucinated text to identify unique
"biomarkers" of hallucination.

Based on the research presented in the Medium article series on
hallucination detection via spectral signatures.
"""

import json
from datetime import datetime
from pathlib import Path

from hallucination_detector import (
    initialize_model_and_sae,
    extract_features,
    decode_feature,
    get_loudest_unique_features,
    run_differential_diagnosis,
)


def main():
    """Run the hallucination biopsy experiment."""
    
    print("=" * 60)
    print("HALLUCINATION BIOPSY EXPERIMENT")
    print("=" * 60)
    print()
    
    experiment_start_time = datetime.now().isoformat()
    
    # 1. Initialize instruments
    print("STEP 1: Loading Model and SAE")
    print("-" * 60)
    model, sae, device = initialize_model_and_sae()
    print()
    
    # 2. Define test case
    print("STEP 2: Defining Test Case")
    print("-" * 60)
    fact = "The Eiffel Tower is located in Paris"
    hallucination = "The Eiffel Tower is located in Rome"
    print(f"FACT:          '{fact}'")
    print(f"HALLUCINATION: '{hallucination}'")
    print()
    
    # 3. Run differential diagnosis
    print("STEP 3: Running Differential Diagnosis")
    print("-" * 60)
    diagnosis = run_differential_diagnosis(fact, hallucination, model, sae)
    
    print("Spectral Metrics:")
    print(f"  Control entropy (fact):       {diagnosis['spectral_metrics']['control_entropy']}")
    print(f"  Sample entropy (hallucination): {diagnosis['spectral_metrics']['sample_entropy']}")
    print(f"  Energy difference:            {diagnosis['spectral_metrics']['energy_diff']:.3f}")
    print()
    
    print("Biomarkers:")
    print(f"  Unique to hallucination: {diagnosis['biomarkers']['unique_to_hallucination_count']}")
    print(f"  Missing from hallucination: {diagnosis['biomarkers']['missing_grounding_count']}")
    print()
    
    # 4. Find loudest unique features
    print("STEP 4: Identifying Loudest Hallucination Signatures")
    print("-" * 60)
    loudest_features = get_loudest_unique_features(fact, hallucination, model, sae, top_k=5)
    
    if loudest_features:
        print(f"Found {len(loudest_features)} unique features (sorted by activation strength):")
        print()
        
        # 5. Decode feature meanings
        print("STEP 5: Decoding Feature Meanings")
        print("-" * 60)
        translations = []
        for i, feat_id in enumerate(loudest_features, 1):
            translation = decode_feature(feat_id, model, sae, top_k=3)
            translations.append(translation)
            words_str = ", ".join(translation['words'])
            print(f"  {i}. Feature #{feat_id:5d} → {words_str}")
        print()
    else:
        print("No unique features found.")
        print()
    
    # 6. Save results
    print("STEP 6: Saving Results")
    print("-" * 60)
    
    results = {
        "experiment_type": "hallucination_biopsy",
        "timestamp": {
            "start": experiment_start_time,
            "end": datetime.now().isoformat(),
        },
        "setup": {
            "device": device,
            "model": "gemma-2-2b",
            "sae": "gemma-scope-2b-pt-res-canonical/layer_5/width_16k/canonical",
        },
        "test_case": {
            "fact": fact,
            "hallucination": hallucination,
        },
        "diagnosis": diagnosis,
        "loudest_features": {
            "indices": loudest_features,
            "translations": translations,
        }
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / f"biopsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print()
    print("Key Findings:")
    print(f"  • {diagnosis['biomarkers']['unique_to_hallucination_count']} unique features in hallucination")
    print(f"  • Energy difference: {diagnosis['spectral_metrics']['energy_diff']:.3f}")
    if loudest_features:
        print(f"  • Top feature: #{loudest_features[0]} → {translations[0]['words']}")
    print()
    print("Next Steps:")
    print("  • Explore features on Neuronpedia: https://neuronpedia.org/gemma-2b")
    print("  • Run more test cases to validate patterns")
    print("  • See tutorials/ for detailed explanations")
    print()


if __name__ == "__main__":
    main()

