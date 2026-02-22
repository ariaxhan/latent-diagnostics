# Experiment 6: Comprehensive Hallucination Analysis

## Overview

The **MEGA EXPERIMENT** - combines all hallucination detection methods into one efficient, optimized experiment. This experiment processes each sample through all 5 tests in a single pass, loading SAEs once and reusing them across all analyses.

## Tests Included

### Test 1: Layer Sensitivity (from Experiment 4)
- **Goal**: Identify where in the model hallucinations emerge
- **Layers**: 5, 12, 20
- **Metrics**:
  - Spectroscopy: L0 norm, reconstruction error, Gini coefficient
  - Geometry: Sphericity, shape class, dimensionality
- **Hypothesis**: Mid-to-late layers show stronger hallucination signatures

### Test 2: Semantic Misalignment (from Experiment 5)
- **Goal**: Detect "confidently wrong" hallucinations via semantic drift
- **Metrics**:
  - Centroid Drift: Angular distance between prompt and response centroids
  - Axis Twist: Angular rotation of principal axis orientation
- **Hypothesis**: Hallucinations drift further from prompt's semantic direction

### Test 3: Stability Test (NEW - "The Earthquake")
- **Goal**: Measure robustness to noise
- **Method**: Add 10% Gaussian noise to residual activations
- **Metric**: Jaccard similarity of top-10 features (clean vs noisy)
- **Hypothesis**: Facts are more stable; hallucinations are brittle

### Test 4: Entropy Test (NEW - "The Fat Tail")
- **Goal**: Measure activation distribution dispersion
- **Method**: Shannon entropy of top-50 feature magnitudes
- **Metric**: Entropy (higher = more dispersed/uncertain)
- **Hypothesis**: Hallucinations have higher entropy (less focused)

### Test 5: Cross-Layer Consistency (NEW - "The Schizo Test")
- **Goal**: Measure internal coherence across layers
- **Method**: Cosine similarity between Layer 12 and Layer 20 features
- **Metric**: Cosine similarity (higher = more consistent)
- **Hypothesis**: Facts maintain consistency; hallucinations diverge

## Optimization Strategy

**Key Innovation**: Single-pass processing with pre-loaded SAEs

1. **Load Once**: All 3 SAEs (layers 5, 12, 20) loaded at start
2. **Process Sequentially**: Each sample goes through all 5 tests
3. **Reuse Activations**: Cache model activations when possible
4. **Efficient Memory**: Clean up after each sample

**Performance Gain**: ~3-5x faster than running experiments 4 and 5 separately, plus 3 new tests!

## Running the Experiment

```bash
# Run the comprehensive analysis
python experiments/06_comprehensive_analysis.py

# Visualize results
python experiments/visualize_comprehensive.py
```

## Output Structure

```
06_comprehensive_analysis/
├── README.md
└── runs/
    └── YYYYMMDD_HHMMSS/
        ├── manifest.json       # Experiment metadata
        ├── metrics.parquet     # All metrics for all samples
        └── figures/
            └── comprehensive_dashboard.png  # 8-panel visualization
```

## Metrics Collected

For each sample, the following metrics are computed:

**Layer Sensitivity (per layer: 5, 12, 20)**:
- `l{layer}_fact_l0_norm`
- `l{layer}_fact_reconstruction_error`
- `l{layer}_fact_gini`
- `l{layer}_fact_sphericity`
- `l{layer}_fact_shape_class`
- `l{layer}_fact_dimensionality`
- (Same for `hall_*`)

**Semantic Misalignment**:
- `centroid_drift_fact`, `centroid_drift_hall`, `centroid_drift_diff`
- `axis_twist_fact`, `axis_twist_hall`, `axis_twist_diff`

**Stability**:
- `stability_fact`, `stability_hall`, `stability_diff`

**Entropy**:
- `entropy_fact`, `entropy_hall`, `entropy_diff`

**Cross-Layer Consistency**:
- `cross_layer_consistency_fact`, `cross_layer_consistency_hall`, `cross_layer_consistency_diff`

## Interpretation Guide

### Good Hallucination Detectors (Large Effect Sizes)

- **Reconstruction Error Δ > 0**: Hallucinations harder to reconstruct
- **Sphericity Δ < 0**: Hallucinations more elongated (less spherical)
- **Centroid Drift Δ > 0**: Hallucinations drift further from prompt
- **Axis Twist Δ > 0**: Hallucinations rotate more from prompt direction
- **Stability Δ > 0**: Facts more stable, hallucinations more brittle
- **Entropy Δ > 0**: Hallucinations more dispersed/uncertain
- **Cross-Layer Consistency Δ > 0**: Facts more consistent across layers

### Effect Size Thresholds (Cohen's d)

- **Small**: |d| = 0.2
- **Medium**: |d| = 0.5
- **Large**: |d| = 0.8

## Expected Results

Based on experiments 1-5, we expect:

1. **Entity Domain**: Strong signals in all tests (easy to detect)
2. **Temporal Domain**: Moderate signals, best detected by misalignment + entropy
3. **Logical Domain**: Strong signals in geometry + cross-layer consistency
4. **Adversarial Domain**: Challenging, requires combination of all tests

## Next Steps

After running this experiment:

1. Analyze which tests work best for each domain
2. Build ensemble detector combining multiple signals
3. Determine optimal layer for each test
4. Create ROC curves for classification performance
5. Test on new domains/models

## Citation

If you use this experiment in your research, please cite:

```bibtex
@misc{neural-polygraph-2025,
  title={Neural Polygraph: Comprehensive Hallucination Detection via Multi-Modal SAE Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/neural-polygraph}
}
```

