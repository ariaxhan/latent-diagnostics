# Research

This directory documents the research methodology behind the latent diagnostics system.

## What This System Actually Does

The latent diagnostics system extracts and analyzes **SAE (Sparse Autoencoder) attribution metrics** from language model internals. It provides:

1. **Metric Extraction**: 9 graph-theoretic metrics from SAE attribution graphs
   - `n_active`: Number of active features
   - `mean_activation`: Average feature activation strength
   - `max_activation`: Peak activation
   - `total_activation`: Sum of all activations
   - `activation_entropy`: Distribution entropy
   - `top_10_concentration`: How concentrated in top features
   - `sparsity`: Fraction of zero activations
   - `mean_node_degree`: Graph connectivity
   - `median_edge_weight`: Typical edge strength

2. **Domain Profiling**: Comparing activation patterns across text types (code, prose, scientific, poetry)

3. **Visualization**: PCA projections showing how different domains cluster in metric space

## What This System Does NOT Do

- **It does not detect prompt injections** (attempted, failed due to length confounding)
- **It does not detect hallucinations** (attempted, failed - signal indistinguishable from noise)
- **It does not detect truthfulness** (tested, d=0.05 effect size - no signal)

## Key Findings

### Confirmed
- Different text domains have measurably different SAE activation profiles
- Metric extraction is reproducible and computationally tractable
- PCA on metrics shows meaningful clustering by domain

### Disproved
- "Injection detection via activation topology" - confounded by text length (r=0.96)
- "Hallucination detection via ghost features" - signal too weak
- "Physics-inspired witnesses (P, E, I)" - never implemented, likely wouldn't work

## Archived Research

See `_archived/` for speculative research directions that were not validated:
- Neural Polygraph injection detection framework
- Physics analogies (Wannier-Stark, GOE-S-Matrix)
- Vector native notation experiments

These are preserved for historical context but should not be treated as working methodology.

## Current Focus

The honest state of this research:
1. **We can extract metrics** - this works reliably
2. **We can see domain differences** - real signal exists
3. **We cannot classify** - detection claims were premature

The value is in the diagnostic toolkit, not in any classifier.
