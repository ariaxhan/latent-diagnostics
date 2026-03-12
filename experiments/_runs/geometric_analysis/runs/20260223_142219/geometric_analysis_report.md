# Geometric Analysis of Attribution Metric Space

## Methodology

Adapted from AIDA-TNG galaxy morphology analysis:
- Each sample is a point in 6D metric space
- Domain clusters are analyzed as point clouds
- Covariance matrix eigendecomposition reveals shape

**Metrics used:** mean_influence, max_influence, top_100_concentration, mean_activation, logit_entropy, max_logit_prob

## Domain Geometries

| Domain | n | Shape | b/a | c/a | Eff. Dim | Entropy | Misalign |
|--------|---|-------|-----|-----|----------|---------|----------|
| grammar | 50 | Prolate | 0.567 | 0.385 | 2.19 | 1.070 | 33.3° |
| commonsense | 50 | Prolate | 0.562 | 0.482 | 2.26 | 1.062 | 67.1° |
| nli | 50 | Prolate | 0.549 | 0.190 | 1.73 | 0.781 | 54.2° |
| paraphrase | 13 | Prolate | 0.517 | 0.482 | 2.16 | 1.028 | 82.1° |

## Shape Interpretation

- **Spherical**: Uniform variance across metrics (isotropic)
- **Oblate**: Disk-like, variance concentrated in 2 dimensions
- **Prolate**: Cigar-like, variance along one dominant axis
- **Triaxial**: Irregular, mixed variance distribution

## Key Findings

- **Most spherical domain**: commonsense (uniform variance)
- **Most elongated domain**: nli (concentrated variance)
- **Highest effective dimensionality**: commonsense (2.26)
- **Lowest effective dimensionality**: nli (1.73)

## Cross-Domain Distances

Mean centroid distance: 3.041 (standardized units)
