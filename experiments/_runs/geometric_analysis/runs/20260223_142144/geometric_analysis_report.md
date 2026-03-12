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
| grammar | 50 | Prolate | 0.543 | 0.384 | 2.14 | 1.056 | 27.0° |
| commonsense | 50 | Prolate | 0.592 | 0.513 | 2.39 | 1.103 | 66.1° |
| Roof shingle removal | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Clean and jerk | 2 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Canoeing | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| High jump | 3 | Triaxial | 0.635 | 0.000 | 1.69 | 0.600 | 83.7° |
| Playing harmonica | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Sumo | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Sharpening knives | 7 | Prolate | 0.256 | 0.197 | 1.22 | 0.405 | 65.0° |
| Playing bagpipes | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Playing water polo | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Having an ice cream | 5 | Triaxial | 0.668 | 0.356 | 2.07 | 0.890 | 87.0° |
| Cheerleading | 3 | Prolate | 0.098 | 0.000 | 1.02 | 0.053 | 68.2° |
| Cutting the grass | 3 | Prolate | 0.435 | 0.000 | 1.36 | 0.438 | 61.8° |
| Washing face | 5 | Oblate | 0.914 | 0.461 | 2.59 | 1.077 | 69.3° |
| Paintball | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Gargling mouthwash | 4 | Prolate | 0.598 | 0.369 | 1.95 | 0.829 | 52.4° |
| Archery | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Surfing | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Plastering | 1 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Beer pong | 2 | Spherical | 0.000 | 0.000 | 0.00 | 0.000 | 0.0° |
| Grooming dog | 3 | Prolate | 0.136 | 0.000 | 1.04 | 0.090 | 63.7° |
| nli | 50 | Prolate | 0.516 | 0.189 | 1.67 | 0.760 | 42.4° |
| paraphrase | 13 | Prolate | 0.506 | 0.464 | 2.11 | 1.013 | 84.1° |

## Shape Interpretation

- **Spherical**: Uniform variance across metrics (isotropic)
- **Oblate**: Disk-like, variance concentrated in 2 dimensions
- **Prolate**: Cigar-like, variance along one dominant axis
- **Triaxial**: Irregular, mixed variance distribution

## Key Findings

- **Most spherical domain**: Washing face (uniform variance)
- **Most elongated domain**: Beer pong (concentrated variance)
- **Highest effective dimensionality**: Washing face (2.59)
- **Lowest effective dimensionality**: Beer pong (0.00)

## Cross-Domain Distances

Mean centroid distance: 1.320 (standardized units)
