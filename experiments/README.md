# Experiments

## Key Finding

**Activation topology measures HOW a model computes, not WHETHER it's correct.**

| Detection Task | Effect Size (Length-Controlled) | Works? |
|----------------|--------------------------------|--------|
| Task type (grammar vs reasoning) | d=1.08 | Yes |
| Computational complexity | d=0.87 | Yes |
| Adversarial inputs | d~0.8 | Yes |
| Truthfulness | d=0.05 | No |

*After residualizing out text length, these are the genuine effects.*

## Directory Structure

```
experiments/
├── core/           # Main validated analysis scripts
├── statistics/     # Statistical tests and validation
├── visualization/  # Figure generation
├── utilities/      # Shared code
├── _archive/       # Historical experiments (disproved/deprecated)
└── _runs/          # Experiment outputs and data
```

## Core Analyses

| Script | Purpose | Key Result |
|--------|---------|------------|
| `core/domain_comparison.py` | Cross-domain comparison | d=1.08 grammar vs others |
| `core/truthfulness.py` | True vs false statements | No signal (d=0.05) |
| `core/cognitive_regimes.py` | Computational complexity analysis | d=0.87 |

## Statistics

| Script | Purpose |
|--------|---------|
| `statistics/bootstrap_ci.py` | Bootstrap confidence intervals |
| `statistics/shuffle_test.py` | Permutation-based validation |
| `statistics/variance_decomposition.py` | Variance attribution |

## Visualization

| Script | Purpose |
|--------|---------|
| `visualization/generate_figures.py` | Main figure generation |
| `visualization/domain_figures.py` | Domain analysis figures |
| `visualization/residual_plots.py` | Residual distribution plots |
| `visualization/pca_plots.py` | PCA attribution visualizations |

## Utilities

| Script | Purpose |
|--------|---------|
| `utilities/load_data.py` | Data loading and preprocessing |

## Data Location

Experiment data is stored in `_runs/data/results/`:

| File | Samples | Finding |
|------|---------|---------|
| `domain_attribution_metrics.json` | 210 | Task type separates |
| `truthfulness_metrics_clean.json` | 200 | No truthfulness signal |
| `pint_attribution_metrics.json` | 136 | Injection separates |

## Running

```bash
# Generate domain analysis figures
python experiments/visualization/domain_figures.py

# Run domain comparison analysis
python experiments/core/domain_comparison.py

# Run bootstrap confidence intervals
python experiments/statistics/bootstrap_ci.py
```

## Robust Metrics

**Use these:**
- `mean_influence` - causal strength between features
- `concentration` - focused vs diffuse computation
- `mean_activation` - signal strength

**Don't use (confounded by length):**
- `n_active` - r=0.98 with text length
- `n_edges` - r=0.96 with text length

## Archive

The `_archive/` directory contains:
- `hallucination_detection/` - Early experiments (d < 0.1, disproved)
- `injection_detection/` - Prompt injection work
- `deprecated/` - Superseded scripts

The breakthrough came from switching to **influence distribution** (how features affect each other) rather than **feature counts** (which just track length).
