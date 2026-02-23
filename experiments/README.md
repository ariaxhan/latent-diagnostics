# Experiments

## Key Finding

**Activation topology measures HOW a model computes, not WHETHER it's correct.**

| Detection Task | Effect Size (Length-Controlled) | Works? |
|----------------|--------------------------------|--------|
| Task type (grammar vs reasoning) | d=1.08 | Yes |
| Computational complexity | d=0.87 | Yes |
| Adversarial inputs | d~0.8 | Yes |
| Truthfulness | d=0.05 | No |

*Note: Raw effect sizes are higher (d=3.2 for task type) but those include length confounding. After residualizing out text length, these are the genuine effects.*

## Active Experiments

| Script | Purpose | Key Result |
|--------|---------|------------|
| `domain_figures.py` | Generate domain analysis figures | 5 figures in `figures/domain_analysis/` |
| `domain_analysis.py` | Cross-domain comparison | d=1.08 grammar vs others (length-controlled) |
| `truthfulness_analysis.py` | True vs false statements | No signal (d=0.05) |
| `diagnostics.py` | Statistical analysis suite | Length confound analysis |

## Data

| File | Samples | Finding |
|------|---------|---------|
| `data/results/domain_attribution_metrics.json` | 210 | Task type separates |
| `data/results/truthfulness_metrics_clean.json` | 200 | No truthfulness signal |
| `data/results/pint_attribution_metrics.json` | 136 | Injection separates |

## Running

```bash
# Generate domain analysis figures
python experiments/domain_figures.py

# Compute new attribution metrics (parallel, crash-safe)
modal run scripts/modal_general_attribution.py \
  --input-file data/domain_analysis/domain_samples.json \
  --output-file data/results/domain_attribution_metrics.json
```

## Robust Metrics

**Use these:**
- `mean_influence` — causal strength between features
- `concentration` — focused vs diffuse computation
- `mean_activation` — signal strength

**Don't use (confounded by length):**
- `n_active` — r=0.98 with text length
- `n_edges` — r=0.96 with text length

## Figures

All in `figures/domain_analysis/`:

| Figure | Shows |
|--------|-------|
| fig1_domain_radar.png | Radar chart of 5 domains |
| fig2_influence_concentration.png | Scatter plot, domain clustering |
| fig3_influence_gradient.png | Bar chart: focused → diffuse |
| fig4_length_control.png | Proof that influence ≠ length |
| fig5_effect_sizes.png | Cohen's d for all metrics |

## Archive

The `archive/` directory contains historical experiments that informed the current approach:

- Hallucination detection (failed, d < 0.1)
- SAE spectroscopy (failed)
- Various geometry approaches (limited signal)

The breakthrough came from switching to **influence distribution** (how features affect each other) rather than **feature counts** (which just track length).
