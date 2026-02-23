# Visualization

Figure generation scripts for analysis results.

## Scripts

| Script | Output | Purpose |
|--------|--------|---------|
| `generate_figures.py` | Multiple | Main figure generation pipeline |
| `domain_figures.py` | `figures/domain_analysis/` | Domain comparison visualizations |
| `residual_plots.py` | Various | Residual distribution analysis |
| `pca_plots.py` | PCA figures | Attribution metric PCA |

## Output Locations

- `figures/domain_analysis/` - Domain radar, scatter, bar charts
- `figures/` - General analysis figures

## Key Figures

| Figure | Shows |
|--------|-------|
| fig1_domain_radar.png | Radar chart of 5 domains |
| fig2_influence_concentration.png | Scatter plot, domain clustering |
| fig3_influence_gradient.png | Bar chart: focused to diffuse |
| fig4_length_control.png | Proof that influence is not length |
| fig5_effect_sizes.png | Cohen's d for all metrics |

## Usage

```bash
python experiments/visualization/domain_figures.py
python experiments/visualization/generate_figures.py
```
