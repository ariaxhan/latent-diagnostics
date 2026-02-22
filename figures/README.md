# Figures

Visual evidence for the causal graph geometry hypothesis.

## Key Figures

| File | Description |
|------|-------------|
| `geometry_scatter.png` | **Main result.** Scatter plot showing injection (red) vs benign (green) in n_active × concentration space. Clear geometric separation. |
| `shape_comparison.png` | Radar chart showing the "profile" difference. Injections expand outward on all axes. |
| `distributions.png` | Histograms of each metric. Shows the distributions are genuinely different, not outliers. |
| `radar_profile.png` | Another view of the shape difference with labeled axes. |
| `distributions_explained.png` | Annotated version of distributions with explanations. |
| `geometry_explained.png` | Annotated scatter plot with "normal zone" and "injection zone" labels. |
| `entropy_analogy.png` | Visual analogy: focused (normal) vs scattered (injection) influence. |
| `per_sample_scores.png` | Per-sample injection-likeness score. Red = actual injection, green = benign. |

## How These Were Generated

```bash
# From the notebook
jupyter nbconvert --execute notebooks/injection_geometry_explained.ipynb

# Or run the analysis script
python experiments/analyze.py
```

## Reading These Figures

**geometry_scatter.png:**
- X-axis: Number of active features (more = thinking about more concepts)
- Y-axis: Concentration (higher = influence focused in few connections)
- Injections cluster in lower-right (many features, low concentration)
- Benign clusters in upper-left (few features, high concentration)

**shape_comparison.png:**
- Each axis represents a metric (normalized to comparable scale)
- Larger area = more "diffuse" causal structure
- Injections consistently have larger, more diffuse shapes

## Caveats

- n=21 injections, n=115 benign
- Single model (Gemma-2-2B)
- See README.md#limitations for full discussion
