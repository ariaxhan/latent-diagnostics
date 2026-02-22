# Neural Polygraph

**Exploring prompt injection detection through causal graph geometry.**

> We observed that prompt injections create DIFFUSE causal graphs, while benign prompts have FOCUSED pathways.

⚠️ **Research prototype.** This is exploratory research with a small dataset. See [Limitations](#limitations) before drawing conclusions.

## What We Found

When processing a prompt injection, the model's internal causal structure looks different:

| Metric | Injection (n=21) | Benign (n=115) | Ratio | Cohen's d |
|--------|------------------|----------------|-------|-----------|
| Active Features | 25,862 | 12,272 | 2.1x | 1.10 |
| Concentration | 0.0025 | 0.0060 | 0.4x | 1.20 |
| Mean Influence | 0.0054 | 0.0097 | 0.6x | 1.25 |
| Graph Edges | 57M | 20M | 2.9x | 0.96 |

**What this means:** Injections activate more features with more connections, but each connection is weaker. The causal graph becomes scattered rather than focused.

**Effect size is large** (Cohen's d > 0.8), but **sample size is small** (n=21 injections). These results need validation on larger datasets.

## Visualizations

The clearest way to understand these results:

| Figure | What It Shows |
|--------|---------------|
| ![Geometry Scatter](figures/geometry_scatter.png) | Injection vs benign in feature-concentration space |
| ![Shape Comparison](figures/shape_comparison.png) | Radar chart of the "shape" difference |
| ![Distributions](figures/distributions.png) | Histogram of each metric by class |

See `figures/` for all visualizations. The [interactive notebook](notebooks/injection_geometry_explained.ipynb) walks through the analysis step by step.

## How It Works

1. **Compute attribution graph** — Use circuit-tracer to extract causal relationships between features
2. **Extract metrics** — Count active features, edges, measure concentration and mean influence
3. **Threshold** — High features + low concentration → likely injection

The hypothesis: injections contain **two competing semantic frames** (original task + injected instruction), creating interference in the causal structure.

## Quick Start

```bash
# Install
pip install -e .

# Run the analysis
python experiments/detection.py

# Generate visualizations
python experiments/analyze.py
```

To compute new attribution metrics (requires GPU):
```bash
modal run scripts/modal_pint_benchmark.py
```

## Project Structure

```
neural-polygraph/
├── src/neural_polygraph/       # Core package
├── experiments/
│   ├── detection.py            # Main experiment
│   ├── analyze.py              # Deep analysis
│   └── visualize.py            # Generate figures
├── data/results/               # Raw results (136 samples)
├── figures/                    # Key visualizations
├── notebooks/                  # Interactive explainer
├── research/                   # Thesis and notes
└── scripts/                    # GPU runners (Modal)
```

## Limitations

**This is early-stage research with significant caveats:**

1. **Small sample size**: Only 21 injection samples. Effect sizes look large but confidence intervals are wide.

2. **Class imbalance**: Dataset is 85% benign. A classifier that always predicts "benign" would score 84.6%. Our method needs to beat this baseline on balanced metrics (precision, recall, F1) not just accuracy.

3. **Single model**: Only tested on Gemma-2-2B with gemma transcoders. Unknown if this generalizes to Llama, Mistral, GPT, etc.

4. **Dataset-dependent calibration**: Thresholds are tuned on the same dataset we evaluate on. A separate calibration set would be more rigorous.

5. **Compute requirements**: ~30 seconds per sample on A100 GPU. Not practical for production without efficiency improvements.

6. **German prompts**: The PINT dataset is primarily German. English/multilingual validation needed.

## What Would Strengthen This

- [ ] Larger balanced dataset (n≥100 per class)
- [ ] Cross-model validation (Llama, Mistral)
- [ ] Separate calibration vs test sets
- [ ] Comparison to simpler baselines (perplexity, word lists)
- [ ] Proper statistical reporting (precision, recall, F1, confidence intervals)

## The Core Observation

Despite limitations, the **effect is real**:
- 95% of injections have more active features than the benign median
- 90% have lower concentration
- Cohen's d > 1.0 on multiple metrics

This suggests there's genuine signal in causal graph topology. Whether it's *useful* for detection (vs. simpler methods) remains to be proven.

## Using This For Your Own Research

The tools work on any model supported by circuit-tracer:

```python
from neural_polygraph.injection_detector import AttributionInjectionDetector, InjectionMetrics
from neural_polygraph import ExperimentStorage

# Load your own metrics
metrics = InjectionMetrics(
    n_active=...,
    n_edges=...,
    mean_influence=...,
    top_100_concentration=...,
)

# Use the detector
detector = AttributionInjectionDetector()
result = detector.classify(metrics)
```

See `scripts/modal_pint_benchmark.py` for how to compute attribution metrics on your own data.

## Citation

```bibtex
@software{neural_polygraph,
  title = {Neural Polygraph: Exploring Prompt Injection Detection via Causal Graph Geometry},
  year = {2026},
  note = {Research prototype, not peer-reviewed}
}
```

## License

MIT License
