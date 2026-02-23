# Latent Diagnostics

**Representation-level analysis of internal activation structure in large language models.**

> Activation topology measures *how* a model computes, not *whether* it's correct.

## Core Thesis

We introduce a framework that analyzes **internal activation topology** to characterize what kind of computation an LLM is performing. By measuring causal influence distribution across features, we can distinguish between different computational regimes.

**What this detects:** The *type* and *complexity* of thinking.
**What this doesn't detect:** Whether the output is correct.

## Key Results (Length-Controlled)

All effect sizes are **controlled for text length** via residualization.

| Detection Task | Effect Size (d) | Status |
|----------------|-----------------|--------|
| Task type (grammar vs reasoning) | **1.08** | Works |
| Computational complexity | **0.87** | Works |
| Adversarial inputs | ~0.8 | Works |
| Truthfulness | 0.05 | Doesn't work |

**The pivot experiment:** After regressing out text length, influence (d=1.08) and concentration (d=0.87) still show large effects. N_active collapses to d=0.07. This is **genuine regime difference**, not length-driven scaling.

## What We Measure

| Metric | What It Captures | Length-Controlled d |
|--------|------------------|---------------------|
| `mean_influence` | Causal strength between features | **1.08** (genuine) |
| `concentration` | Focused vs diffuse computation | **0.87** (genuine) |
| `mean_activation` | Signal strength | 0.64 (medium) |
| `n_active` | Feature count | 0.07 (collapses) |

**Use:** influence, concentration
**Don't use:** n_active (confounded by length, r=0.98)

## Use Cases

**Works for:**
- Input classification (what type of task is this?)
- Anomaly detection (is this input unusual?)
- Complexity estimation (how hard is the model working?)

**Doesn't work for:**
- Hallucination detection
- Fact-checking
- Output quality assessment

## Quick Start

```bash
pip install -e .

# Generate figures (all length-controlled)
python experiments/generate_all_figures.py

# Compute attribution metrics (parallel, crash-safe)
modal run scripts/modal_general_attribution.py \
  --input-file data/domain_analysis/domain_samples.json \
  --output-file data/results/domain_attribution_metrics.json
```

## Figures

All in `figures/paper/` — every figure uses **length-controlled metrics**:

| Figure | Shows |
|--------|-------|
| **central_summary.png** | Main figure (8 panels) |
| length_control_comparison.png | Before/after length control |
| detection_summary.png | What it can/can't detect |
| pca_clustering.png | Domains cluster after length control |
| boxplots_significance.png | Per-domain distributions |

## Limitations

1. **Requires model internals** — Only works on models with SAE/transcoder access
2. **Compute intensive** — ~30 sec/sample on A100
3. **Measures structure, not correctness** — Can't detect hallucinations
4. **Must control for length** — Raw n_active is confounded (r=0.98)

## Citation

```bibtex
@software{latent_diagnostics,
  title = {Latent Diagnostics: Representation-Level Analysis of Internal Activation Structure in Large Language Models},
  year = {2026}
}
```

## License

MIT
