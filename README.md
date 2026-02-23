# Latent Diagnostics

**Representation-level analysis of internal activation structure in large language models.**

> Activation topology measures *how* a model computes, not *whether* it's correct.

## Core Thesis

We introduce a framework that analyzes **internal activation topology** to characterize what kind of computation an LLM is performing. By measuring causal influence distribution across features, we can distinguish between different computational regimes.

**What this detects:** The *type* and *complexity* of thinking.
**What this doesn't detect:** Whether the output is correct.

## Key Results

| Detection Task | Raw d | After Length Control | Status |
|----------------|-------|---------------------|--------|
| Task type (grammar vs reasoning) | 3.2 | **1.08** | **Works** |
| Computational complexity | 2.4 | **0.87** | **Works** |
| Adversarial/anomalous inputs | 1.2 | — | Works |
| Truthfulness | 0.05 | — | Doesn't work |

**The pivot experiment:** Signal PERSISTS after regressing out text length. This is genuine computational regime difference, not length-driven scaling.

**The finding:** Simple tasks (grammar) produce focused, high-influence computation. Complex tasks (reasoning) produce diffuse, low-influence computation. True vs false statements look identical internally.

## What We Measure

| Metric | What It Captures | Diagnostic? |
|--------|------------------|-------------|
| `mean_influence` | Average causal strength between features | Yes (d=3.2) |
| `concentration` | Is influence focused or spread out? | Yes (d=2.4) |
| `mean_activation` | Signal strength | Yes (d=1.7) |
| `n_active` | Feature count | No (confounded by length) |

**Robust metrics:** influence, concentration, activation
**Confounded metrics:** n_active, n_edges (just track text length)

## Use Cases

**Works for:**
- Input classification (what type of task is this?)
- Anomaly detection (is this input unusual?)
- Complexity estimation (how hard is the model working?)

**Doesn't work for:**
- Hallucination detection
- Fact-checking
- Output quality assessment

## Architecture

```
LatentDiagnostics
├── Feature extraction       SAE/transcoder decomposition
├── Attribution graph        Causal influence between features
├── Metric computation       influence, concentration, activation
└── Diagnostic analysis      Compare distributions across input classes
```

## Quick Start

```bash
pip install -e .

# Generate figures
python experiments/domain_figures.py

# Compute new attribution metrics (GPU)
modal run scripts/modal_general_attribution.py \
  --input-file data/domain_analysis/domain_samples.json \
  --output-file data/results/domain_attribution_metrics.json
```

## Project Structure

```
latent-diagnostics/
├── src/neural_polygraph/        # Core framework
├── experiments/
│   ├── domain_figures.py        # Paper figures
│   ├── diagnostics.py           # Statistical analysis
│   └── domain_analysis.py       # Cross-domain comparison
├── figures/domain_analysis/     # Generated figures
├── data/results/                # Attribution metrics
└── scripts/                     # Modal GPU runners (parallel)
```

## Limitations

1. **Requires model internals** — Only works on models with SAE/transcoder access
2. **Compute intensive** — ~30 sec/sample on A100
3. **Measures structure, not correctness** — Can't detect hallucinations or errors
4. **Length confound** — Must use influence/concentration, not feature counts

## Citation

```bibtex
@software{latent_diagnostics,
  title = {Latent Diagnostics: Representation-Level Analysis of Internal Activation Structure in Large Language Models},
  year = {2026}
}
```

## License

MIT
