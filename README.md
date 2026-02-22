# Latent Diagnostics

**Representation-level analysis of internal activation structure in large language models.**

> Latent structure is measurable, stable, and diagnostic.

## Core Thesis

Current evaluation of LLMs focuses on outputs. This is insufficient.

We introduce a representation-level diagnostic framework that analyzes **internal activation topology** to characterize model behavior independently of surface text. By measuring feature activation density, causal graph structure, concentration, and distributional entropy across prompts, we demonstrate that distinct behavioral regimes correspond to distinct internal structural patterns.

The framework enables systematic inspection of internal state rather than relying solely on output behavior.

## What We Measure

| Metric | What It Captures |
|--------|------------------|
| `n_active` | Feature count — how many pathways participate |
| `n_edges` | Causal connections — interaction density |
| `mean_influence` | Edge strength — pathway dominance |
| `concentration` | Top-k share — focused vs diffuse computation |
| `entropy` | Output uncertainty |

**Key insight:** Inputs change the *shape* of computation, not just outputs.

## Architecture

```
LatentDiagnostics
├── Feature extraction layer        SAE, transcoder, attention
├── Attribution graph construction  Causal influence edges
├── Metric computation              n_active, n_edges, concentration, entropy
├── Diagnostic analysis layer       Distributional comparison, regime identification
└── Experiment harness              Reproducible, immutable runs
```

This is not a model. It is a **measurement system**.

## Quick Start

```bash
pip install -e .

# Run diagnostics
python experiments/diagnostics.py

# Compute attribution metrics (GPU)
modal run scripts/modal_general_attribution.py \
  --input-file data/domain_analysis/domain_samples.json \
  --output-file data/results/domain_attribution_metrics.json
```

## Project Structure

```
latent-diagnostics/
├── src/neural_polygraph/        # Core measurement framework
│   ├── datasets.py              # Unified loaders (10+ datasets)
│   ├── feature_extractors.py    # SAE/transcoder interface
│   └── geometry.py              # Graph metrics
├── experiments/
│   ├── diagnostics.py           # Comprehensive A-G diagnostic suite
│   ├── domain_analysis.py       # Cross-domain signatures
│   └── truthfulness_analysis.py # Factual coherence
├── data/
│   ├── domain_analysis/         # 400 samples, 8 domains
│   ├── truthfulness/            # 200 balanced samples
│   └── results/                 # Attribution metrics
└── scripts/                     # Modal GPU runners
```

## Why This Is Novel

Most prior work evaluates outputs, measures perplexity/accuracy, or probes neurons in isolation.

This framework:
1. Treats activation topology as a structured object
2. Computes graph-level metrics over attribution flows
3. Compares internal regimes across behavioral classes
4. Provides a reusable diagnostic layer for LLM systems

## Input Classes

| Class | Purpose |
|-------|---------|
| Domain-specific | Code, scientific, legal, poetry signatures |
| Truthful vs false | Factual coherence |
| Prompt injection | Adversarial stress test (labeled data) |
| Benign queries | Well-formed baseline |

Prompt injection is a labeled stress test dataset, not the research subject.

## Current State

- **Infrastructure:** Complete
- **Data prepared:** Domain (400), Truthfulness (200)
- **Data computed:** PINT (136) — others pending Modal runs
- **Known:** Length confound r=0.96 for n_active; mean_activation least confounded (r=-0.224), best AUC (0.830)

## Citation

```bibtex
@software{latent_diagnostics,
  title = {Latent Diagnostics: Representation-Level Analysis of Internal Activation Structure in Large Language Models},
  year = {2026}
}
```

## License

MIT
