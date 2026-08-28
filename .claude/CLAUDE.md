# latent-diagnostics

## compression — mandatory

minimum text. zero meaningful loss.

* delete anything removable.
* merge anything redundant.
* shorten anything compressible.
* prefer structure over prose.
* preserve all meaning that affects correctness, clarity, decisions, evidence, uncertainty, or action.
* length follows information. never pad; never truncate for brevity.

before emitting:

1. perform a deletion pass.
2. remove every word, sentence, bullet, section, preamble, recap, transition, or explanation whose removal does not materially reduce meaning.
3. repeat until no further lossless deletion is possible.

do not emit while removable text remains.

verbosity is a defect.
omission is also a defect.

tokens: ~120 | inherits: CodingVault KERNEL

---

## Project

**Name:** neural-polygraph (latent-diagnostics)
**Description:** Adversarial geometry - detecting malicious prompts via causal graph topology
**Domain:** ML interpretability research, SAE spectroscopy, vector native analysis

## Stack

- Python 3.11.7
- PyTorch >= 2.0.0
- transformer-lens >= 1.16.0
- sae-lens >= 3.0.0
- polars, pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn, umap-learn
- Jupyter notebooks

## Quality Baseline (2026-03-12)

| Metric | Count | Status |
|--------|-------|--------|
| Empty except blocks | 63 | Review |
| Missing type hints | 109 | Acceptable (research code) |
| SQL injection risk | 0 | Clean |
| Hardcoded secrets | 0 | Clean |
| Input validation | 0 | N/A (no web inputs) |

## Structure

```
experiments/          # Active experiments
  core/              # Main research code
  statistics/        # Statistical analysis
  visualization/     # Figure generation
  utilities/         # Shared helpers
notebooks/           # Jupyter notebooks (research narrative)
src/                 # Package code (neural_polygraph)
tests/               # pytest tests
data/                # Input data
figures/             # Generated figures
archive/             # Disproved/superseded work
_meta/               # KERNEL metadata
```

## Conventions

- Research code: type hints optional, clarity over strictness
- Notebooks: narrative-first, executable documentation
- Scripts: type hints required, production-quality
- Tests: pytest, hypothesis-driven

## Active Contract

See `_meta/agentdb/` for active work state.
