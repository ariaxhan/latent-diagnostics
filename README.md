# Latent Diagnostics

Measuring computational regimes inside LLMs via attribution graph geometry.

## What This Is

We extract attribution graphs from model internals (via transcoders/SAEs) and compute metrics that characterize different computational patterns. Different task types produce measurably different geometric signatures.

## Key Findings

### 1. Length-Controlled Metric Effects

| Metric | What It Measures | Effect Size (Cohen's d) |
|--------|------------------|------------------------|
| Influence | Causal strength between features | d=1.08 |
| Concentration | Focused vs diffuse computation | d=0.87 |
| N_active | Feature count | d=0.07 (length artifact) |

### 2. Geometric Structure of Task Domains

Using inertia tensor analysis (adapted from AIDA-TNG galaxy morphology):

| Domain | Shape | Effective Dim | Interpretation |
|--------|-------|---------------|----------------|
| Grammar | Prolate | 2.19 | Focused along one computational axis |
| Commonsense | Prolate | 2.26 | Slightly more distributed |
| NLI | Prolate | 1.73 | Most concentrated variance |
| Paraphrase | Prolate | 2.16 | Similar to grammar |

All task domains are **prolate** (cigar-shaped in metric space) — variance concentrates along a dominant axis rather than spreading uniformly.

### 3. What Works vs What Doesn't

**Works:**
- Task type classification (grammar vs reasoning)
- Computational complexity estimation
- Anomaly detection (out-of-distribution inputs)

**Doesn't work:**
- Hallucination detection
- Truthfulness detection (d=0.05)
- Output correctness prediction

The model processes hallucinations with the same internal geometry as truthful statements.

## The Journey

1. **Dec 2025:** Started with hallucination detection via feature spectroscopy
2. **Jan 2026:** Discovered most "signal" was text length confounding (r=0.98)
3. **Feb 2026:** Pivoted to task-type diagnostics with length control
4. **Mar 2026:** Added geometric analysis — domain shapes in metric space

See `archive/disproved/` for early experiments with honest disclaimers.

## Directory Structure

```
notebooks/                    # START HERE - narrative series
  01_introduction.ipynb       # What this project discovers
  02_the_journey.ipynb        # From hallucination detection to task diagnostics
  03_methodology.ipynb        # How we extract and analyze metrics
  04_core_results.ipynb       # Main findings with visualizations
  05_negative_results.ipynb   # What doesn't work (and why)

experiments/
  core/                       # Main analyses (geometric_analysis.py, etc.)
  statistics/                 # Statistical tests
  visualization/              # Figure generation
  utilities/                  # Shared code
  _archive/                   # Historical experiments
  _runs/                      # Timestamped analysis outputs

figures/paper/                # Core figures
data/results/                 # Computed metrics (JSON)
scripts/                      # Modal GPU runners
archive/disproved/            # Early work with honest post-mortems
```

## Quick Start

```bash
pip install -e .

# Run geometric analysis
python experiments/core/geometric_analysis.py --analyze

# Generate figures
python experiments/visualization/generate_figures.py

# Compute attribution metrics (GPU, parallel)
modal run scripts/modal_general_attribution.py \
  --input-file data/domain_analysis/domain_samples.json \
  --output-file data/results/domain_attribution_metrics.json
```

## Methodology

1. **Attribution Graphs:** Extract causal graphs via circuit-tracer showing feature→feature influence during inference

2. **Metrics:**
   - `mean_influence`: Average edge weight (causal strength)
   - `concentration`: Gini coefficient of influence distribution
   - `mean_activation`: Feature activation strength

3. **Length Control:** Residualize against text length (raw n_active correlates r=0.98 with tokens)

4. **Geometric Analysis:** Treat domain samples as point clouds in 6D metric space, compute shape via inertia tensor eigendecomposition (axis ratios, effective dimensionality)

## Limitations

- **Requires model internals** — SAE/transcoder access (currently Gemma 2 via Goodfire)
- **Compute intensive** — ~30 sec/sample on A100
- **Measures structure, not correctness** — can't detect hallucinations
- **Length confounding** — must residualize (raw n_active is artifact)

## License

MIT
