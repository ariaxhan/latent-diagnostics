# Latent Diagnostics

**Language models switch between different internal processing modes depending on the task type - and those modes are measurable.**

## What This Does

We measure *how* a model computes, not *whether* it's correct. By extracting attribution graphs from model internals (via transcoders/SAEs), we compute metrics that characterize the computational regime: is the model doing focused grammatical processing, or diffuse multi-hop reasoning?

This is like measuring heart rate and brain patterns: you can tell if someone is doing math versus poetry (different patterns), and whether they're confused (elevated activity) - but you can't tell if they got the math right.

## Key Discovery

After controlling for text length, we found:

| Metric | What It Measures | After Length Control |
|--------|------------------|---------------------|
| Influence | Causal strength between features | d=1.08 (genuine signal) |
| Concentration | Focused vs diffuse computation | d=0.87 (genuine signal) |
| N_active | Feature count | d=0.07 (COLLAPSES - was length artifact) |

**The pattern:**
- Grammar tasks (CoLA): High influence, high concentration = focused computation
- Reasoning tasks (WinoGrande, HellaSwag): Low influence, low concentration = diffuse computation
- Truthfulness (TruthfulQA): No signal (d=0.05) - true/false statements look identical internally

## The Journey

This repo documents our research journey:

1. **Started:** Hallucination detection via feature spectroscopy (Dec 2025)
2. **Realized:** Most "signal" was text length confounding (Jan 2026)
3. **Pivoted:** Task-type diagnostics with length control (Feb 2026)
4. **Found:** Genuine computational regime differences

See `archive/disproved/` for our early experiments with honest disclaimers about what didn't work.

## Directory Structure

```
notebooks/           # Main analysis notebooks (START HERE)
experiments/         # Reproducible experiment scripts
figures/             # All generated figures
  paper/             # Publication-quality figures
data/                # Input data and results
  results/           # Computed metrics (JSON)
scripts/             # Modal runners for GPU computation
archive/             # Historical experiments
  disproved/         # Early work superseded by length control
```

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

## What Works vs What Doesn't

**Works:**
- Task type classification (grammar vs reasoning vs paraphrase)
- Computational complexity estimation
- Anomaly/adversarial input detection

**Doesn't Work:**
- Hallucination detection
- Truthfulness detection
- Output correctness prediction

The model processes hallucinations and false statements with the same internal structure as truthful ones. This is a fundamental limitation: we measure computation type, not output quality.

## How It Works

1. **Attribution Graphs:** We use circuit-tracer to extract causal graphs showing how features influence each other during inference. Each node is a feature (sparse autoencoder direction), each edge is causal influence.

2. **Metrics:** From these graphs we extract:
   - `mean_influence`: Average edge weight (how strongly features drive each other)
   - `concentration`: How focused the influence is (Gini coefficient)
   - `mean_activation`: Average feature activation strength

3. **Length Control:** Raw feature counts correlate r=0.98 with text length - longer inputs activate more features, trivially. We residualize metrics against length to isolate genuine computational differences.

4. **The Signal:** After length control, grammar tasks still show d=1.08 higher influence than reasoning tasks. This isn't length - it's genuine regime difference.

## Limitations

1. **Requires model internals** - Only works on models with SAE/transcoder access (currently Gemma 2 via Goodfire)
2. **Compute intensive** - ~30 sec/sample on A100
3. **Measures structure, not correctness** - Can't detect hallucinations or factual errors
4. **Must control for length** - Raw n_active is confounded (r=0.98 with token count)

## License

MIT
