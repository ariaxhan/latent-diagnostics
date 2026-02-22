# Experiments

## Active Experiments (Injection Detection)

| Script | Purpose |
|--------|---------|
| `detection.py` | Main detection experiment — calibrate and evaluate on PINT benchmark |
| `analyze.py` | Deep geometry analysis — entropy, distributions, extreme cases |
| `visualize.py` | Generate figures — ROC, distributions, radar charts |

### Core Data

- **Attribution metrics**: `../data/results/pint_attribution_metrics.json` (136 samples from PINT)
- **Trained detector**: `injection_detection/runs/*/detector.json`

### Running

```bash
# Step 1: Compute attribution metrics on Modal GPU
modal run scripts/modal_pint_benchmark.py

# Step 2: Run detection experiment
python experiments/detection.py

# Step 3: Visualize results
python experiments/visualize.py

# Step 4: Deep analysis
python experiments/analyze.py
```

## Archive

The `archive/` directory contains historical experiments from hallucination detection research. These informed the pivot to injection detection but are not part of the active workflow.

**Key insight from archived work:** SAE activation patterns (spectroscopy) failed to discriminate hallucinations (Cohen's d < 0.1). The breakthrough came from switching to attribution graphs — measuring how features influence each other, not just which ones activate.

### Archived Experiments

| Experiment | Method | Result |
|------------|--------|--------|
| `01_spectroscopy.py` | L0 norm, reconstruction error, Gini | Failed (d < 0.1) |
| `02_geometry.py` | Inertia tensor, sphericity | Limited signal |
| `03_ghost_features.py` | Differential feature identification | Interesting but not detection |
| `04-07_*.py` | Various approaches | See individual files |

The archived work is preserved as research history — it shows why we pivoted from "what features activate" to "how features influence each other."
