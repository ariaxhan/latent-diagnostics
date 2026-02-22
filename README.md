# Neural Polygraph

Interpretability-based detection for language models: hallucinations and prompt injection.

## Prompt Injection Detection (NEW)

**Breakthrough:** Attribution graphs strongly discriminate injection from benign (Cohen's d > 1.0).

```bash
# Run on Modal GPU
modal run modal_attribution.py
```

| Metric | Cohen's d | Pattern |
|--------|-----------|---------|
| Top-100 Concentration | 1.30 | Injection LOWER |
| N Active Features | 1.07 | Injection HIGHER |
| N Edges | 1.03 | Injection HIGHER |
| Mean Influence | 1.12 | Injection LOWER |

**Key insight:** Injection creates diffuse, scattered causal graphs. Benign prompts have focused pathways.

See `_meta/handoffs/2026-02-21-attribution-breakthrough.md` for details.

---

## Hallucination Detection (Original)

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Verify (quick check)
python verify_setup.py

# 3. Run experiment
python run_experiment.py 01_spectroscopy

# 4. Visualize
python experiments/visualize_spectroscopy.py
```

**Note:** Use `python test_setup.py` for a comprehensive verification of all dependencies.

## Structure

```
neural-polygraph/
├── src/hallucination_detector/    # Core package
│   ├── sae_utils.py                # SAE feature extraction
│   ├── geometry.py                 # Geometric analysis
│   ├── data_loader.py              # HB-1000 benchmark loader
│   └── storage.py                  # Experiment storage
│
├── experiments/                    # Experiment protocols
│   ├── 01_spectroscopy.py          # Experiment A
│   ├── visualize_spectroscopy.py   # Visualization
│   └── data/                       # HB-1000 benchmark (~1000 samples)
│
├── run_experiment.py               # Universal runner
├── test_setup.py                   # Setup verification
└── TESTING-PLANS.MD                # Research plan
```

## Usage

### Run Experiments

```bash
# List available experiments
python run_experiment.py --list

# Run Experiment A: Spectroscopy
python run_experiment.py 01_spectroscopy

# View results
python run_experiment.py --view 01_spectroscopy
```

### Programmatic Usage

```python
from hallucination_detector import (
    HB_Benchmark,
    ExperimentStorage,
    compute_inertia_tensor,
)

# Load benchmark
benchmark = HB_Benchmark("experiments/data")
benchmark.load_datasets()
benchmark.load_model_and_sae(layer=5, width="16k")

# Get activations
activations = benchmark.get_activations("The Eiffel Tower is in Paris")
print(f"L0 Norm: {activations.l0_norm}")
print(f"Reconstruction Error: {activations.reconstruction_error:.4f}")

# Save results
from pathlib import Path
storage = ExperimentStorage(Path("experiments/my_experiment"))
storage.write_manifest({"experiment": "my_experiment"})
storage.write_metrics({"metric": [...]})
```

## Experiments

### Experiment A: Spectroscopy ✅

**Goal:** Demonstrate distinct spectral signatures of hallucinations

**Metrics:** L0 Norm, Reconstruction Error, Gini Coefficient

**Run:** `python run_experiment.py 01_spectroscopy`

### Experiment B: Geometry 🚧

**Goal:** Measure the "shape" of thoughts using inertia tensors

**Status:** Coming soon

### Experiment C: Ghost Features 🚧

**Goal:** Identify features unique to hallucinations

**Status:** Coming soon

## Data: HB-1000 Benchmark

| Dataset | Samples | Description |
|---------|---------|-------------|
| Entity Swaps | 230 | Geographic/entity errors |
| Temporal Shifts | 270 | Temporal errors |
| Logical Inversions | 250 | Logical flips |
| Adversarial Traps | 250 | High-probability misconceptions |

**Total:** ~1,000 fact/hallucination pairs in `experiments/data/`

## Dependencies

Core: `torch`, `transformer-lens`, `sae-lens`, `numpy`, `polars`

Viz: `matplotlib`, `seaborn`, `plotly`

Analysis: `scikit-learn`, `umap-learn`

See `pyproject.toml` for complete list.

## Troubleshooting

**Import errors:** `pip install -e .`

**Memory issues:** Use CPU mode or smaller batches

**Model download:** Models download from Hugging Face (~2GB)

**Test setup:** `python test_setup.py` verifies everything

## Research Plan

See `TESTING-PLANS.MD` for detailed experimental protocols and hypotheses.

## License

MIT License
