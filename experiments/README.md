# Hallucination Detection Experiments

This directory contains the complete experimental pipeline for the paper:

**"The Shape of Errors: Spectroscopic and Geometric Analysis of SAE Features"**

## Overview

We analyze hallucinations in language models using Sparse Autoencoders (SAEs) as measurement tools. Our approach combines two novel methodologies:

1. **Spectroscopy**: Analyzing the "spectrum" of feature activations (sparsity, reconstruction error)
2. **Geometric Topology**: Treating features as point masses and computing their inertia tensor to reveal structural properties

## Experiments

### Experiment A: Pure Spectroscopy
**Script**: `01_spectroscopy.py`  
**Goal**: Demonstrate that hallucinations have distinct spectral signatures

**Metrics**:
- L0 Norm (sparsity)
- Reconstruction Error (off-manifold detection)
- Gini Coefficient (focus/diffusion)

**Hypothesis**: Hallucinations activate "loose association" features that lack inhibition.

**Output**: Figure 1 - Spectral shift between facts and hallucinations

---

### Experiment B: Geometric Topology (CORE NOVELTY)
**Script**: `02_geometry.py`  
**Goal**: Measure the "shape" of thoughts using inertia tensor analysis

**Metrics**:
- Sphericity (c/a ratio) - Are facts triaxial, hallucinations spherical?
- Shape Classification (Spherical, Oblate, Prolate, Triaxial)
- Effective Dimensionality (participation ratio)
- Misalignment Angle (centroid vs major axis)

**Hypothesis**: 
- Facts are **TRIAXIAL** (highly structured, specific direction)
- Hallucinations are **SPHERICAL** (isotropic confusion) or **OBLATE** (flat, lacking depth)

**Output**: **Figure 2 - Topological Phase Plot** (Main result)

---

### Experiment C: Ghost Feature Finder
**Script**: `03_ghost_features.py`  
**Goal**: Identify features uniquely active in hallucinations

**Method**:
1. Calculate differential spectrum: `Act_diff = Act_hallucination - Act_fact`
2. Identify top-K features with highest positive values
3. Decode features to vocabulary to understand semantic meaning

**Hypothesis**: Ghost features represent conflicting or tangential semantic concepts.

**Output**: **Figure 3 - Feature Prism** (Case study with semantic interpretation)

---

### Experiment D: Layer Sensitivity
**Script**: `04_layer_sensitivity.py`  
**Goal**: Show *where* in the model hallucinations emerge

**Method**: Run spectroscopy and geometry analysis across layers 5, 12, and 20

**Hypothesis**: Mid-to-late layers show stronger hallucination signatures as semantic processing deepens.

**Output**: **Figure 4 - Layer Sensitivity** (Detection accuracy across layers)

---

## Quick Start

### Run All Experiments

```bash
# Full pipeline (takes ~2-3 hours)
python experiments/run_all_experiments.py

# Quick test (reduced samples)
python experiments/run_all_experiments.py --quick

# Skip visualizations
python experiments/run_all_experiments.py --skip-viz
```

### Run Individual Experiments

```bash
# Experiment A
python experiments/01_spectroscopy.py

# Experiment B (Core novelty)
python experiments/02_geometry.py

# Experiment C
python experiments/03_ghost_features.py

# Experiment D
python experiments/04_layer_sensitivity.py
```

### Generate Visualizations

```bash
# After running experiments, generate figures
python experiments/visualize_spectroscopy.py
python experiments/visualize_geometry.py
python experiments/visualize_ghost_features.py
python experiments/visualize_layer_sensitivity.py
```

## Data Requirements

### Benchmark Datasets (HB-1000)

The experiments require 4 benchmark datasets in `experiments/data/`:

- `bench_entity_swaps.json` - Concrete noun/location errors (Paris → Rome)
- `bench_temporal_shifts.json` - Continuous variable errors (1969 → 1975)
- `bench_logical_inversions.json` - Structural/relational flips (A > B → B > A)
- `bench_adversarial_traps.json` - High-probability misconceptions

Each dataset contains ~250 samples with structure:
```json
{
  "id": "entity_001",
  "domain": "entity",
  "complexity": 1,
  "prompt": "The Eiffel Tower is located in",
  "fact": "Paris",
  "hallucination": "Rome"
}
```

## Output Structure

Each experiment creates a timestamped run directory:

```
experiments/
├── 01_spectroscopy/
│   └── runs/
│       └── 20251227_120000/
│           ├── manifest.json       # Experiment metadata
│           ├── metrics.parquet     # Results data
│           └── figures/            # Generated visualizations
│               └── fig1_*.png
├── 02_geometry/
│   └── runs/
│       └── 20251227_130000/
│           ├── manifest.json
│           ├── metrics.parquet
│           └── figures/
│               └── fig2_topology_phase_plot.png  # KEY FIGURE
├── 03_ghost_features/
│   └── runs/
│       └── 20251227_140000/
│           ├── manifest.json
│           ├── metrics.parquet
│           ├── feature_decodings.json
│           └── figures/
│               └── fig3_feature_prism.png        # KEY FIGURE
└── 04_layer_sensitivity/
    └── runs/
        └── 20251227_150000/
            ├── manifest.json
            ├── metrics.parquet
            └── figures/
                └── fig4_layer_sensitivity.png    # KEY FIGURE
```

## Key Figures for Paper

After running all experiments and visualizations:

1. **Figure 1**: Spectral Shift (Reconstruction Error histograms)
   - `01_spectroscopy/figures/reconstruction_error_comparison.png`

2. **Figure 2**: Topological Phase Plot (MAIN NOVELTY)
   - `02_geometry/figures/fig2_topology_phase_plot.png`
   - Shows sphericity vs sparsity, demonstrating geometric degradation

3. **Figure 3**: Feature Prism (Case Study)
   - `03_ghost_features/figures/fig3_feature_prism.png`
   - Qualitative proof of "ghost features"

4. **Figure 4**: Layer Sensitivity
   - `04_layer_sensitivity/figures/fig4_layer_sensitivity.png`
   - Shows where hallucinations emerge in the network

## Technical Details

### Model & SAE
- **Model**: Gemma-2-2B (via TransformerLens)
- **SAE**: GemmaScope (16k features, canonical)
- **Layers**: 5 (default), 12, 20 (for layer sensitivity)

### Compute Requirements
- **Device**: Auto-detects MPS (Apple Silicon), CUDA, or CPU
- **Memory**: ~8GB GPU memory recommended
- **Time**: 
  - Single experiment: 20-30 minutes
  - Full pipeline: 2-3 hours

### Dependencies
```
torch
transformer_lens
sae_lens
polars
matplotlib
seaborn
numpy
```

## Interpretation Guide

### Spectroscopy Metrics
- **High L0 Norm**: More features active (diffuse representation)
- **High Reconstruction Error**: Off-manifold (SAE struggles to reconstruct)
- **Low Gini Coefficient**: Unfocused (energy spread across many features)

### Geometry Metrics
- **High Sphericity (c/a → 1)**: Isotropic, confused representation
- **Low Sphericity (c/a → 0)**: Directional, structured representation
- **Triaxial Shape**: Highly structured (typical of facts)
- **Spherical/Oblate Shape**: Degraded structure (typical of hallucinations)

### Ghost Features
- Features uniquely active in hallucinations
- Represent conflicting or tangential semantic concepts
- Decoded to vocabulary for interpretation

## Troubleshooting

### Out of Memory
- Reduce batch size in data loader
- Use CPU instead of GPU: `export DEVICE=cpu`
- Run experiments individually instead of full pipeline

### Missing Data
- Ensure all 4 benchmark files exist in `experiments/data/`
- Check file format matches expected JSON structure

### Import Errors
- Ensure you're running from project root
- Check that `src/hallucination_detector` is installed: `pip install -e .`

## Citation

If you use this experimental framework, please cite:

```bibtex
@article{hallucination_geometry_2025,
  title={The Shape of Errors: Spectroscopic and Geometric Analysis of SAE Features},
  author={[Your Name]},
  journal={[Venue]},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

