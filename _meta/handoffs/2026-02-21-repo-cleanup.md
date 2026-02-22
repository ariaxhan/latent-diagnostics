# CONTEXT HANDOFF: Repo Restructure + README Reframe

**Summary**: neural-polygraph evolved from hallucination detection (failed) to injection detection via attribution graphs (succeeded) — repo structure doesn't reflect this pivot.

---

## Goal

Clean up repo structure and rewrite README to reflect the **Adversarial Geometry** thesis: using mechanistic interpretability to detect malicious prompts by their causal graph topology.

## Current State

- **Branch**: `feature/injection-detector` (8 commits ahead of main)
- **Working code**: Injection detector achieving 80.5% on PINT benchmark
- **Key finding**: Cohen's d > 1.0 on 4 metrics (concentration, n_active, mean_influence, n_edges)
- **Problem**: File structure is a mess from evolution — hallucination stuff mixed with injection stuff

## What Exists (Categorized)

### KEEP - Core Package
```
src/hallucination_detector/
├── __init__.py
├── injection_detector.py    # NEW: Attribution-based detector
├── pint_loader.py           # NEW: PINT benchmark loader
├── storage.py               # Experiment storage (works well)
├── data_loader.py           # HB-1000 loader (hallucination)
├── sae_utils.py             # SAE extraction (shared)
├── geometry.py              # Geometric metrics (shared)
└── feature_extractors.py    # NEW: SAE/Transcoder extractors
```

### KEEP - Injection Detection (New Work)
```
experiments/08_injection_detection.py       # Main experiment
experiments/analyze_injection_geometry.py   # Deep analysis
experiments/visualize_injection_detection.py
modal_pint_benchmark.py                     # Modal runner
notebooks/injection_geometry_explained.ipynb # Research notebook
pint_metrics.json                           # Results data
```

### KEEP - Research Notes
```
_meta/research/adversarial-geometry-thesis.md  # Core thesis
_meta/research/physics-foundations.md          # Theory
_meta/data/attribution-pilot-results.json      # Pilot data
```

### ARCHIVE/REORGANIZE - Hallucination Experiments
```
experiments/01_spectroscopy.py    # Failed - d < 0.1
experiments/02_geometry.py        # Limited success
experiments/03_ghost_features.py  # Interesting but not detection
experiments/04-07_*.py            # Various attempts
```

### DELETE/ARCHIVE - Clutter
```
test_*.py (at root)              # One-off tests, not real test suite
modal_app.py                     # Old, superseded by modal_pint_benchmark.py
modal_attribution.py             # Old, superseded
docker/                          # Never worked well
.cursor/                         # Cursor-specific, not needed
CLI_SUMMARY.md, README_CLI.md    # Old CLI docs
QUICK_START.md                   # Outdated
tutorials/                       # May be outdated
```

## Decisions Made

1. **Attribution graphs > cosine similarity** — Cosine similarity failed (d=0.24), attribution succeeded (d>1.0)
2. **"Diffuse graph = injection" hypothesis** — Injection creates scattered influence, benign has focused pathways
3. **Entropy framing** — Physics analogy works well for explaining the phenomenon
4. **ExperimentStorage pattern** — Works well, keep using it

## Proposed New Structure

```
neural-polygraph/
├── README.md                    # NEW: Adversarial Geometry thesis
├── pyproject.toml
├── requirements.txt
│
├── src/
│   └── neural_polygraph/        # RENAME from hallucination_detector
│       ├── detection/
│       │   ├── injection.py     # Injection detector
│       │   └── hallucination.py # (archive or remove)
│       ├── analysis/
│       │   ├── attribution.py   # Graph metrics
│       │   └── geometry.py      # Shape metrics
│       ├── data/
│       │   ├── pint.py          # PINT loader
│       │   └── hb1000.py        # HB-1000 loader
│       └── storage.py
│
├── experiments/
│   ├── injection/               # NEW: Group injection experiments
│   │   ├── 01_attribution_analysis.py
│   │   ├── 02_threshold_calibration.py
│   │   └── visualize.py
│   └── archive/                 # OLD: Hallucination experiments
│       └── hallucination/
│
├── notebooks/
│   └── injection_geometry_explained.ipynb
│
├── scripts/
│   └── modal_pint_benchmark.py  # Move from root
│
├── research/                    # Move from _meta/research
│   ├── adversarial-geometry-thesis.md
│   └── ...
│
└── data/
    └── results/
        └── pint_metrics.json
```

## README Should Cover

1. **What this is**: Adversarial geometry — detecting malicious prompts by their internal "shape"
2. **The discovery**: Injection = diffuse graphs, benign = focused graphs
3. **Key metrics**: concentration, n_active, mean_influence, n_edges
4. **Quick start**: How to run the detector
5. **Results**: 80.5% PINT score, Cohen's d > 1.0
6. **Theory**: Entropy analogy, competing semantic frames
7. **Status**: Research in progress, not production-ready

## Warnings

- **Don't delete experiment results** — `experiments/*/runs/` has valuable data
- **Package rename is breaking** — Will need to update all imports
- **modal_pint_benchmark.py has uncommitted changes** — Commit first
- **Hallucination work isn't useless** — Archive, don't delete (ghost features interesting)

## Open Questions for New Session

1. Rename package `hallucination_detector` → `neural_polygraph`? (breaking change)
2. Keep hallucination experiments or archive completely?
3. How much of _meta/ to preserve vs move to research/?

## File Paths to Read First

1. `_meta/research/adversarial-geometry-thesis.md` — Core thesis
2. `notebooks/injection_geometry_explained.ipynb` — Full explainer
3. `README.md` — Current state (outdated)
4. `experiments/08_injection_detection.py` — Main experiment

## Continuation Prompt

```
Resume neural-polygraph repo cleanup. The project evolved from hallucination detection (failed, Cohen's d < 0.1) to injection detection via attribution graphs (succeeded, Cohen's d > 1.0). The thesis: "injection = diffuse causal graphs, benign = focused pathways."

Current state: working detector at 80.5% PINT score, but file structure is messy — old hallucination experiments mixed with new injection work.

Task: (1) Reorganize file structure to reflect the Adversarial Geometry thesis, (2) Rewrite README.md to explain the project clearly. Read _meta/handoffs/2026-02-21-repo-cleanup.md for full context and proposed structure.
```
