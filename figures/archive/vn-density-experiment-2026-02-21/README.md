# Vector Native Density Experiment (2026-02-21)

**Status:** Archived - Superseded by length-controlled analysis

**Commit:** cb37a487977ddf69de314f1c81f92555e6dbb1bd

---

## Historical Context

This experiment was investigating prompt injection detection using activation topology. The initial hypothesis was that injections had a distinct "geometric signature" - more active features, lower concentration.

## Discovery

While analyzing the results, we discovered that the "geometric signature" was actually a **length confound** (r=0.96 correlation between N Active and text length). This led to two pivotal findings:

1. **The injection detection hypothesis was wrong** - the signal was a length artifact
2. **Vector Native density hypothesis emerged** - structured symbols activate denser representations

## VN Density Results (Gemma-2-2b)

### Symbol Comparison (VN wins all 4)
- ● = 865 features/char vs "attention" = 94 (9.2x)
- ⊕ = 912 features/char vs "add" = 256 (3.6x)
- → = 826 features/char vs "then" = 165 (5.0x)
- ≠ = 809 features/char vs "not" = 191 (4.2x)

### Format Comparison (VN wins all 4)
- VN syntax beats natural language by 50-105%

## Key Insight

Vector Native isn't just token compression - it's **density optimization**. Structured symbols activate more internal model representations per character.

---

## Archived Figures

| Figure | Description |
|--------|-------------|
| `density_predictors.png` | Predictors of feature density |
| `feature_density_factors.png` | Factors affecting feature activation density |
| `length_activation_core.png` | Core relationship between length and activation |
| `length_geometry_relationship.png` | Comprehensive length-geometry analysis |
| `vn_format_density_real.png` | VN format vs natural language density |
| `vn_symbol_density_real.png` | VN symbol vs word density |

---

## What Happened Next

This discovery triggered the **pivot experiment** (commit 34dc255):
- Implemented length-controlled (residualized) metrics
- Tested if domain signatures persist after length control
- Result: **Signal persists** - Influence (d=1.08), Concentration (d=0.87)
- N Active collapsed to d=0.07 (was pure length artifact)

The project pivoted from "injection detection" to "latent diagnostics of cognitive regimes" - a more fundamental finding.

---

## Lessons Learned

1. **Always control for confounds** - length is a major one in NLP
2. **Negative results can lead to discoveries** - the injection hypothesis failing revealed something more important
3. **Residualization is crucial** - raw metrics can be misleading

---

## Related Scripts

- Generation: `scripts/modal_vn_density.py` (runs on Modal with GPU)
- Analysis: `experiments/vn_density_test.py`

---

Archived: 2026-02-23
Reason: Superseded by length-controlled analysis (figures/paper/)
