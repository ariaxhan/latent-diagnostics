# Archive

Historical experiments preserved for reference. Most were disproved or superseded.

## Contents

### hallucination_detection/

Early experiments (01-07) attempting to detect hallucinations via activation patterns.

**Why archived:** All effect sizes < 0.1 after length control. The "signal" was text length, not hallucination. See `archive/disproved/` in repo root for detailed post-mortems.

### injection_detection/

Prompt injection detection experiments.

**Status:** Partially validated (d~0.8) but superseded by domain analysis approach.

### deprecated/

Scripts superseded by cleaner implementations:

| Script | Replaced By |
|--------|-------------|
| `diagnostics.py` | Split into core/ and statistics/ |
| `analyze.py` | `core/domain_comparison.py` |
| `detection.py` | `core/` scripts |
| `visualize.py` | `visualization/` scripts |
| `feature_overlap.py` | Metric-based similarity approach |
| `vn_density_test.py` | No longer used |

## Lessons Learned

1. **Length confounding**: n_active and n_edges correlate r>0.95 with text length
2. **Threshold calibration**: Never calibrate on the same data you evaluate
3. **Class imbalance**: Accuracy is misleading with unbalanced classes

See `archive/disproved/README.md` in repo root for full details on why early approaches failed.
