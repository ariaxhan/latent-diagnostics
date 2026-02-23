# Disproved Experiments: The Path to Task-Type Diagnostics

This folder contains our early experiments on **prompt injection detection** via activation geometry. While the original hypothesis did not hold up, **the failure taught us critical lessons** that led to our current, validated approach.

---

## The Original Hypothesis

We hypothesized that **prompt injections create geometrically distinct activation patterns**:
- More features active
- Lower influence concentration
- Weaker per-edge influence

Initial results looked promising: Cohen's d > 1.0 on multiple metrics, p < 0.001.

## The Problem: Length Confounding

When we dug deeper, we discovered a critical confound:

| Metric | Correlation with Text Length |
|--------|------------------------------|
| `n_active` | r = 0.96 |
| `n_edges` | r = 0.92 |
| `top_100_concentration` | r = -0.74 |
| `mean_influence` | r = -0.83 |

**Injection prompts in our dataset were 74% longer than benign prompts.** The "geometric signature" was almost entirely explained by: *longer texts activate more features*.

When we controlled for length:
- Effect sizes collapsed from d > 1.0 to d ~ 0.1-0.5
- No metrics remained statistically significant

## What We Learned

1. **Raw feature counts are unreliable** - they scale with input length, not semantic content
2. **Always control for confounds** - especially length in text analysis
3. **Length-normalized metrics are essential** - influence per feature, not total influence
4. **Task-type detection works; injection-as-category does not** - different tasks genuinely differ in their activation profiles, but "injection" is not a coherent task category

## The Pivot

This failure led us to reframe the question:

**Old question:** Can we detect injection by activation geometry?
**New question:** Can we characterize **task types** by their activation profiles?

The answer to the second question is **yes**, with strong effect sizes (d > 1.5) that survive length control. Different cognitive tasks (QA, NLI, coreference, paraphrase detection) have distinct, reproducible activation signatures.

## Contents

### Notebooks

| File | Description |
|------|-------------|
| `unified_injection_analysis.ipynb` | Comprehensive analysis of injection geometry with class balance investigation |
| `balanced_injection_geometry.ipynb` | Balanced 1:1 sampling to test if signal persists |
| `injection_geometry_explained.ipynb` | Beginner-friendly explainer with visualizations |
| `injection_geometry_truth.ipynb` | **The key notebook** - documents the length confound discovery |

### Design Documents

| File | Description |
|------|-------------|
| `injection-detection-framework.md` | Physics-inspired detection framework (never implemented due to confound) |
| `adversarial-geometry-thesis.md` | Theoretical framework for adversarial activation patterns |
| `attack-cross-sections.md` | Proposed taxonomy of attack types and their "cross sections" |

### Scripts

| File | Description |
|------|-------------|
| `modal_balanced_benchmark.py` | Modal script for running balanced experiments |
| `modal_length_matched.py` | Modal script for length-controlled experiments |

### Figures

Various visualization outputs from the notebooks, including the critical `length_confound.png` that shows the r=0.96 correlation.

---

## Timeline

1. **Initial exploration** - Observed large effect sizes, claimed geometric signature
2. **Sanity check** - Balanced sampling, effects persisted but weakened
3. **Length analysis** - Discovered r=0.96 correlation between length and n_active
4. **Confound confirmation** - Length-matched analysis showed no significant differences
5. **Pivot** - Reframed to task-type diagnostics, which showed robust signal

## Key Takeaway

> **Negative results are results.** This failure was essential - it taught us what NOT to measure (raw counts) and what TO measure (length-normalized, influence-based metrics). The current diagnostics system exists because of what we learned from these failures.

---

## See Also

- `../notebooks/` - Current validated analysis with length-controlled metrics
- `../experiments/` - Replication scripts and raw data

*Last updated: February 2026*
