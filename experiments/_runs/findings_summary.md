# Latent Diagnostics: Consolidated Findings

Generated: 2026-03-12

## Dataset Overview

- **Samples analyzed:** 163 (filtered to domains with n≥10)
- **Domains:** grammar (50), commonsense (50), nli (50), paraphrase (13)
- **Model:** Gemma 2 2B via circuit-tracer

## 1. Length Confounding (Critical)

**Every metric correlates with text length.** Must residualize before comparison.

| Metric | Length r | Status |
|--------|----------|--------|
| n_active | +0.98 | CONFOUNDED |
| n_edges | +0.96 | CONFOUNDED |
| mean_influence | -0.81 | CONFOUNDED |
| top_100_concentration | -0.63 | moderate |
| logit_entropy | -0.54 | moderate |

**n_active and n_edges are unusable without residualization.**

## 2. Clean Effect Sizes (After Length Control)

Top discriminating metrics after residualization:

| Comparison | Metric | Cohen's d |
|------------|--------|-----------|
| commonsense vs paraphrase | logit_entropy | +4.06 |
| commonsense vs paraphrase | mean_influence | -3.25 |
| commonsense vs paraphrase | top_100_concentration | -3.22 |
| commonsense vs paraphrase | max_logit_prob | -3.01 |
| grammar vs paraphrase | logit_entropy | +2.47 |
| grammar vs commonsense | logit_entropy | -2.04 |

**Paraphrase is the outlier** — lowest logit_entropy (0.53 vs 1.0-1.6), meaning highest-confidence outputs.

## 3. Geometric Structure (Inertia Tensor Analysis)

All domains are **prolate** (cigar-shaped in metric space):

| Domain | Shape | Effective Dim | b/a | c/a |
|--------|-------|---------------|-----|-----|
| Grammar | Prolate | 2.19 | 0.567 | 0.385 |
| Commonsense | Prolate | 2.26 | 0.562 | 0.482 |
| NLI | Prolate | 1.73 | 0.549 | 0.190 |
| Paraphrase | Prolate | 2.16 | 0.517 | 0.482 |

**NLI has lowest effective dimensionality** (1.73) — most concentrated variance.

## 4. Metric Redundancy

Strong correlations (potential redundancy):

| Pair | r |
|------|---|
| n_active ↔ n_edges | 0.985 |
| logit_entropy ↔ max_logit_prob | -0.966 |
| mean_influence ↔ n_active | -0.796 |
| mean_influence ↔ top_100_concentration | 0.795 |

**Recommend:** Use (logit_entropy OR max_logit_prob) + (mean_influence OR top_100_concentration). Drop n_active/n_edges unless residualized.

## 5. What Works vs What Doesn't

### Works (d > 1.5 after length control)
- Task type classification (grammar vs commonsense vs nli vs paraphrase)
- Output confidence estimation (logit_entropy)
- Computational focus measurement (mean_influence, concentration)

### Doesn't Work
- Hallucination detection (d ≈ 0)
- Truthfulness detection (d ≈ 0.05)
- Raw feature counts (length confounded)

## 6. Next Steps (No New GPU)

1. **Cross-validate** — train classifier on 3 domains, test on 4th
2. **Bootstrap CIs** — confidence intervals on effect sizes
3. **Feature importance** — which metrics drive classification
4. **Residualized PCA** — cluster on length-controlled metrics

## 7. Next Steps (With GPU)

1. **Run new metrics** — PageRank, clustering, spectral gap (script ready)
2. **Cognitive regimes** — math/code/summarization (data prepared)
3. **Ablation experiments** — lesion top features, measure effect

---

*Source: experiments/_runs/existing_data_analysis/, experiments/_runs/geometric_analysis/*
