# Existing Data Analysis Report

Generated: 2026-03-12T14:47:21.164492

## Dataset Summary

- Total samples: 163
- Metrics: 9
- Domains: 4

### Samples per Domain

| Domain | Count |
|--------|-------|
| grammar | 50 |
| commonsense | 50 |
| nli | 50 |
| paraphrase | 13 |

## Length Correlations (CONFOUND CHECK)

| Metric | Pearson r | Interpretation |
|--------|-----------|----------------|
| n_active | 0.983 | CONFOUNDED |
| n_edges | 0.960 | CONFOUNDED |
| mean_influence | -0.809 | CONFOUNDED |
| top_100_concentration | -0.627 | moderate |
| max_logit_prob | 0.566 | moderate |
| logit_entropy | -0.540 | moderate |
| mean_activation | -0.533 | moderate |
| max_activation | 0.521 | moderate |
| max_influence | 0.498 | moderate |

## Top Effect Sizes (Cohen's d)

| Comparison | Metric | Cohen's d | p-value |
|------------|--------|-----------|---------|
| grammar_vs_paraphrase | logit_entropy | 8.68 | 0.0000*** |
| grammar_vs_paraphrase | n_active | -7.32 | 0.0000*** |
| commonsense_vs_paraphrase | logit_entropy | 7.26 | 0.0000*** |
| grammar_vs_paraphrase | n_edges | -7.08 | 0.0000*** |
| grammar_vs_paraphrase | max_logit_prob | -6.20 | 0.0000*** |
| commonsense_vs_paraphrase | max_logit_prob | -5.40 | 0.0000*** |
| commonsense_vs_paraphrase | n_active | -4.51 | 0.0000*** |
| commonsense_vs_paraphrase | n_edges | -4.43 | 0.0000*** |
| commonsense_vs_paraphrase | mean_influence | 3.25 | 0.0000*** |
| grammar_vs_commonsense | n_active | -3.19 | 0.0000*** |
| grammar_vs_nli | n_active | -3.11 | 0.0000*** |
| grammar_vs_commonsense | n_edges | -2.85 | 0.0000*** |
| grammar_vs_paraphrase | max_influence | -2.76 | 0.0000*** |
| grammar_vs_nli | n_edges | -2.70 | 0.0000*** |
| grammar_vs_nli | mean_influence | 2.45 | 0.0000*** |
| grammar_vs_paraphrase | mean_influence | 2.31 | 0.0000*** |
| nli_vs_paraphrase | n_edges | -2.21 | 0.0000*** |
| grammar_vs_commonsense | mean_influence | 2.09 | 0.0000*** |
| grammar_vs_paraphrase | max_activation | -2.05 | 0.0000*** |
| nli_vs_paraphrase | n_active | -1.95 | 0.0000*** |

## Length-Residualized Effect Sizes (CLEAN SIGNAL)

| Comparison | Metric | Cohen's d (resid) | p-value |
|------------|--------|-------------------|---------|
| commonsense_vs_paraphrase | logit_entropy | 4.06 | 0.0000*** |
| commonsense_vs_paraphrase | mean_influence | -3.25 | 0.0000*** |
| commonsense_vs_paraphrase | top_100_concentration | -3.22 | 0.0000*** |
| commonsense_vs_paraphrase | max_logit_prob | -3.01 | 0.0000*** |
| grammar_vs_paraphrase | logit_entropy | 2.47 | 0.0000*** |
| nli_vs_paraphrase | n_edges | -2.27 | 0.0000*** |
| commonsense_vs_paraphrase | n_edges | -2.24 | 0.0000*** |
| grammar_vs_commonsense | logit_entropy | -2.04 | 0.0000*** |
| nli_vs_paraphrase | top_100_concentration | -1.92 | 0.0000*** |
| nli_vs_paraphrase | mean_activation | -1.78 | 0.0000*** |
| nli_vs_paraphrase | mean_influence | -1.68 | 0.0000*** |
| grammar_vs_commonsense | max_logit_prob | 1.65 | 0.0000*** |
| grammar_vs_commonsense | n_edges | 1.55 | 0.0000*** |
| grammar_vs_nli | n_edges | 1.52 | 0.0000*** |
| grammar_vs_paraphrase | n_edges | -1.35 | 0.0001*** |

## Metric Correlations (|r| > 0.5)

| Metric 1 | Metric 2 | Pearson r |
|----------|----------|-----------|
| mean_influence | top_100_concentration | 0.795 |
| mean_influence | mean_activation | 0.706 |
| mean_influence | max_activation | -0.523 |
| mean_influence | n_active | -0.796 |
| mean_influence | n_edges | -0.704 |
| max_influence | max_activation | 0.799 |
| max_influence | n_active | 0.536 |
| max_influence | n_edges | 0.521 |
| top_100_concentration | mean_activation | 0.722 |
| top_100_concentration | n_active | -0.603 |
| top_100_concentration | n_edges | -0.528 |
| max_activation | n_active | 0.552 |
| max_activation | n_edges | 0.518 |
| logit_entropy | max_logit_prob | -0.966 |
| logit_entropy | n_active | -0.563 |
| logit_entropy | n_edges | -0.587 |
| max_logit_prob | n_active | 0.586 |
| max_logit_prob | n_edges | 0.601 |
| n_active | n_edges | 0.985 |

## Key Insights

- **Most discriminative metric:** logit_entropy
- **Most length-confounded:** n_active (r=0.98)