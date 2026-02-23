# Statistical Tests

Statistical validation and analysis utilities.

## Scripts

| Script | Purpose |
|--------|---------|
| `bootstrap_ci.py` | Bootstrap confidence intervals for effect sizes |
| `shuffle_test.py` | Permutation tests (1000 shuffles) to validate genuine signal |
| `variance_decomposition.py` | Decompose variance across factors |

## Key Patterns

- **Bootstrap**: scipy-style resampling with replacement, 2.5th/97.5th percentiles
- **Permutation**: Observed d must exceed null distribution (mean ~0.13) to be genuine
- **Length control**: All metrics residualized against text length before analysis

## Usage

```bash
python experiments/statistics/bootstrap_ci.py
python experiments/statistics/shuffle_test.py
```
