# Core Analyses

Validated analysis scripts that produce the main research findings.

## Scripts

| Script | Purpose | Key Finding |
|--------|---------|-------------|
| `domain_comparison.py` | Compare activation patterns across task types | d=1.08 (grammar vs others) |
| `truthfulness.py` | Test if activation patterns distinguish true/false | No signal (d=0.05) |
| `cognitive_regimes.py` | Analyze computational complexity signatures | d=0.87 |

## Key Insight

These analyses demonstrate that activation topology measures **how** a model computes (task type, complexity) rather than **whether** it computes correctly (truthfulness).

## Usage

```bash
python experiments/core/domain_comparison.py
python experiments/core/truthfulness.py
python experiments/core/cognitive_regimes.py
```

All scripts output to `figures/` and print statistical summaries.
