# Archived Research

This directory contains research documents that were **not empirically validated** or represent **speculative directions** that were never implemented.

## Why These Are Archived

The documents here describe an ambitious theoretical framework called "Neural Polygraph" for prompt injection detection. While intellectually interesting, the core claims were never tested:

1. **Physics analogies** (Wannier-Stark, GOE-S-Matrix, etc.) were never validated on real data
2. **Three Witnesses** (P, E, I) were never implemented or tested
3. **Cross section tables** were never computed
4. **Layered defense architecture** was never built

When we finally ran experiments, we discovered fundamental issues:
- **Length confounding**: r=0.96 correlation between feature count and text length
- **Weak signal**: Best metric (mean_activation) achieved AUC=0.830, not competitive with 95%+ baselines
- **Small samples**: Only 21 injection samples in initial tests

## What Actually Works

For validated work, see:
- `experiments/` - Actual metric extraction code
- `notebooks/` - Analysis that confronts the data honestly

## Archive Structure

```
_archived/
  injection-detection-theory/   # "Neural Polygraph" speculative framework
    README.md                   # System overview (never built)
    physics-foundations.md      # Physics analogies (never tested)
    system-design.md            # Architecture (never implemented)
    enforcement-evolution.md    # STL rules + evolution (never built)
    witness-checklist.md        # Detection checklist (never calibrated)
    open-questions.md           # Research questions (never answered)
    2026-02-20-theory-benchmarks.md  # Literature review

  vector-native-experiments/    # Prompt notation experiments
    vector_native_empirical_validation.md  # VN density claims

  session-notes/                # Raw session notes
    session-2026-02-22.md       # Key session with length confound discovery
```

## Lessons Learned

1. **Theory without data is speculation** - The physics analogies were elegant but untested
2. **Length confounds everything** - Always control for text length in activation analysis
3. **Small samples hide problems** - 21 injection samples gave false confidence
4. **Baselines matter** - 80% accuracy is not useful when baselines hit 95%

## Historical Value

These documents preserve the intellectual journey. They show:
- How research directions can seem promising but fail empirically
- The importance of null results
- Why honest reporting matters

The failure of the "Neural Polygraph" concept led to the more grounded approach in the current codebase: extracting interpretable metrics without overclaiming classification ability.
