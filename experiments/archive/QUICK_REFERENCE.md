# Quick Reference: Experiment 6 Versions

## Which Version Should I Use?

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION TREE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Need maximum speed? ──YES──> Use PARALLEL (batch=8-16)    │
│         │                                                    │
│        NO                                                    │
│         │                                                    │
│  Have < 8 GB memory? ──YES──> Use SEQUENTIAL               │
│         │                                                    │
│        NO                                                    │
│         │                                                    │
│  First time running? ──YES──> Use SEQUENTIAL               │
│         │                                                    │
│        NO ──────────────────> Use PARALLEL (batch=8)        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Command Cheat Sheet

### Sequential Version (Safe, Tested)
```bash
# Standard run
python experiments/06_comprehensive_analysis.py

# Expected time: ~30 minutes
# Memory: ~8 GB
# Speedup: 3x vs separate experiments
```

### Parallel Version (Fast, Optimized)
```bash
# Default (batch size 8)
python experiments/06_comprehensive_analysis_parallel.py

# Custom batch size
python experiments/06_comprehensive_analysis_parallel.py --batch-size 16

# Expected time: ~10-15 minutes
# Memory: ~10-13 GB
# Speedup: 6-9x vs separate experiments
```

### Benchmark (Find Best Settings)
```bash
# Quick test (20 samples)
python experiments/benchmark_versions.py --quick

# Full test (all samples)
python experiments/benchmark_versions.py --full

# Test specific batch sizes
python experiments/benchmark_versions.py --batch-sizes 4 8 16
```

### Visualization (Both Versions Use Same Script!)
```bash
# Generate dashboard (works for BOTH sequential and parallel)
python experiments/visualize_comprehensive.py

# Output: 8-panel comprehensive dashboard
# Note: Both versions save to the same directory with identical format
```

## Batch Size Guide

| Your Hardware | Recommended Batch Size | Command |
|---------------|------------------------|---------|
| **Mac (8 GB)** | 8 | `--batch-size 8` |
| **Mac (16 GB)** | 16 | `--batch-size 16` |
| **PC/Linux (8 GB)** | 4-8 | `--batch-size 4` |
| **PC/Linux (16 GB)** | 8-16 | `--batch-size 8` |
| **GPU (CUDA)** | 16-32 | `--batch-size 16` |

## Performance Expectations

| Version | Time | Memory | Speedup |
|---------|------|--------|---------|
| **Sequential** | 30 min | 8 GB | 3x |
| **Parallel (batch=4)** | 18 min | 9 GB | 5x |
| **Parallel (batch=8)** | 12 min | 10 GB | 7x |
| **Parallel (batch=16)** | 10 min | 13 GB | 9x |

## Troubleshooting

### Problem: Out of Memory
```bash
# Solution: Reduce batch size
python experiments/06_comprehensive_analysis_parallel.py --batch-size 4
```

### Problem: Too Slow (< 5 samples/sec)
```bash
# Solution 1: Increase batch size
python experiments/06_comprehensive_analysis_parallel.py --batch-size 16

# Solution 2: Check if GPU is being used
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Problem: Results Look Different
```bash
# Solution: Use sequential version for consistency
python experiments/06_comprehensive_analysis.py
```

## Output Files

✅ **Both versions produce IDENTICAL output structure** (same directory, same format):

```
06_comprehensive_analysis/
└── runs/
    └── YYYYMMDD_HHMMSS/
        ├── manifest.json       # Experiment metadata
        ├── metrics.parquet     # All 33 metrics (same column names)
        └── figures/            # (after visualization)
            └── comprehensive_dashboard.png
```

**Key Point**: Use the same `visualize_comprehensive.py` for both versions!

## Metrics Collected

**33 metrics per sample**:
- 18 layer sensitivity (6 per layer × 3 layers)
- 6 semantic misalignment
- 3 stability (NEW)
- 3 entropy (NEW)
- 3 cross-layer consistency (NEW)

## Quick Comparison

| Feature | Sequential | Parallel |
|---------|-----------|----------|
| **Speed** | Medium | Fast |
| **Memory** | Low | Medium |
| **Complexity** | Simple | Simple |
| **Stability** | High | High |
| **Batch Size** | N/A | Configurable |
| **Real-time Stats** | No | Yes |
| **Recommended For** | First run, Low memory | Production, Speed |

## Example Workflow

### First Time User
```bash
# 1. Run sequential version (safe)
python experiments/06_comprehensive_analysis.py

# 2. Visualize results
python experiments/visualize_comprehensive.py

# 3. If satisfied, try parallel for speed
python experiments/06_comprehensive_analysis_parallel.py --batch-size 8
```

### Power User
```bash
# 1. Benchmark to find optimal settings
python experiments/benchmark_versions.py --quick

# 2. Use recommended settings
python experiments/06_comprehensive_analysis_parallel.py --batch-size <optimal>

# 3. Visualize
python experiments/visualize_comprehensive.py
```

### Production Use
```bash
# Use parallel with tested batch size
python experiments/06_comprehensive_analysis_parallel.py --batch-size 8

# Monitor progress in real-time
# Adjust batch size based on performance
```

## Performance Monitoring

### Sequential Version
```
Progress: 10/200 samples processed...
Progress: 20/200 samples processed...
```

### Parallel Version
```
Progress: 80/200 samples | Rate: 8.5 samples/sec | ETA: 14s
```

**Good Performance**: Rate > 8 samples/sec
**Okay Performance**: Rate 5-8 samples/sec
**Slow Performance**: Rate < 5 samples/sec (adjust batch size)

## Common Questions

**Q: Which version is more accurate?**
A: Both produce identical results (within numerical precision)

**Q: Can I run both versions?**
A: Yes! They save to the same directory structure

**Q: How do I know if parallel is working?**
A: Look for "Rate: X samples/sec" in output

**Q: What if I get an error?**
A: Try sequential version first, then parallel with smaller batch size

**Q: Can I stop and resume?**
A: No, but experiments are fast enough to rerun

## Best Practices

1. **Start with sequential** on first run
2. **Benchmark** to find optimal batch size
3. **Monitor** real-time performance
4. **Adjust** batch size based on memory/speed
5. **Visualize** results to verify correctness

## Getting Help

- Check `OPTIMIZATION_GUIDE.md` for detailed performance analysis
- Check `PARALLEL_OPTIMIZATION_SUMMARY.md` for complete documentation
- Check `EXP_4_5_VS_6_COMPARISON.md` for version comparison
- Run `benchmark_versions.py` to test your hardware

## Summary

**Recommended for most users**:
```bash
python experiments/06_comprehensive_analysis_parallel.py --batch-size 8
```

**Expected results**:
- Time: ~10-15 minutes
- Memory: ~10 GB
- Speedup: 6-9x vs baseline
- Output: 33 metrics per sample
- Visualization: 8-panel dashboard

**That's it! You're ready to run the optimized experiment.**

