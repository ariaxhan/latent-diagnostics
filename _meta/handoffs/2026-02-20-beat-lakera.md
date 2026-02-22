# CONTEXT HANDOFF: Beat Lakera on PINT Benchmark

**Date:** 2026-02-20
**Goal:** Build prompt injection detector that beats Lakera Guard's 95.22% on PINT

---

## Summary

Building a prompt injection detector using SAE/transcoder features and probability tables. Target: beat Lakera Guard (95.22%) on PINT benchmark. Have extensive theoretical framework (6 physics analogies) but teardown revealed only probability tables + cosine similarity matter. Pivot to transcoders (more interpretable per recent research) and test on standardized benchmarks.

---

## Goal

**Beat Lakera Guard's 95.22% F1 on PINT benchmark** using SAE/transcoder feature analysis.

---

## Current State

### What Exists
| Component | Status | Location |
|-----------|--------|----------|
| SAE feature extraction | Done | `src/hallucination_detector/sae_utils.py` |
| Geometric analysis | Done | `src/hallucination_detector/geometry.py` |
| HB-1000 hallucination benchmark | Done | `experiments/data/bench_*.json` |
| 6 physics frameworks (theory) | Done | `_meta/research/*.md` |
| Teardown (cut the fat) | Done | `_meta/reviews/injection-detection-teardown.md` |
| Benchmark survey (PINT, Open-PI) | Done | `_meta/research/2026-02-20-theory-benchmarks.md` |

### What's Missing
| Component | Priority | Effort |
|-----------|----------|--------|
| PINT benchmark integration | HIGH | 2 hours |
| Transcoder training/loading | HIGH | 4 hours |
| Probability table builder | HIGH | 4 hours |
| Cosine similarity witness | LOW | 30 min |
| InjectionDetector class | MEDIUM | 2 hours |

---

## Decisions Made

1. **Pivot from SAE to transcoder** — Paulo et al. (arXiv:2501.18823) showed transcoders are more interpretable. Test both.

2. **Use PINT benchmark** — 4,314 inputs, multilingual, proprietary data, active leaderboard. Gold standard.

3. **Cut physics metaphor overhead** — Teardown identified: 70% metaphor, 30% actionable. Keep probability tables + cosine similarity. Cut STL, evolution, TDA.

4. **Core algorithm is simple** — Cluster attacks in feature space, pre-compute success rates, lookup at runtime. ~50 lines of useful code.

---

## Artifacts Created

```
neural-polygraph/_meta/research/
├── 2026-02-20-theory-benchmarks.md   # NEW: Papers + benchmarks + transcoder research
├── injection-detection-framework.md  # Two-stage detection pipeline
├── physics-foundations.md            # 4 physics frameworks mapped
├── attack-cross-sections.md          # 5 attack types with signatures
├── open-questions.md                 # Research hypotheses
└── README.md                         # Updated with new doc

neural-polygraph/_meta/reviews/
└── injection-detection-teardown.md   # Critical review: what to cut

neural-polygraph/_meta/handoffs/
└── 2026-02-20-beat-lakera.md         # THIS FILE
```

---

## Open Threads

1. **Transcoder availability** — Are there pre-trained transcoders for Gemma-2-2B? Check SAE-Lens/TransformerLens.

2. **Core hypothesis unvalidated** — "Injection causes distinctive features" is assumed. Test with 10 injection + 10 benign prompts.

3. **Threshold calibration** — Current thresholds (0.8, 0.4, 0.6) are guesses. Need empirical calibration on PINT.

4. **DeepMind skepticism** — Their March 2025 post questions SAE value for downstream tasks. Need to prove them wrong or pivot.

---

## Next Steps

### Day 1: Validate Core Hypothesis
```bash
1. Download PINT benchmark
2. Run 10 injection + 10 benign through existing SAE pipeline
3. Compute cosine similarity baseline → features
4. Question: Does similarity discriminate injection vs benign?
5. If NO → STOP, rethink approach
6. If YES → Continue to Day 2
```

### Day 2: Build Probability Tables
```bash
1. Extract SAE features for full PINT dataset (or subset)
2. Cluster in feature space (k-means, k=100)
3. Compute injection success rate per cluster
4. Build lookup table: cluster_id → {risk_score, n_samples}
5. Identify resonance peaks (high-risk clusters)
```

### Day 3: Build Detector + Benchmark
```bash
1. Implement InjectionDetector class (~50 lines)
2. Run on PINT test set
3. Compute precision, recall, F1
4. Compare to Lakera (95.22%), AWS (89.24%), Azure (89.12%)
```

### Day 4: Transcoder Comparison
```bash
1. Check if transcoders available for Gemma-2-2B
2. If yes: Train or load transcoder
3. Repeat Days 1-3 with transcoder instead of SAE
4. Compare: which gives better discrimination?
```

---

## Context Essentials

### Lakera's Leaderboard (May 2025)
| System | PINT Score |
|--------|------------|
| Lakera Guard | 95.22% |
| AWS Bedrock | 89.24% |
| Azure AI Prompt Shield | 89.12% |
| ProtectAI DeBERTa | 79.14% |
| Llama Prompt Guard 2 | 78.76% |

### Transcoder Key Finding
From arXiv:2501.18823: "Transcoder features are **significantly more interpretable** than SAE features. Skip transcoders achieve lower reconstruction loss with no effect on interpretability."

### The Core Algorithm (from teardown)
```python
class InjectionDetector:
    def analyze(self, prompt):
        features = self.extract_features(prompt)
        cluster = self.find_nearest_cluster(features)
        table_risk = self.table[cluster]["σ_injection"]
        similarity = cosine_similarity(self.baseline, features)
        risk = 0.6 * table_risk + 0.4 * (1 - similarity)
        return {"risk_score": risk, "recommendation": "BLOCK" if risk > 0.7 else "ALLOW"}
```

### Dynamical Phenotype Insight (from arXiv:2602.15691)
Use LDOI coverage metric for feature selection: "A good PDN set has high coverage — fixing those features determines the state of many other features." Apply to ghost feature selection.

---

## Warnings

1. **Don't over-engineer** — Teardown showed 70% of framework is overhead. Start with 50-line detector.

2. **Validate before building** — Core hypothesis (injection → distinctive features) is UNPROVEN. Test first.

3. **DeepMind is skeptical** — Their March 2025 post: "SAEs not likely to be a gamechanger." Need strong results.

4. **Benchmarks get stale** — PINT uses proprietary data to prevent gaming, but adversarial evolution continues.

---

## File Paths to Read

```
# Core theory (skim)
neural-polygraph/_meta/reviews/injection-detection-teardown.md    # CRITICAL: What to cut
neural-polygraph/_meta/research/2026-02-20-theory-benchmarks.md   # NEW: Benchmarks + transcoders

# Implementation
neural-polygraph/src/hallucination_detector/sae_utils.py         # Feature extraction
neural-polygraph/src/hallucination_detector/geometry.py          # Geometric analysis

# Existing experiments (for patterns)
neural-polygraph/experiments/01_spectroscopy.py                  # How experiments are structured
```

---

## Continuation Prompt for New Session

> We're building a prompt injection detector to beat Lakera Guard (95.22%) on the PINT benchmark. We have SAE feature extraction working for hallucinations; now pivoting to injection detection with transcoders. The teardown identified probability tables + cosine similarity as the only valuable components — everything else is metaphor overhead. Next: download PINT, validate that SAE features discriminate injection vs benign (10+10 test), then build probability tables. Check if transcoders exist for Gemma-2-2B.

---

## Quick Commands

```bash
# Check transcoder availability
python -c "from sae_lens import SAE; print(dir(SAE))"

# Run existing setup verification
cd neural-polygraph && python verify_setup.py

# Check experiment structure
ls -la experiments/

# View teardown (critical)
cat _meta/reviews/injection-detection-teardown.md | head -100
```
