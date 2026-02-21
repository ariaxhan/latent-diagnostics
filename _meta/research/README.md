# Neural Polygraph: Prompt Injection Detection System

**Project:** SAE-based injection detection with physics foundations
**Status:** Research → System transition
**Last Updated:** 2026-02-20

---

## Vision

A **three-layer defense system** for prompt injection:

```
EVOLUTION    → Self-improving defense (AlphaEvolve, regret minimization)
ENFORCEMENT  → Multi-agent STL rules with penalties
DETECTION    → SAE features, witnesses, cross sections
```

---

## Theoretical Foundations (Six Frameworks)

| # | Framework | Application |
|---|-----------|-------------|
| 1 | **Wannier-Stark Localization** | Phase transitions (localized → ergodic) |
| 2 | **GOE-S-Matrix** | Cross sections and probability tables |
| 3 | **Hyperon Many-Body Effects** | Layered defense through collective stiffness |
| 4 | **Knot Theory y-ification** | Topological signatures for disambiguation |
| 5 | **Multi-Agent STL** | Temporal logic rules with penalties |
| 6 | **AlphaEvolve/MARL** | Evolutionary defense, regret minimization |

---

## Document Index

| Document | Purpose |
|----------|---------|
| [`injection-detection-framework.md`](./injection-detection-framework.md) | Two-stage detection pipeline |
| [`physics-foundations.md`](./physics-foundations.md) | Frameworks 1-4 with AI mappings |
| [`enforcement-evolution.md`](./enforcement-evolution.md) | Frameworks 5-6: STL + AlphaEvolve |
| [`attack-cross-sections.md`](./attack-cross-sections.md) | Cross section mapping for 5 attack types |
| [`witness-checklist.md`](./witness-checklist.md) | Operational decision tree |
| [`open-questions.md`](./open-questions.md) | Research hypotheses |
| [`2026-02-20-theory-benchmarks.md`](./2026-02-20-theory-benchmarks.md) | Physics papers, transcoders, PI benchmarks |

---

## Handoffs

| Document | Purpose |
|----------|---------|
| [`../handoffs/2026-02-20-beat-lakera.md`](../handoffs/2026-02-20-beat-lakera.md) | Session handoff: Beat Lakera on PINT |

---

## Quick Reference

### The Three Witnesses
```
P(t) = Survival Probability (system prompt preserved?)
E(t) = Entanglement (user input mixing with core logic?)
I(t) = Imbalance (model focus shifted to injection?)
```

### Phase Classification
```
Localized: P > 0.8, E < 0.2, I < 0.2  → Safe
Critical:  (in between)               → Warning
Ergodic:   P < 0.4 or E > 0.6 or I > 0.6  → Compromised
```

### Two-Stage Architecture
```
Stage 1: Pre-generation screening (cross section lookup)
Stage 2: Runtime monitoring (witness computation)
```

---

## Implementation Status

### Layer 1: Detection
| Component | Status | Location |
|-----------|--------|----------|
| SAE feature extraction | ✅ Done | `src/hallucination_detector/sae_utils.py` |
| Geometric analysis | ✅ Done | `src/hallucination_detector/geometry.py` |
| Survival probability | ❌ Not implemented | `witness_metrics.py` |
| Three witnesses | ❌ Not implemented | `witness_metrics.py` |
| Cross section tables | ❌ Not implemented | `probability_tables.py` |
| IPI-1000 benchmark | ❌ Not created | `experiments/data/` |

### Layer 2: Enforcement
| Component | Status | Location |
|-----------|--------|----------|
| STL rule parser | ❌ Not implemented | `stl_checker.py` |
| Penalty functions | ❌ Not implemented | `penalty.py` |
| Security Agent | ❌ Not designed | — |
| Multi-agent coordinator | ❌ Not implemented | — |

### Layer 3: Evolution
| Component | Status | Location |
|-----------|--------|----------|
| Defense evolver | ❌ Not implemented | `evolve.py` |
| Regret tracker | ❌ Not implemented | `regret.py` |
| Attack co-evolution | ❌ Not implemented | — |

---

## Key Hypotheses

1. Successful injection = localized→ergodic phase transition
2. Attack prompts cluster in high-σ regions (resonances)
3. Multi-layer defense > single-layer (stiffness)
4. Topological signatures distinguish similar-looking prompts

---

## Related Work

**Physics Sources:**
- arXiv:2602.01835 — GOE-S-Matrix probability tables
- arXiv:2602.07939 — Hyperon puzzle
- arXiv:2602.17435 — Knot theory y-ification

**AI Security Sources:**
- PromptArmor threat intelligence
- OWASP LLM Top 10
- CaMeL (Google)

**SAE/Mech Interp:**
- Anthropic SAE research
- Apple instruction-following dimension
