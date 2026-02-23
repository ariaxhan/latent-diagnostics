> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.

# Open Research Questions

**Source:** LOG entries from physics framework analysis
**Status:** Tracking hypotheses and investigation paths

---

## From Wannier-Stark Framework

### Δ: Prompt injection can be modeled as a phase transition

**Assumption:** The transition from "model follows system prompt" to "model follows injection" is a sharp phase boundary, not gradual degradation.

**Test:**
- Plot (P, E, I) witnesses across many injection attempts
- Look for: sharp boundary vs gradual transition
- If sharp → phase transition model is valid
- If gradual → need continuous risk score instead

**Priority:** HIGH — Core hypothesis

---

## From GOE-S-Matrix Framework

### Δ: AI input/output can be modeled as a scattering event

**Assumption:** The prompt→output relationship follows scattering matrix mathematics.

**Test:**
- Compute S-matrix elements for different prompt types
- Check if unitarity holds (σ_total = Σ σ_partial)
- Verify cross sections are well-defined and stable

**Priority:** MEDIUM — Mathematical elegance, not strictly necessary

### →: How do "cross sections" map to specific cyber attacks?

**Documented:** See `attack-cross-sections.md`

Five attack types mapped:
1. σ_exfil — Data exfiltration
2. σ_override — Instruction override
3. σ_hijack — Context hijacking
4. σ_tool — Tool manipulation
5. σ_persona — Persona hijacking

---

## From Hyperon Puzzle Framework

### Δ: Prompt injection acts as localized pressure on rule sets

**Assumption:** Injection attempts apply "pressure" that can be resisted by collective defense.

**Test:**
- Define "pressure" metric: injection strength/complexity
- Define "deformation" metric: change in survival probability
- Compute stiffness κ = ΔP / Δσ
- Compare stiffness for single-layer vs multi-layer defense

**Priority:** HIGH — Validates layered defense approach

### →: Does "stiffening" specific layers affect model speed?

**Investigation path:**
1. Profile latency for single-layer witness computation
2. Profile latency for all-layer witness computation
3. Identify which layers are most diagnostic (can we skip non-informative layers?)
4. Measure accuracy vs latency tradeoff

**Expected finding:** Early layers (syntactic) may be less informative than late layers (semantic). Could skip layers 1-4 and focus on layers 5+.

**Priority:** MEDIUM — Performance optimization, not correctness

---

## From Knot Theory Framework

### Δ: Prompt logic can be mapped to topological structures

**Assumption:** The logical structure of prompts has topological properties (loops, crossings, connectivity) that are invariant under surface-level paraphrasing.

**Test:**
- Apply TDA (Topological Data Analysis) to SAE feature clouds
- Compute persistent homology / Betti numbers
- Check if safe vs malicious prompts have different topological signatures
- Check if paraphrased versions of same attack have same topology

**Priority:** MEDIUM — Novel, high-risk/high-reward

### →: Explore "Topological Data Analysis" for real-time firewalling

**Investigation path:**
1. Select TDA library (giotto-tda, ripser, gudhi)
2. Compute persistence diagrams for SAE activation patterns
3. Define topological features: β₀ (connected components), β₁ (loops), β₂ (voids)
4. Train classifier on topological features
5. Benchmark against geometric features (c/a, entropy, etc.)

**Expected finding:** TDA may capture structure that geometry misses, especially for "nested" or "recursive" injection patterns.

**Priority:** LOW (exploratory) — Try after core framework validated

---

## Cross-Framework Questions

### Q1: Do the four frameworks give consistent predictions?

**Test:**
- Run same injection attempt through all four analyses
- Check if cross section predicts same outcome as witnesses
- Check if stiffness correlates with survival probability
- Check if topological signature correlates with phase classification

**Expected:** Frameworks should be complementary, not contradictory

### Q2: Which framework is most predictive?

**Test:**
- Compute detection accuracy for each framework independently
- Compute detection accuracy for combined framework
- Identify which framework contributes most to final accuracy

**Expected:** Combined > any single framework

### Q3: What's the minimum viable implementation?

If we can only implement one thing first, which has highest ROI?

**Candidates:**
1. Survival probability P(t) only — simplest witness
2. Cross section screening only — pre-generation block
3. Shape classification only — geometric signature

**Recommendation:** Start with survival probability. Requires only cosine similarity of SAE features. Fast, interpretable, testable.

---

## Empirical Questions (Require Data)

### E1: What are the actual threshold values?

Current thresholds (0.8, 0.4, 0.6) are guesses. Need calibration on IPI-1000.

### E2: How fast does the correlation hole appear?

Does P(t) dip before or after the exfiltration token? Critical for real-time blocking.

### E3: Do signatures transfer across models?

Test on Gemma-2-2B first, then Llama, Mistral, etc.

### E4: What's the false positive rate?

Legitimate complex prompts (code, roleplay, multi-step reasoning) may trigger false alarms.

### E5: Can attackers evade the detectors?

Adversarial robustness: can someone craft prompts that appear "localized" but still inject?

---

## Next Actions

| Question | Method | Priority | Status |
|----------|--------|----------|--------|
| Phase transition hypothesis | Plot witness distributions | HIGH | Not started |
| Stiffness effect | Multi-layer vs single-layer comparison | HIGH | Not started |
| Topological signatures | TDA exploration | MEDIUM | Not started |
| Layer speed impact | Profiling | MEDIUM | Not started |
| Threshold calibration | IPI-1000 benchmark | HIGH | Blocked (need dataset) |
| Transfer across models | Multi-model testing | LOW | Not started |
