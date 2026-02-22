# Neural Polygraph: Prompt Injection Detection Framework

**Status:** Research Design
**Date:** 2026-02-20
**Origin:** Synthesis of SAE spectroscopy + quantum ergodicity + nuclear scattering theory

---

## Core Thesis

Prompt injection can be modeled as a **phase transition** in the model's information dynamics. Successful injection corresponds to a transition from **localized** (system prompt dominates) to **ergodic** (instructions mixed, boundaries dissolved).

We propose a two-stage detection architecture grounded in physics:

1. **Pre-generation screening** — S-matrix / Cross section formalism
2. **Runtime monitoring** — Wannier-Stark localization witnesses

---

## Theoretical Foundations

### Framework 1: Wannier-Stark Localization (Runtime Detection)

**Source:** Quantum ergodicity in tilted lattice systems

**Core Insight:** A system under "tilt" (external perturbation) can either:
- **Localize**: Stay stuck in its original state despite perturbation
- **Thermalize**: Forget initial conditions, mix everything together

**Mapping to Prompt Injection:**

| Physics Concept | AI Security Equivalent |
|-----------------|----------------------|
| Initial localized state | System prompt activations |
| Tilt (external field) | User input / injected instructions |
| Localization | Model follows system instructions |
| Thermalization | Model follows injected instructions |
| Phase transition | The boundary where injection succeeds |

**Three Witnesses (Detection Metrics):**

1. **Survival Probability P(t)**
   - Definition: How much of the system prompt's SAE feature signature survives in current activations?
   - Formula: `P(t) = cosine_similarity(system_features, current_features)`
   - Signal: **Correlation hole** = sudden dip indicates injection taking effect

2. **Entanglement E(t)**
   - Definition: How much is user input mixing with core logic?
   - Formula: `E(t) = misalignment_angle / 90°` (normalized)
   - Signal: High entanglement = model losing independence from external input

3. **Imbalance I(t)**
   - Definition: Is the model's focus shifting entirely to injection?
   - Formula: `I(t) = (entropy_shift + dimensionality_shift) / 2`
   - Signal: High imbalance = "particle count" has moved to attacker's territory

**Phase Classification:**

```
if P > 0.8 and E < 0.2 and I < 0.2:
    phase = "Localized"      # Safe
elif P < 0.4 or E > 0.6 or I > 0.6:
    phase = "Ergodic"        # Compromised
else:
    phase = "Critical"       # Warning — near phase boundary
```

---

### Framework 2: S-Matrix / Cross Section (Pre-Generation Screening)

**Source:** Nuclear probability tables, random matrix theory (GOE)

**Core Insight:** Nuclear physicists pre-compute "probability tables" for how likely various reactions are. We can pre-compute "injection cross sections" for input classes.

**Mapping to Prompt Injection:**

| Nuclear Physics | AI Security Equivalent |
|-----------------|----------------------|
| Neutron (projectile) | User prompt |
| Target nucleus | Model + system prompt |
| Cross section σ | Probability of safety violation |
| Resonance peak | Vulnerable prompt patterns |
| S-matrix | Model's input→output mapping |
| Probability table | Pre-computed risk lookup |

**Key Concepts:**

1. **Injection Cross Section σ_inj(prompt)**
   - Definition: Probability that a given prompt class leads to injection compliance
   - Computed empirically from benchmark data
   - Used for pre-screening before generation

2. **Resonance Peaks**
   - Definition: Specific prompt patterns where injection success rate is anomalously high
   - Examples: "Ignore previous instructions", base64 encoded commands, markdown image injection
   - Detection: Cluster analysis in SAE feature space to find high-σ regions

3. **Probability Tables**
   - Pre-computed lookup: Given input SAE signature, return injection risk score
   - Indexed by: Feature cluster, entropy level, shape class
   - Enables fast runtime screening without full witness computation

4. **Noise Handling (GOE/RMT)**
   - Adversarial inputs are "noisy" (obfuscation, paraphrasing)
   - Random Matrix Theory provides tools for statistical fluctuations
   - Expected distribution of cross sections helps identify anomalies

---

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PROMPT                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: PRE-GENERATION SCREENING (S-Matrix)               │
│                                                             │
│  1. Extract SAE features from prompt                        │
│  2. Compute distance to known resonance peaks               │
│  3. Lookup cross section σ_inj from probability table       │
│  4. If σ_inj > threshold → BLOCK or FLAG                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (if σ_inj acceptable)
┌─────────────────────────────────────────────────────────────┐
│  GENERATION BEGINS                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: RUNTIME MONITORING (Wannier-Stark)                │
│                                                             │
│  For each generated token:                                  │
│    1. Compute three witnesses: P(t), E(t), I(t)             │
│    2. Classify phase: Localized / Critical / Ergodic        │
│    3. If phase == Critical → increase scrutiny              │
│    4. If phase == Ergodic → HALT generation                 │
│                                                             │
│  Detection target: "Correlation hole" in P(t)               │
│  (Sudden dip = injection taking effect)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT (if safe) or BLOCKED (if compromised)               │
└─────────────────────────────────────────────────────────────┘
```

---

## SAE Feature Space Geometry (From AIDA-TNG)

Complementary to the physics frameworks, we use geometric analysis of SAE activations:

**Shape Classification:**
- **Spherical** (b/a > 0.9, c/a > 0.9): Balanced, stable — likely safe
- **Prolate** (b/a < 0.6, c/a < 0.6): Elongated, unstable — hallucination/injection risk
- **Oblate** (b/a > 0.8, c/a < 0.6): Disk-like — intermediate
- **Triaxial**: Complex, potentially confused state

**Geometric Metrics:**
- `c/a ratio`: Sphericity (higher = more stable)
- `misalignment_angle`: Deviation between centroid and principal axis
- `eigenvalue_entropy`: How dispersed is activation energy?
- `dimensionality`: Effective degrees of freedom

**Hypothesis:** Injection compliance correlates with:
- Lower c/a ratio (more prolate/unstable)
- Higher misalignment angle (model pulled away from original direction)
- Higher entropy (more "confused" activation distribution)

---

## Implementation Roadmap

### Phase 1: Benchmark Creation
- [ ] Create IPI-1000 dataset (Indirect Prompt Injection pairs)
- [ ] Structure: (legitimate_response, injection_compliant_response)
- [ ] Categories: Data exfiltration, instruction override, context hijacking, tool manipulation
- [ ] Source: Adapt from PromptArmor, AgentDojo, existing red-team datasets

### Phase 2: Witness Implementation
- [ ] Add `survival_probability()` to geometry.py
- [ ] Add `compute_entanglement()` wrapper
- [ ] Add `compute_imbalance()` wrapper
- [ ] Add `classify_phase()` function
- [ ] Create `witness_metrics.py` module

### Phase 3: Cross Section Calibration
- [ ] Run HB-1000 (hallucination) + IPI-1000 (injection) through pipeline
- [ ] Compute witness values for each sample
- [ ] Find empirical thresholds for phase classification
- [ ] Build probability tables (feature signature → risk score)
- [ ] Identify resonance peaks (high-σ regions in feature space)

### Phase 4: SAE vs Transcoder Comparison
- [ ] Benchmark both architectures on IPI-1000
- [ ] Compare: feature stability, witness correlation, detection accuracy
- [ ] Decision: Which gives more reliable signatures?

### Phase 5: Real-Time Integration
- [ ] Implement token-by-token witness computation
- [ ] Profile latency overhead
- [ ] Optimize for production deployment
- [ ] Create API: `injection_risk_score(prompt, context) → float`

---

## Open Questions

1. **Threshold Calibration**: What are the actual boundaries for Localized/Critical/Ergodic?
2. **Temporal Dynamics**: How fast does the correlation hole appear? Can we detect before exfiltration token?
3. **Cross-Model Transfer**: Do signatures generalize across model families?
4. **Adversarial Robustness**: Can attackers craft inputs that appear localized but aren't?
5. **False Positive Rate**: How often do legitimate complex prompts trigger false alarms?

---

## Relationship to Existing Work

**What exists (and their limitations):**
- Input filtering (semantic evasion defeats it)
- LLM-as-judge (same vulnerability as primary model)
- Attention tracking (requires internal access, not interpretable)
- Embedding classifiers (no SAE decomposition)

**What's novel here:**
1. Theoretical grounding in physics (localization theory, scattering theory)
2. SAE feature-level detection (interpretable, potentially transferable)
3. Two-stage architecture (pre-screening + runtime monitoring)
4. Three independent witnesses (triangulation for confidence)
5. Phase transition framing (principled rather than empirical)

---

## References

**Physics Sources:**
- Wannier-Stark localization in tilted lattice systems (quantum ergodicity)
- arXiv:2602.01835 — Physics-based probability tables using random-matrix approach
- AIDA-TNG cosmology — Halo morphology analysis

**AI Security Sources:**
- PromptArmor threat intelligence (Writer.com, Claude Code exploits)
- OWASP LLM01:2025 Prompt Injection guidance
- Simon Willison on fundamental unsolvability
- CaMeL (Google) — Architectural defenses
- Attention Tracker — Attention-based detection

**SAE/Mech Interp Sources:**
- SAE Lens, TransformerLens libraries
- Anthropic SAE research on deception features
- Apple: "Do LLMs know internally when they follow instructions?"

---

## Next Actions

**Immediate (This Session):**
1. Implement witness_metrics.py with three witnesses
2. Add survival_probability to existing pipeline
3. Test on existing HB-1000 samples to validate metrics

**Short-term:**
1. Create IPI-1000 benchmark structure
2. Calibrate thresholds empirically
3. Build probability tables

**Medium-term:**
1. Transcoder comparison study
2. Real-time detection prototype
3. Paper draft
