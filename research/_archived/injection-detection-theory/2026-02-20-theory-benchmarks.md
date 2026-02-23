> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.

# Research Notes: Theory Foundations + Benchmark Survey

**Date:** 2026-02-20
**Topic:** Physics analogies, transcoder comparison, prompt injection benchmarks
**Status:** Active research direction

---

## Part 1: Physics Paper Analysis

### Paper A: Liouvillian Exceptional Points (arXiv:2602.01375)

**Authors:** Molina (IEM-CSIC)
**Core finding:** Non-Hermitian degeneracies in Lindblad generators create super-Lorentzian spectral signatures that are **state-dependent** in visibility.

**Key insight for Neural Polygraph:**
- Exceptional points exist in the dynamical generator but **steady-state emission can't see them**
- You need generic/infinite-temperature initial states to reveal the signature
- Implication: Hallucination/injection features might be **invisible in certain activation regimes**

**Mapping:**
| Physics | Neural Polygraph |
|---------|------------------|
| Liouvillian resolvent | SAE decoder weights |
| Jordan block (defective mode) | Ghost feature (hallucination-only) |
| Super-Lorentzian = 2nd-order pole | Reconstruction error spike |
| State-dependent visibility | Prompt context affects detectability |

**Actionable:** Vary "initial state" of prompts — cold start vs primed context. Detection accuracy may depend on preparation state.

**BIC model selection:** Paper uses Bayesian Information Criterion to distinguish Lorentzian (normal) from super-Lorentzian (exceptional). We could use similar model selection for feature activation distributions.

---

### Paper B: genriesz / Automatic Debiased ML (arXiv:2602.17543)

**Authors:** Kato (Mizuho-DL)
**Core finding:** Unified framework for Riesz representer estimation using Bregman divergences. Automatic regressor balancing (ARB) constructs link functions for moment-matching optimality.

**Key insight for Neural Polygraph:**
- Ghost feature detection is implicitly doing **causal estimation**: which SAE features *cause* the difference between fact and hallucination?
- genriesz provides machinery for this with valid inference (confidence intervals, p-values)

**Mapping:**
| genriesz | Neural Polygraph |
|----------|------------------|
| m(W, γ) = γ(hall) - γ(fact) | Treatment effect on activations |
| Riesz representer α₀ | Feature weights for causal signal |
| Bregman divergence | Loss function for feature selection |
| Entropy balancing | Sparse ghost feature selection |

**Relevance:** Medium. Provides statistical rigor for causal feature attribution. Could improve ghost feature selection with valid inference.

---

### Paper C: Dynamical Trap Spaces (arXiv:2602.15691)

**Authors:** Pastva, Park, Rozum, Trinh, Albert
**Core finding:** Phenotypes = complete trap spaces (partial steady states committed to specific biomarker values). Efficient identification without full attractor enumeration.

**Key insight for Neural Polygraph — HIGHLY RELEVANT:**

This paper provides **exactly the formal framework** Neural Polygraph needs:
- **Dynamical phenotype** = minimally constrained state committed to specific outcome
- **Phenotype-determining nodes (PDNs)** = minimal biomarker set that characterizes state
- **LDOI (Logical Domain of Influence)** = which nodes causally determine others

**Mapping:**
| Boolean Network | Neural Polygraph |
|-----------------|------------------|
| Dynamical phenotype | Hallucination/injection signature |
| PDN set | Ghost features |
| Trap space | Activation pattern model gets "stuck" in |
| LDOI | Feature cascade (which features determine which) |
| Input-phenotype map | Prompt → activation pattern |
| Coverage metric | How well PDNs determine network state |

**Actionable experiments:**

1. **SAE Feature Influence Graph**
   - Compute which features causally affect which (LDOI analog)
   - High out-degree features = better hallucination predictors?

2. **Coverage-Based Ghost Feature Selection**
   - For each candidate ghost feature, compute coverage (how many other features it determines)
   - Greedily select to maximize total coverage
   - Compare vs differential activation method

3. **Trap Space Identification**
   - Find self-reinforcing activation patterns
   - Hallucinations = falling into specific trap spaces
   - Explains why hallucinations are *stable* once entered

---

## Part 2: Transcoder vs SAE Comparison

### Key Paper: "Transcoders Beat Sparse Autoencoders" (arXiv:2501.18823)

**Authors:** Paulo, Shabalin, Belrose (Jan 2025)

**Core finding:** Transcoder features are **significantly more interpretable** than SAE features.

**Architecture comparison:**
| Aspect | SAE | Transcoder | Skip Transcoder |
|--------|-----|------------|-----------------|
| Input | Layer activations | Component input | Component input |
| Output | Reconstruct same activations | Reconstruct component output | + affine skip |
| Interpretability | Baseline | Higher | Highest |
| Reconstruction loss | Baseline | Similar | Lower |

**Skip transcoder:** Adds affine skip connection. Lower loss, no interpretability penalty.

**Recommendation:** "Interpretability researchers should shift focus away from SAEs trained on MLP outputs toward (skip) transcoders."

### Contrasting View: DeepMind Safety Research (March 2025)

**Finding:** "SAEs and SAE-based techniques (transcoders, crosscoders) are not likely to be a gamechanger any time soon."

**Concern:** Lack of compelling scenarios where they beat baselines.

**Nuance (June 2025 position paper):**
- SAEs good for **discovering unknown concepts**
- SAEs less effective for **acting on known concepts**

### Crosscoders (Anthropic, 2024)

- Read/write to **multiple layers**
- Cross-layer transcoders (CLT) in circuit tracing
- Each feature reads residual stream at one layer, contributes to all subsequent MLP layers
- ~50% output matching when substituting CLT features

### Recommendation for Neural Polygraph

**Test both architectures:**
1. Train SAE on target layer (existing approach)
2. Train transcoder MLP_in → MLP_out
3. Train skip transcoder (same + affine skip)
4. Compare: feature stability, witness correlation, detection accuracy

**Hypothesis:** Transcoders may give cleaner injection/hallucination signatures because they model the *transformation* rather than the *state*.

---

## Part 3: Prompt Injection Benchmarks

### Tier 1: Production-Ready Benchmarks

#### PINT Benchmark (Lakera)
- **URL:** https://github.com/lakeraai/pint-benchmark
- **Size:** 4,314 inputs (3,016 English, 1,298 non-English)
- **Categories:**
  - Prompt injections: 5.2%
  - Jailbreaks: 0.9%
  - Hard negatives (deceptive benign): 20.9%
  - Chat interactions: 36.5%
  - Public documents: 36.5%
- **Languages:** 14+ (French, German, Russian, Chinese, Japanese, Korean, Arabic, etc.)
- **Key feature:** Proprietary data ensures systems aren't trained on it
- **Leaderboard (May 2025):**
  - Lakera Guard: 95.22%
  - AWS Bedrock: 89.24%
  - Azure AI Prompt Shield: 89.12%
  - ProtectAI DeBERTa v3: 79.14%
  - Llama Prompt Guard 2: 78.76%

#### Open-Prompt-Injection
- **URL:** https://github.com/liu00222/Open-Prompt-Injection
- **Type:** Open-source toolkit
- **Attack types:** Combined attacks, injection into various tasks
- **Metrics:** ASV (Attack Success Value)
- **Defense:** DataSentinelDetector, PromptLocate

### Tier 2: Domain-Specific Benchmarks

#### RAG Benchmark (Nov 2025)
- **Size:** 847 adversarial test cases
- **Categories:**
  1. Direct injection
  2. Context manipulation
  3. Instruction override
  4. Data exfiltration
  5. Cross-context contamination

#### LLMail-Inject Challenge (Dec 2024 - Feb 2025)
- **Submissions:** 370,724 total
- **Teams:** 292 (621 participants)
- **Success rate:** Only 0.8% of submissions resulted in end-to-end attacks
- **Unique prompts:** 208,095
- **Auto-labeled injections:** 29,011 (triggered send_email API)

### Tier 3: Research Datasets

| Dataset | Date | Details |
|---------|------|---------|
| PI_HackAPrompt_SQuAD | Dec 2025 | Competition attacks + SQuAD benign |
| Privacy-preserving | Nov 2025 | Synthetic, federated splits |
| qualifire/prompt-injections | HuggingFace | Community benchmark |
| Kaggle evaluation framework | 2025 | Prompt injection + benign |

### Evaluation Metrics (2025 Standard)

| Metric | Definition |
|--------|------------|
| **ASR** | Attack Success Rate |
| **TPR** | True Positive Rate |
| **RDI** | Resilience Degradation Index |
| **SCC** | Safety Compliance Coefficient |
| **IIM** | Instructional Integrity Metric |
| **URS** | Unified Resilience Score |

### Known Limitations

1. **Static datasets become outdated** as defenders train against known attacks
2. **English-dominant** (though PINT addresses this)
3. **Handcrafted attacks** don't capture adaptive adversaries
4. **LLM-based scoring** may have same vulnerabilities

---

## Part 4: New Research Direction

### Proposed: Transcoder-Based Prompt Injection Detection

**Thesis:** Transcoders model the *transformation* (MLP_in → MLP_out) rather than the *state* (activations). This may give cleaner signatures for detecting when the model's processing is being hijacked.

**Why transcoders for injection:**
1. Injection = model processes input differently than intended
2. Transcoders capture *how* processing happens, not just *what* state exists
3. May reveal "computation hijacking" that SAEs miss

**Why injection instead of hallucination:**
1. More adversarial = harder problem = more value
2. Clear ground truth (attack succeeded or not)
3. Standardized benchmarks exist (vs HB-1000 which we created)
4. Security applications have immediate value

### Proposed Experiments

**Exp 1: Architecture Comparison on PINT**
- Train SAE, transcoder, skip transcoder on same model/layer
- Extract features for PINT benchmark samples
- Compare: interpretability, separation between inject/benign, detection accuracy

**Exp 2: Trap Space Analysis for Injection**
- Apply dynamical phenotype framework (Paper C)
- Identify "injection trap spaces" = activation patterns that lead to compliance
- Test if injection = falling into specific trap space

**Exp 3: State-Dependent Visibility (Paper A)**
- Vary prompt context (system prompt length, complexity)
- Test if injection signatures are more/less visible under different preparations
- Hypothesis: Complex system prompts may "hide" injection signature (like steady-state emission)

**Exp 4: Causal Feature Attribution (Paper B)**
- Apply Riesz regression to identify causal injection features
- Compare: entropy balancing vs differential activation
- Get confidence intervals on ghost feature importance

### Implementation Priority

1. **Integrate PINT benchmark** into experiments/data/
2. **Train transcoder** alongside existing SAE
3. **Implement LDOI-style coverage metric** for feature selection
4. **Adapt three witnesses** for injection (survival probability of system prompt features)

---

## References

### Physics Papers
- arXiv:2602.01375 — Liouvillian exceptional points (Molina)
- arXiv:2602.17543 — genriesz / Riesz regression (Kato)
- arXiv:2602.15691 — Dynamical trap spaces (Pastva et al.)

### Transcoder Research
- arXiv:2501.18823 — Transcoders beat SAEs (Paulo et al.)
- Anthropic Crosscoders — https://transformer-circuits.pub/2024/crosscoders/
- Anthropic Circuit Tracing — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- DeepMind negative results — https://deepmindsafetyresearch.medium.com/

### Benchmarks
- PINT — https://github.com/lakeraai/pint-benchmark
- Open-Prompt-Injection — https://github.com/liu00222/Open-Prompt-Injection
- HuggingFace qualifire — https://huggingface.co/datasets/qualifire/prompt-injections-benchmark
- arXiv:2511.15759 — Securing AI Agents framework
- arXiv:2506.09956 — LLMail-Inject dataset

---

## Next Actions

1. [ ] Download PINT benchmark, integrate into experiments/data/
2. [ ] Implement skip transcoder training pipeline
3. [ ] Add LDOI coverage metric to feature selection
4. [ ] Adapt witness_metrics.py for injection detection
5. [ ] Run architecture comparison (SAE vs transcoder) on PINT
