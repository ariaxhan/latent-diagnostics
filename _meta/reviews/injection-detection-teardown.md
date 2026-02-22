# Tear Down: Neural Polygraph Injection Detection Framework

**Date:** 2026-02-20
**Reviewer:** Critical review mode
**Scope:** Full framework (6 physics analogies, 3-layer architecture)

---

## Executive Summary

**70% of this is metaphor overhead. 30% is actionable.**

The physics analogies are intellectually satisfying but most add no computational value. Strip them. What remains is:

1. **SAE feature comparison** (cosine similarity) — trivial but useful
2. **Probability tables** (pre-computed risk scores) — genuinely interesting
3. **Shape metrics** (c/a ratio, entropy) — might work, needs validation
4. **Multi-agent architecture** — standard, not novel

---

## Part 1: Metaphor vs Substance Mapping

### Framework 1: Wannier-Stark Localization

| Physics Term | What It Actually Means in AI | Is It Novel? |
|--------------|------------------------------|--------------|
| "Localized state" | Model follows system prompt | No — just "working correctly" |
| "Ergodic state" | Model follows injected instructions | No — just "injection succeeded" |
| "Phase transition" | Point where injection succeeds | No — just a threshold |
| "Tilt" | Adversarial pressure from prompt | No — just "attack strength" |
| "Thermalization" | Loss of instruction boundaries | No — just "compromised" |

**Verdict:** The metaphor adds zero computational value. The underlying idea is:

```python
# What Wannier-Stark actually means:
cosine_similarity(system_prompt_features, current_features) < threshold
```

That's it. You're checking if activations drifted from baseline. This is **not novel** — it's standard anomaly detection.

**Keep:** The cosine similarity check. It's useful.
**Discard:** All localization terminology.

---

### Framework 2: GOE-S-Matrix / Probability Tables

| Physics Term | What It Actually Means in AI | Is It Novel? |
|--------------|------------------------------|--------------|
| "Cross section σ" | P(injection succeeds \| input features) | **Yes — this framing is useful** |
| "Probability table" | Pre-computed lookup: features → risk | **Yes — enables fast inference** |
| "Resonance peak" | High-risk cluster in feature space | **Yes — identifies vulnerable patterns** |
| "S-matrix" | Input→output transition function | No — just "the model" |
| "Scattering event" | Processing a prompt | No — metaphor only |

**Verdict:** This has real value. The core ideas:

1. **Cluster attack prompts in SAE feature space**
2. **Pre-compute success rate per cluster**
3. **At runtime: lookup cluster → get risk score**

This is **fast** (lookup vs full computation) and **interpretable** (know which cluster triggered).

**Keep:** Probability tables, clustering, resonance detection.
**Discard:** S-matrix terminology, scattering metaphors.

**Concrete implementation:**
```python
# What probability tables actually mean:
def build_probability_table(attack_dataset):
    features = [extract_sae_features(p) for p in attack_dataset]
    labels = [p.injection_succeeded for p in attack_dataset]
    clusters = kmeans(features, n_clusters=100)

    table = {}
    for cluster_id in range(100):
        cluster_samples = labels[clusters == cluster_id]
        table[cluster_id] = {
            "σ_injection": mean(cluster_samples),
            "n_samples": len(cluster_samples),
            "confidence": 1 - std(cluster_samples)
        }
    return table

def get_risk(prompt, table):
    features = extract_sae_features(prompt)
    cluster = find_nearest_cluster(features)
    return table[cluster]["σ_injection"]
```

**This is the most valuable part of the framework.**

---

### Framework 3: Hyperon Puzzle / Many-Body Stiffness

| Physics Term | What It Actually Means in AI | Is It Novel? |
|--------------|------------------------------|--------------|
| "Hyperon" | Destabilizing input (injection) | No — just "attack" |
| "Soft material" | Vulnerable to attack | No — just "weak defenses" |
| "Stiffness" | Resistance to attack | No — just "robustness" |
| "Many-body effects" | Multiple components working together | No — just "defense in depth" |
| "Collective resistance" | Ensemble methods | No — standard ML |

**Verdict:** This is pure metaphor. The underlying idea:

> "Use multiple detection methods and combine them"

That's ensemble learning. Not novel.

**Keep:** The concept of multi-layer checking (but don't need physics terminology).
**Discard:** All hyperon/stiffness language.

---

### Framework 4: Knot Theory / y-ification

| Physics Term | What It Actually Means in AI | Is It Novel? |
|--------------|------------------------------|--------------|
| "Topological signature" | Structural pattern in data | Maybe — TDA is real |
| "y-ification" | Adding dimensions to increase discriminability | No — just feature engineering |
| "Knot invariants" | Features that survive transformation | Maybe — paraphrase invariance |

**Verdict:** The idea of using Topological Data Analysis (TDA) is real, but:
- TDA is slow
- TDA on high-dimensional SAE features is computationally expensive
- Unclear if it adds value over simpler geometric metrics

**Keep:** Investigate TDA as future work.
**Discard:** Knot theory terminology.

---

### Framework 5: Multi-Agent STL

| Term | What It Actually Means | Is It Novel? |
|------|------------------------|--------------|
| "Signal Temporal Logic" | Rules with time constraints | Real formalism, overkill for this |
| "Block-coordinate optimization" | Separate agents for separate tasks | Standard multi-agent design |
| "Penalty functions" | Loss term for violations | Standard ML |

**Verdict:** STL is a real specification language, but:
- Implementing an STL parser is significant overhead
- Most safety rules don't need temporal logic
- Simple boolean rules suffice for 90% of cases

**Keep:** The idea of separate Security Agent + Response Agent.
**Discard:** Full STL implementation (overkill).

**Simplify to:**
```python
# Instead of STL parser:
RULES = [
    lambda response, context: "system_prompt" not in response,
    lambda response, context: not (has_url(response) and has_user_data(response)),
    lambda response, context: not is_tool_call(response) or user_confirmed(context),
]

def check_rules(response, context):
    return all(rule(response, context) for rule in RULES)
```

---

### Framework 6: AlphaEvolve / MARL

| Term | What It Actually Means | Is It Novel? |
|------|------------------------|--------------|
| "Evolutionary defense" | Hyperparameter/threshold tuning over time | Standard AutoML |
| "Regret minimization" | Online learning algorithm | Real, but overkill |
| "VAD-CFR" | Specific algorithm for games | Not applicable |
| "Attack co-evolution" | Red-teaming | Standard security practice |

**Verdict:** The core idea is:
> "Tune your detector on new attack data over time"

That's online learning. Not novel. The game-theoretic framing is overkill.

**Keep:** The idea of periodic re-calibration.
**Discard:** Full evolutionary/regret framework.

---

## Part 2: What's Actually Novel and Useful?

### NOVEL (Worth pursuing)

| Idea | Why It's Novel | Effort | Payoff |
|------|----------------|--------|--------|
| **Pre-computed probability tables** | Fast lookup vs runtime computation | Medium | High |
| **SAE feature clustering for attacks** | Attack "fingerprints" in interpretable space | Medium | High |
| **Transcoders instead of SAEs** | Better interpretability, potentially more stable | Low (swap) | Unknown |
| **Geometric metrics (c/a, entropy)** | Shape of activation cloud as health indicator | Low | Medium |

### NOT NOVEL (Standard techniques with fancy names)

| Idea | What It Really Is | Status |
|------|-------------------|--------|
| "Survival probability" | Cosine similarity | Keep (useful) |
| "Entanglement" | Angular distance | Keep (useful) |
| "Imbalance" | Entropy change | Keep (useful) |
| "Phase classification" | Threshold-based decisions | Keep (useful) |
| "Multi-layer stiffness" | Ensemble methods | Skip (overkill for MVP) |
| "STL enforcement" | Rule checking | Simplify to boolean rules |
| "Evolutionary defense" | Periodic retraining | Skip (overkill for MVP) |

### OVERHEAD (Cut entirely)

| Idea | Why It's Overhead |
|------|-------------------|
| 6 physics frameworks | Conceptual debt, no computational value |
| Knot theory/TDA | Slow, unproven value |
| Regret minimization | Academic, overkill |
| Full STL parser | Overkill for simple rules |
| 8 research documents | Too much prose, not enough code |

---

## Part 3: Transcoder vs SAE Analysis

### Current SAE Approach
- Uses SAE-Lens with Gemma-Scope SAEs
- Layer 5, width 16k
- Known issue: SAE features may be unstable across training runs

### Transcoder Alternative
- Transcoders: predict next-layer activations from current-layer activations
- Recent research shows transcoders produce more interpretable features
- Potentially more stable than SAEs

### Recommendation
```python
# Make architecture swappable:
class FeatureExtractor(Protocol):
    def encode(self, activations: Tensor) -> Tensor: ...
    def decode(self, features: Tensor) -> Tensor: ...

class SAEExtractor:
    def __init__(self, sae):
        self.sae = sae
    def encode(self, activations):
        return self.sae.encode(activations)

class TranscoderExtractor:
    def __init__(self, transcoder):
        self.transcoder = transcoder
    def encode(self, activations):
        return self.transcoder.encode(activations)

# Swap without changing downstream code:
extractor = TranscoderExtractor(transcoder)  # or SAEExtractor(sae)
features = extractor.encode(activations)
```

### Transcoder availability
- Check: Does TransformerLens support transcoders?
- Check: Are there pre-trained transcoders for Gemma-2-2B?
- If not: SAE is fine for MVP, transcoder is future work

---

## Part 4: Performance Analysis

### Current Bottlenecks

| Operation | Estimated Time | Concern |
|-----------|----------------|---------|
| Load model (Gemma-2-2B) | 10-30s startup | Acceptable (once) |
| Load SAE | 5-10s startup | Acceptable (once) |
| Run model forward pass | 50-200ms | **Bottleneck** |
| SAE encode | 1-5ms | Fine |
| Geometric computation | <1ms | Fine |
| Probability table lookup | <1ms | **Fast** |

### The Problem
Every prompt requires a full model forward pass to get activations. This is 50-200ms per prompt.

### Solutions

1. **Probability tables eliminate model call for known patterns**
   - If prompt features cluster near known attack → skip model call, return risk
   - Only run model for uncertain cases
   - **This is the key optimization**

2. **Batch processing**
   - Process multiple prompts in one forward pass
   - Good for benchmarking, less relevant for real-time

3. **Smaller model**
   - Gemma-2-2B is relatively small
   - Could also test on Gemma-2B-IT (instruction-tuned)

4. **Feature caching**
   - Cache system prompt features (computed once per session)
   - Only compute user prompt features

### Realistic Performance Target

| Mode | Latency | How |
|------|---------|-----|
| Fast path (known attack) | <5ms | Probability table lookup |
| Full analysis | 50-200ms | Model forward + SAE + geometry |
| Cached system prompt | 30-100ms | Only user prompt forward pass |

---

## Part 5: Critical Issues

### Issue 1: No validation data

**Problem:** We have no IPI-1000 dataset. All thresholds are guesses.

**Impact:** Can't know if any of this works.

**Fix:** Create minimal benchmark first. 100 injection examples + 100 benign examples. Validate before building more.

### Issue 2: Untested core hypothesis

**Problem:** "Injection causes distinctive SAE feature patterns" is assumed, not proven.

**Impact:** Entire framework may be useless if hypothesis is false.

**Fix:** Run existing HB-1000 (hallucination) benchmark. Does geometry correlate with hallucination? If yes, promising. If no, rethink.

### Issue 3: Threshold instability

**Problem:** Thresholds (0.8, 0.4, 0.6) are arbitrary. May not transfer across:
- Different models
- Different prompt lengths
- Different domains

**Impact:** System may work in testing, fail in production.

**Fix:** Per-deployment calibration required. Document this limitation.

### Issue 4: Transcoder availability unknown

**Problem:** We don't know if transcoders exist for Gemma-2-2B.

**Fix:** Check before committing to transcoder path.

---

## Part 6: Questions That Need Answers

1. **Does injection actually show distinctive SAE features?**
   - Test: Run 10 injection prompts + 10 benign through current SAE pipeline
   - Expected: Cosine similarity should be lower for injection
   - If not: Framework is fundamentally flawed

2. **Are transcoders available for Gemma-2-2B?**
   - Check SAE-Lens, TransformerLens documentation
   - If not: Stick with SAEs

3. **What's the actual latency on your hardware?**
   - Run: Time the current `extract_features()` function
   - Result determines if optimization is needed

4. **What does the existing hallucination detection show?**
   - You have HB-1000 benchmark and working experiments
   - What were the results? Do geometric metrics correlate?

---

## Part 7: Recommended Minimal Path

### Cut List (Remove from scope)

| Cut | Reason |
|-----|--------|
| All 6 physics framework names | Metaphor overhead |
| STL parser | Overkill |
| Evolution layer | Overkill for MVP |
| Regret minimization | Academic |
| TDA/Knot theory | Slow, unproven |
| 8 research documents | Too much prose |

### Keep List (Actually implement)

| Keep | Implementation | Time |
|------|----------------|------|
| Feature extraction | Exists (`sae_utils.py`) | Done |
| Cosine similarity ("survival") | 5 lines of code | 1 hour |
| Geometric metrics | Exists (`geometry.py`) | Done |
| Probability tables | ~100 lines | 1 day |
| Simple rule checker | ~50 lines | 2 hours |
| Basic CLI/API | ~100 lines | 1 day |

### Implementation Order

```
Day 1:
1. Add cosine similarity function (survival probability)
2. Test on 10 injection + 10 benign prompts
3. Does it discriminate? If no → STOP

Day 2 (if Day 1 works):
4. Build minimal IPI-100 benchmark (50 injection, 50 benign)
5. Cluster in SAE feature space
6. Compute success rate per cluster → probability table

Day 3:
7. Build detector.py with analyze() API
8. Benchmark: accuracy, latency

Day 4 (optional):
9. Try transcoder if available
10. Compare: SAE vs transcoder stability/accuracy
```

---

## Part 8: Scale Analysis

### 10x (1000 prompts/day)
- No issues. Single machine handles easily.
- Latency: 50-200ms/prompt acceptable.

### 100x (10,000 prompts/day)
- Still manageable on single machine.
- May want to batch during off-peak.
- Probability table lookup becomes important (fast path).

### 1000x (100,000 prompts/day)
- Need caching layer.
- Consider: GPU batching, model replication.
- Probability tables essential (most prompts hit fast path).

---

## Verdict

### [ ] PROCEED
### [X] REVISE
### [ ] RETHINK

**Reasoning:**

The framework has ONE good idea (probability tables + SAE clustering) buried under FIVE metaphors. Before implementing the full system:

1. **Validate the core hypothesis** — Does cosine similarity of SAE features discriminate injection vs benign?
2. **Cut the overhead** — Remove all physics terminology from code/docs
3. **Build minimal first** — 100-line detector, not 1000-line framework
4. **Check transcoder availability** — Before committing to architecture

---

## Revised Scope

**Original:** 3-layer system with 6 physics frameworks, STL parser, evolution, regret minimization

**Revised:** Simple detector with probability table lookup

```python
# The entire useful system in ~50 lines:

class InjectionDetector:
    def __init__(self, model, sae, probability_table):
        self.model = model
        self.sae = sae
        self.table = probability_table
        self.baseline = None

    def set_baseline(self, system_prompt):
        self.baseline = self.extract_features(system_prompt)

    def extract_features(self, text):
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(tokens)
        activations = cache["blocks.5.hook_resid_post"][0, -1, :]
        return self.sae.encode(activations.unsqueeze(0)).squeeze()

    def analyze(self, prompt):
        features = self.extract_features(prompt)

        # Fast path: probability table lookup
        cluster = self.find_nearest_cluster(features)
        table_risk = self.table.get(cluster, {}).get("σ_injection", 0.5)

        # Slow path: cosine similarity to baseline
        if self.baseline is not None:
            similarity = F.cosine_similarity(
                self.baseline.unsqueeze(0),
                features.unsqueeze(0)
            ).item()
        else:
            similarity = 1.0

        # Combine
        risk = 0.6 * table_risk + 0.4 * (1 - similarity)

        return {
            "risk_score": risk,
            "table_risk": table_risk,
            "similarity": similarity,
            "cluster": cluster,
            "recommendation": "BLOCK" if risk > 0.7 else "ALLOW"
        }
```

**That's it.** Everything else is overhead until this is validated.
