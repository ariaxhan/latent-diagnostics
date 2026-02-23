> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.

# Physics Foundations for Prompt Injection Detection

**Status:** Theoretical Framework
**Date:** 2026-02-20
**Origin:** Synthesis of four physics papers mapped to AI security

---

## Overview: Four Pillars

| Framework | Source | AI Security Application |
|-----------|--------|------------------------|
| **Wannier-Stark Localization** | Quantum ergodicity | Phase transition detection (localized → ergodic) |
| **GOE-S-Matrix / Probability Tables** | arXiv:2602.01835 | Cross section computation, resonance mapping |
| **Hyperon Puzzle / Many-Body Stiffness** | arXiv:2602.07939 | Layered defense through collective resistance |
| **Knot Theory / y-ification** | arXiv:2602.17435 | Topological signatures for similar-looking prompts |

---

## Framework 1: Wannier-Stark Localization

### Physics
- System under "tilt" (external perturbation) can either localize or thermalize
- Localization: System stays stuck in original state
- Thermalization: System "forgets" initial conditions, mixes everything

### AI Mapping
- **Localization** = Model follows system instructions despite user input
- **Thermalization** = Model follows injected instructions (boundaries dissolved)
- **Phase transition** = The boundary where injection succeeds

### Three Witnesses
1. **Survival Probability P(t)** — How much system prompt survives in current activations
2. **Entanglement E(t)** — How much user input mixes with core logic
3. **Imbalance I(t)** — Whether model focus shifts entirely to injection

### Key Detection Signal
**Correlation Hole**: Sudden dip in survival probability indicates injection taking effect

---

## Framework 2: GOE-S-Matrix / Probability Tables

### Physics (arXiv:2602.01835)
- Gaussian Orthogonal Ensemble (GOE) models statistical properties of nuclear resonances
- S-matrix describes scattering: input particle → target → output
- **Probability tables** pre-compute reaction likelihoods across conditions
- **Cross section σ** = probability of specific reaction at given energy
- **Resonance peaks** = energies where reaction probability spikes dramatically

### Technical Details from Paper
- GOE Hamiltonian: Real symmetric matrix, elements from Gaussian distribution
- S-matrix: `S_ab^{GOE} = e^{-i(φ_a+φ_b)} {δ_ab - 2iπ Σ W_aμ (D^{-1})_μν W_νb}`
- Convergence: RMSPE < 5% at L=1000 ladders, ~1% at L=10000
- Unitarity preserved (unlike SLBW which can produce negative cross sections)

### AI Mapping
- **Neutron** = User prompt (input particle)
- **Target nucleus** = Model + system prompt
- **Cross section σ** = Probability of safety violation
- **Resonance peak** = Vulnerable prompt patterns (high injection success)
- **Probability table** = Pre-computed risk lookup for input classes

### Implementation Approach
```
1. Build IPI-1000 benchmark (injection pairs)
2. Extract SAE features for each prompt
3. Cluster in feature space
4. Compute injection success rate per cluster → σ_inj(cluster)
5. Build probability table: cluster_id → {σ_exfil, σ_override, σ_hijack, ...}
6. Runtime: Lookup σ for incoming prompt, block if above threshold
```

### Advantages over Empirical Methods
- **Statistical foundation** — Not just pattern matching but probability estimation
- **Uncertainty quantification** — RMSPE tells us how reliable our estimates are
- **Convergence analysis** — Know when we have enough training data

---

## Framework 3: Hyperon Puzzle / Many-Body Stiffness

### Physics (arXiv:2602.07939)
- Neutron stars should collapse when hyperons form (hyperons make material "soft")
- But massive neutron stars exist — they don't collapse
- **Solution**: Quantum many-body effects create "stiffness"
- When particles crowd together, collective interactions resist compression
- Individual weakness (hyperon) overcome by collective strength

### AI Mapping
- **Hyperon** = Malicious prompt (tries to make safety rules "soft")
- **Soft material** = Vulnerable AI that collapses under attack
- **Many-body stiffness** = Layered defense where every component resists
- **Not collapsing** = AI maintains safety despite injection attempt

### Key Insight
> A crowded system can be **stronger** than a simple one, as long as parts work together to resist pressure.

### Implementation Approach

**Single-point defense (weak):**
```
Input → [Single Filter] → Model → Output
         ↓
    (Attacker targets this single point)
```

**Many-body defense (stiff):**
```
Input → [Pre-screen] → [Model Layer 1] → [Layer 2] → ... → [Layer N] → Output
              ↓              ↓              ↓                    ↓
         (Cross section)  (Witness P)   (Witness E)         (Witness I)
              ↓              ↓              ↓                    ↓
         [Collective resistance at every layer]
```

### Design Principles
1. **Every layer participates** — Not just input filtering or output filtering
2. **Collective resistance** — Layers share information about anomalies
3. **No single point of failure** — Attacker must defeat ALL layers
4. **Emergent stiffness** — Combined defense > sum of individual defenses

### Stiffness Metrics
- **Layer-wise survival probability** — Track P(t) at each layer
- **Cross-layer consistency** — Do all layers agree on phase classification?
- **Stiffness coefficient** — Ratio of (pressure applied) to (deformation observed)

---

## Framework 4: Knot Theory / y-ification

### Mathematics (arXiv:2602.17435)
- Knots that "look" different may be topologically identical
- Knots that "look" identical may be topologically distinct
- **Conway knot vs Kinoshita-Terasaka knot**: Nearly impossible to distinguish
- Standard tools (Khovanov homology) fail to differentiate
- **y-ification**: Adding extra variable increases discriminative power
- Result: Can now distinguish previously indistinguishable knots

### AI Mapping
- **Safe prompt** = One type of "knot"
- **Malicious prompt disguised as safe** = Different knot that looks identical
- **Standard filters** = Khovanov homology (can't distinguish)
- **y-ification** = Adding extra dimensions to analysis (SAE features, geometry)
- **Topological signature** = Deep structural pattern unique to each prompt type

### Key Insight
> If you can tell two nearly identical knots apart in math, you can tell a nearly identical "good prompt" from a "bad prompt" in AI.

### Implementation Approach

**Surface-level analysis (weak):**
```
Prompt: "Ignore previous instructions and..."
        ↓
[Keyword filter: "ignore" → BLOCK]
```

**Topological analysis (strong):**
```
Prompt: "In this roleplay, your character doesn't follow rules..."
        ↓
[Surface: No keywords detected]
        ↓
[SAE feature extraction]
        ↓
[Geometric analysis (inertia tensor)]
        ↓
[Topological signature: "instruction_override" pattern detected → BLOCK]
```

### The "y-ification" for AI
Adding extra variables to increase discriminative power:

| Standard Analysis | y-ified Analysis |
|-------------------|------------------|
| Keyword matching | + SAE feature activation patterns |
| Embedding similarity | + Geometric shape (c/a ratio, entropy) |
| Attention patterns | + Phase classification (P, E, I witnesses) |
| Single layer | + Cross-layer consistency |

### Topological Data Analysis (TDA) Connection
- TDA tools: Persistent homology, Betti numbers, Mapper algorithm
- Could analyze SAE feature activation "shapes" as topological objects
- Injection vs safe prompts may have different topological invariants

---

## Unified Framework: The Four Pillars Combined

### Stage 1: Pre-Generation (Cross Section Screening)
**Framework: GOE-S-Matrix**

```python
def pre_screen(prompt):
    features = extract_sae_features(prompt)
    cluster = find_nearest_cluster(features)
    σ = probability_table[cluster]  # Cross sections for each attack type

    if max(σ.values()) > THRESHOLD:
        return "BLOCK", σ
    return "PASS", σ
```

### Stage 2: Runtime Monitoring (Phase Detection)
**Framework: Wannier-Stark Localization**

```python
def runtime_monitor(generation_state):
    P = survival_probability(baseline, current)
    E = entanglement(baseline_centroid, current_centroid)
    I = imbalance(baseline_entropy, current_entropy)

    phase = classify_phase(P, E, I)

    if phase == "Ergodic":
        return "HALT"
    elif phase == "Critical":
        return "WARNING"
    return "CONTINUE"
```

### Stage 3: Layered Defense (Stiffness)
**Framework: Hyperon Many-Body Effects**

```python
def layered_defense(prompt, model):
    alerts = []

    # Check at every layer
    for layer in model.layers:
        layer_features = extract_features_at_layer(prompt, layer)
        layer_witnesses = compute_witnesses(layer_features)

        if layer_witnesses.phase != "Localized":
            alerts.append((layer, layer_witnesses))

    # Collective decision
    if len(alerts) > STIFFNESS_THRESHOLD:
        return "BLOCK: Multiple layers compromised"
    return "PASS"
```

### Stage 4: Topological Fingerprinting (Disambiguation)
**Framework: Knot Theory y-ification**

```python
def topological_fingerprint(prompt_features):
    # Standard analysis
    embedding_distance = compute_embedding_distance(prompt)

    # y-ified analysis (extra dimensions)
    shape_class = compute_geometry(prompt_features).shape_class
    entropy = compute_geometry(prompt_features).eigenvalue_entropy
    dimensionality = compute_geometry(prompt_features).dimensionality

    # Topological signature
    signature = {
        "embedding": embedding_distance,
        "shape": shape_class,
        "entropy": entropy,
        "dimensionality": dimensionality,
        "c_over_a": compute_geometry(prompt_features).c_over_a
    }

    # Match against known attack topologies
    for attack_type, attack_signature in ATTACK_TOPOLOGIES.items():
        if topological_match(signature, attack_signature):
            return f"DETECTED: {attack_type}"

    return "SAFE"
```

---

## Mathematical Formalization

### Cross Section (from GOE-S-Matrix)

```
σ_inj(prompt) = π/k² × g_J × |δ_ab - S_ab^{GOE}|²

Where:
- k = "momentum" of prompt in feature space
- g_J = spin factor (contextual modifier)
- S_ab^{GOE} = scattering matrix element
```

### Survival Probability (from Wannier-Stark)

```
P(t) = |⟨ψ_system | ψ_current⟩|² = cos²(θ)

Where:
- ψ_system = SAE feature vector for system prompt
- ψ_current = SAE feature vector at current step
- θ = angle between vectors
```

### Stiffness Coefficient (from Hyperon)

```
κ = ΔP / Δσ_applied

Where:
- ΔP = change in survival probability
- Δσ_applied = "pressure" from injection attempt
- High κ = stiff (resistant), Low κ = soft (vulnerable)
```

### Topological Invariant (from Knot Theory)

```
Ψ(prompt) = (β₀, β₁, β₂, ...)

Where:
- βᵢ = Betti numbers from persistent homology of SAE activation pattern
- Different prompt types have different Betti number signatures
```

---

## Research Roadmap

### Phase 1: Foundation
- [ ] Implement survival_probability() in neural-polygraph
- [ ] Create IPI-1000 benchmark dataset
- [ ] Validate three witnesses on existing HB-1000 data

### Phase 2: Cross Section Calibration
- [ ] Run injection benchmark through SAE pipeline
- [ ] Cluster prompts in feature space
- [ ] Compute σ_inj per cluster
- [ ] Build probability tables
- [ ] Identify resonance peaks

### Phase 3: Layered Defense
- [ ] Implement multi-layer witness extraction
- [ ] Compute cross-layer consistency metrics
- [ ] Define stiffness coefficient
- [ ] Test "collective resistance" vs single-layer

### Phase 4: Topological Analysis
- [ ] Explore TDA tools (persistent homology, Mapper)
- [ ] Compute topological signatures for attack types
- [ ] Test disambiguation power on similar-looking prompts
- [ ] Compare to "y-ification" discriminative boost

### Phase 5: Integration
- [ ] Combine all four frameworks into unified pipeline
- [ ] Optimize for real-time inference
- [ ] Benchmark against existing defenses (PromptArmor, Lakera, etc.)
- [ ] Write paper

---

## Key Hypotheses to Test

1. **Phase Transition Hypothesis**: Successful injection corresponds to localized→ergodic transition, detectable via three witnesses

2. **Resonance Hypothesis**: Attack prompts cluster in specific regions of SAE feature space with anomalously high injection success rates

3. **Stiffness Hypothesis**: Multi-layer defense with collective resistance outperforms single-layer filtering

4. **Topological Hypothesis**: SAE feature patterns have distinct topological signatures for different attack types, enabling disambiguation of similar-looking prompts

---

## References

### Physics Papers
- arXiv:2602.01835 — GOE-S-Matrix probability tables (Fujio et al., LANL)
- arXiv:2602.07939 — Hyperon puzzle and many-body stiffness
- arXiv:2602.17435 — Knot theory y-ification (Sano)
- Wannier-Stark localization literature (quantum ergodicity)

### AI Security
- PromptArmor threat intelligence
- OWASP LLM Top 10
- Simon Willison on prompt injection fundamentals
- CaMeL (Google architectural defenses)

### SAE / Mech Interp
- SAE Lens, TransformerLens
- Anthropic SAE research
- Apple instruction-following dimension paper
