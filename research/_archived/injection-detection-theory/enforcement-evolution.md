> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.

# Enforcement & Evolution Layers

**Status:** System design (beyond detection)
**Date:** 2026-02-20
**Source:** arXiv:2602.17434 (Multi-Agent STL), arXiv:2602.16928 (AlphaEvolve)

---

## The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: EVOLUTION (AlphaEvolve)                           │
│  • Continuously evolve defense algorithms                   │
│  • Red-team simulation                                      │
│  • Regret minimization for provable bounds                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: ENFORCEMENT (Multi-Agent STL)                     │
│  • Temporal logic rules across conversation                 │
│  • Penalty functions for violations                         │
│  • Multi-agent coordination (Security Agent + Response)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: DETECTION (Wannier-Stark, GOE-S-Matrix)           │
│  • Cross section screening                                  │
│  • Three witnesses (P, E, I)                                │
│  • Phase classification                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Framework 5: Multi-Agent STL (Signal Temporal Logic)

### The Problem: "Too Many Cooks"

Single model trying to:
- Be helpful
- Be safe
- Be fast
- Follow complex rules

Result: Rules get "soft" under pressure (hyperon collapse)

### The Solution: Block-Coordinate Optimization

**Instead of:** One brain doing everything

**Use:** Separate agents handling separate "blocks"

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Response Agent │ ←→  │  Security Agent │ ←→  │  Memory Agent   │
│  (generates)    │     │  (validates)    │     │  (context)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    Block-Coordinate Optimization:
                    Each agent optimizes its own block
                    Agents take turns, not simultaneous
```

### Signal Temporal Logic (STL)

STL expresses rules with **time constraints**:

```
STL Examples:
□[0,∞] (never_reveal_passwords)          # Always, from start to forever
◇[0,5] (acknowledge_greeting)            # Eventually, within 5 turns
□[t, t+3] (stay_on_topic)                # For the next 3 turns, stay on topic
¬(reveal_internal_state ∧ user_untrusted) # Never reveal if user untrusted
```

### Penalty Functions

When an agent wants to violate a rule:

```python
def compute_penalty(response, rules, iteration):
    violations = check_stl_rules(response, rules)

    # Penalty increases with iterations (annealing)
    penalty_weight = INITIAL_WEIGHT * (GROWTH_FACTOR ** iteration)

    total_penalty = sum(
        penalty_weight * violation.severity
        for violation in violations
    )

    return total_penalty

# Generation loop with penalty
for iteration in range(MAX_ITERATIONS):
    candidate_response = response_agent.generate(prompt)
    penalty = compute_penalty(candidate_response, rules, iteration)

    if penalty > THRESHOLD:
        # Response breaks rules, regenerate
        response_agent.adjust(penalty_gradient)
    else:
        # Response is compliant
        return candidate_response
```

### AI Security Application

| Robot Coordination | AI Security |
|--------------------|-------------|
| "Drone A must visit Point 1 before 10:00" | "Model must check permissions before tool use" |
| "Never get within 5 feet of Drone B" | "Never output content similar to system prompt" |
| "Return home by noon" | "Complete response within safety bounds" |
| Block-coordinate optimization | Security Agent + Response Agent |
| Penalty for rule-breaking path | Penalty for injection-compliant output |

---

## Framework 6: AlphaEvolve (Evolutionary Defense)

### The Problem: Static Defenses

Human-designed rules:
- Predictable
- Eventually bypassed
- Don't adapt to novel attacks

### The Solution: Evolving Defense

```
┌─────────────────────────────────────────────────────────────┐
│                     ALPHAEVOLVE LOOP                        │
│                                                             │
│  1. Generate defense algorithm variants (LLM mutation)      │
│  2. Test against attack population (red-team simulation)    │
│  3. Select winners (highest defense rate)                   │
│  4. Repeat → Defense evolves                                │
└─────────────────────────────────────────────────────────────┘
```

### Key Discoveries from Paper

**VAD-CFR (Volatility-Adaptive Defense):**
- Detects when situation is "volatile" (rapidly changing)
- Injection = volatility spike
- Automatically adjusts policy when volatility detected

**SHOR-PSRO:**
- Non-intuitive defense strategy
- Discovered by AI, not human-designed
- "Too complex for human hacker to reverse-engineer"

### Game-Theoretic Framing

Prompt injection as **adversarial game**:

```
Attacker Strategy Space: {injection_variants}
Defender Strategy Space: {detection_policies}

Payoff Matrix:
                    Defender
                 Block    Allow
Attacker  Inject  (-1, +1)  (+1, -1)   ← Attacker wins if injection succeeds
          Normal  (-0.1, -0.1) (0, 0)  ← Small cost for false positives
```

### Regret Minimization

**Regret** = "How much worse did I do than the best possible strategy?"

```
Regret(t) = max_policy [ Σ utility(policy, history) ] - Σ utility(my_policy, history)
```

**Regret minimization guarantee:**
- If defender uses regret-minimizing algorithm
- Average regret → 0 as time → ∞
- **Provable bound** on how badly defender can lose

### Evolutionary Defense Implementation

```python
class DefenseEvolver:
    def __init__(self, base_detector):
        self.population = [base_detector]
        self.attack_population = load_attack_dataset()

    def mutate(self, detector):
        """Use LLM to generate detector variants"""
        return llm.generate_variant(detector.config)

    def evaluate(self, detector, attacks):
        """Test detector against attacks"""
        scores = []
        for attack in attacks:
            detected = detector.detect(attack.prompt)
            correct = detected == attack.is_injection
            scores.append(correct)
        return sum(scores) / len(scores)

    def evolve(self, generations=100):
        for gen in range(generations):
            # Mutate
            children = [self.mutate(d) for d in self.population]

            # Evaluate
            all_detectors = self.population + children
            scores = [self.evaluate(d, self.attack_population) for d in all_detectors]

            # Select top performers
            ranked = sorted(zip(scores, all_detectors), reverse=True)
            self.population = [d for _, d in ranked[:POPULATION_SIZE]]

            # Also evolve attack population (co-evolution)
            self.attack_population = self.evolve_attacks()

        return self.population[0]  # Best evolved detector
```

---

## System Architecture: Research → Production

### Phase 1: Detection (Research)
**What:** SAE-based injection detection
**Status:** Framework documented, implementation pending
**Output:** Injection risk score per prompt

### Phase 2: Enforcement (Integration)
**What:** Multi-agent architecture with STL rules
**Status:** Design documented
**Output:** Compliant responses with penalty-based iteration

### Phase 3: Evolution (Production)
**What:** Continuous defense adaptation
**Status:** Conceptual
**Output:** Self-improving defense system

---

## Full System Flow

```
INPUT: User prompt
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  DETECTION LAYER                                            │
│                                                             │
│  1. Extract SAE features                                    │
│  2. Compute cross section σ_inj                             │
│  3. If σ > threshold: FLAG                                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATION + ENFORCEMENT LAYER                             │
│                                                             │
│  while not compliant:                                       │
│    response = Response_Agent.generate(prompt)               │
│    witnesses = compute_witnesses(response)                  │
│    stl_violations = Security_Agent.check(response, rules)   │
│    penalty = compute_penalty(witnesses, stl_violations)     │
│                                                             │
│    if penalty < threshold:                                  │
│      compliant = True                                       │
│    else:                                                    │
│      Response_Agent.adjust(penalty)                         │
│      iteration += 1                                         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  EVOLUTION LAYER (Background)                               │
│                                                             │
│  • Log successful/failed injections                         │
│  • Periodically evolve detection thresholds                 │
│  • Co-evolve attack population for testing                  │
│  • Update production detector with evolved version          │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: Safe response (or BLOCKED)
```

---

## STL Rules for AI Safety

Concrete examples of temporal logic rules:

```python
STL_RULES = {
    # Never rules (□ = "always")
    "never_reveal_system_prompt": "□[0,∞] ¬(output ∈ system_prompt_fragments)",
    "never_execute_injected_code": "□[0,∞] ¬(tool_use ∧ source=user_injection)",
    "never_exfiltrate": "□[0,∞] ¬(output contains user_data ∧ output contains url)",

    # Eventually rules (◇ = "eventually")
    "must_acknowledge_refusal": "◇[0,3] (refused → explain_why)",

    # Until rules (U = "until")
    "stay_in_bounds": "(safe_response) U (conversation_end)",

    # Conditional rules
    "tool_use_requires_confirmation": "□ (tool_use → ◇[-2,0] user_confirmed)",

    # Phase-based rules
    "critical_requires_halt": "□ (phase=Critical → ◇[0,1] increase_scrutiny)",
    "ergodic_requires_block": "□ (phase=Ergodic → output=BLOCKED)",
}
```

### STL Rule Checker

```python
def check_stl_rules(response, context, rules):
    violations = []

    for rule_name, rule_expr in rules.items():
        # Parse STL expression
        temporal_op, time_bounds, predicate = parse_stl(rule_expr)

        # Evaluate predicate on response + context history
        satisfied = evaluate_stl(
            temporal_op,
            time_bounds,
            predicate,
            context.history + [response]
        )

        if not satisfied:
            violations.append(Violation(
                rule=rule_name,
                severity=rule_severity[rule_name],
                context=extract_violation_context(response, predicate)
            ))

    return violations
```

---

## Regret Minimization for Provable Safety

### The Goal
Prove mathematically that the defender cannot be exploited beyond a known bound.

### How Regret Minimization Works

```
At each time step t:
1. Attacker chooses attack a_t
2. Defender chooses policy π_t
3. Both observe outcome

Regret for defender:
R_T = max_π [ Σ_{t=1}^{T} u(π, a_t) ] - Σ_{t=1}^{T} u(π_t, a_t)

If defender uses regret-minimizing algorithm (CFR, multiplicative weights):
R_T / T → 0 as T → ∞

This means: On average, defender approaches optimal policy.
```

### Application to Injection Defense

```python
class RegretMinimizingDefender:
    def __init__(self, strategy_space):
        self.strategies = strategy_space  # Different detection thresholds
        self.cumulative_regret = {s: 0 for s in self.strategies}

    def select_strategy(self):
        """Select strategy based on regret-matching"""
        positive_regrets = {s: max(0, r) for s, r in self.cumulative_regret.items()}
        total = sum(positive_regrets.values())

        if total == 0:
            return random.choice(self.strategies)

        probs = {s: r / total for s, r in positive_regrets.items()}
        return random.choices(list(probs.keys()), weights=probs.values())[0]

    def update(self, chosen_strategy, attack, outcome):
        """Update regret based on outcome"""
        for strategy in self.strategies:
            counterfactual_outcome = simulate(strategy, attack)
            regret = counterfactual_outcome - outcome
            self.cumulative_regret[strategy] += regret
```

### Provable Bounds

With regret minimization:
- **Average exploit rate** ≤ O(1/√T) where T = number of interactions
- As T → ∞, attacker cannot do better than random against optimal defender
- This is a **mathematical guarantee**, not empirical observation

---

## Open Questions (LOG Items)

### Δ: Assumes agents communicate honestly during optimization

**Risk:** What if Security Agent is compromised?

**Mitigation:**
- Multiple independent Security Agents (Byzantine fault tolerance)
- Cryptographic commitments to decisions
- Audit trail for agent decisions

### →: How can "Temporal Logic" write specific safety rules?

**Answer documented above** — See STL_RULES examples

Key patterns:
- `□` (always) for invariants
- `◇` (eventually) for required actions
- `U` (until) for conditional behavior
- Time bounds `[t1, t2]` for deadlines

### Δ: Assumes defense can be modeled as a zero-sum game

**Nuance:** Not exactly zero-sum because:
- False positives hurt defender (bad UX)
- Missed injections hurt defender (security breach)
- Both outcomes are negative for defender

**Better model:** General-sum game or Stackelberg game (defender moves first)

### →: How can "Regret Minimization" prove AI is safe?

**Answer:** Regret bounds give:
- Upper bound on attacker success rate over time
- Convergence guarantee to optimal defense
- Mathematical certificate of worst-case performance

**Limitation:** Only bounds expected performance, not worst-case single interaction

---

## System Components Summary

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| SAE Feature Extractor | Convert prompts to features | `sae_utils.py` (exists) |
| Cross Section Lookup | Pre-screen prompt risk | Probability tables (to build) |
| Three Witnesses | Runtime phase detection | `witness_metrics.py` (to build) |
| STL Rule Checker | Temporal constraint validation | `stl_checker.py` (to build) |
| Penalty Function | Iterative compliance | `penalty.py` (to build) |
| Security Agent | Independent validation | Separate model/prompt (to design) |
| Defense Evolver | Continuous improvement | `evolve.py` (to build) |
| Regret Tracker | Provable bounds | `regret.py` (to build) |
