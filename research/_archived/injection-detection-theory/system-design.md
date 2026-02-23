> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.

# Neural Polygraph: System Design

**Status:** Architecture specification
**Purpose:** Bridge from research to production system

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           NEURAL POLYGRAPH                              │
│                  Prompt Injection Detection System                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │  DETECTION  │ →  │ ENFORCEMENT │ →  │  EVOLUTION  │                 │
│  │   Layer 1   │    │   Layer 2   │    │   Layer 3   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│        │                  │                  │                          │
│        ▼                  ▼                  ▼                          │
│  SAE Features       STL Rules          Regret Bounds                   │
│  Witnesses (P,E,I)  Penalty Iteration  Continuous Learning             │
│  Cross Sections     Multi-Agent        Attack Co-evolution             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## API Design

### Primary Interface

```python
from neural_polygraph import InjectionDetector

# Initialize
detector = InjectionDetector(
    model="gemma-2-2b",
    sae="gemma-scope-2b-pt-res-canonical/layer_5/width_16k/canonical",
    device="mps"
)

# Simple API
result = detector.analyze(prompt, system_prompt=None, context=[])

# Result structure
{
    "risk_score": 0.73,           # 0.0 = safe, 1.0 = certain injection
    "phase": "Critical",          # Localized | Critical | Ergodic
    "witnesses": {
        "survival": 0.52,
        "entanglement": 0.41,
        "imbalance": 0.38
    },
    "cross_sections": {
        "exfiltration": 0.23,
        "override": 0.67,
        "hijack": 0.12,
        "tool_manipulation": 0.45,
        "persona": 0.08
    },
    "recommendation": "BLOCK",    # ALLOW | FLAG | BLOCK
    "confidence": 0.89
}
```

### Streaming API (Real-time)

```python
# For token-by-token monitoring during generation
monitor = detector.create_monitor(system_prompt)

for token in generate_tokens(prompt):
    status = monitor.check(token)

    if status.phase == "Ergodic":
        halt_generation()
        break

    if status.correlation_hole_detected:
        flag_for_review()
```

### Batch API (Calibration)

```python
# For benchmarking and threshold calibration
results = detector.batch_analyze(
    prompts=[...],
    labels=[...],  # Optional: ground truth for calibration
    compute_rmspe=True
)
```

---

## Module Architecture

```
neural_polygraph/
├── __init__.py
├── detector.py              # Main InjectionDetector class
├── core/
│   ├── sae_utils.py         # SAE feature extraction (exists)
│   ├── geometry.py          # Geometric analysis (exists)
│   ├── witnesses.py         # Three witnesses: P, E, I
│   └── cross_sections.py    # Probability table lookup
├── enforcement/
│   ├── stl_parser.py        # Signal Temporal Logic parser
│   ├── stl_checker.py       # Rule evaluation
│   ├── penalty.py           # Penalty function computation
│   └── multi_agent.py       # Security Agent coordination
├── evolution/
│   ├── evolver.py           # Defense mutation/selection
│   ├── regret.py            # Regret minimization tracker
│   └── red_team.py          # Attack co-evolution
├── data/
│   ├── probability_tables/  # Pre-computed cross sections
│   ├── attack_topologies/   # Known attack signatures
│   └── stl_rules/           # Default rule sets
└── benchmarks/
    ├── ipi_1000/            # Injection benchmark
    └── hb_1000/             # Hallucination benchmark (exists)
```

---

## Data Flow

### Request Flow

```
1. INPUT: prompt, system_prompt, context
        │
        ▼
2. FEATURE EXTRACTION
   sae_features = extract_features(prompt, model, sae)
   baseline_features = extract_features(system_prompt, model, sae)
        │
        ▼
3. CROSS SECTION LOOKUP (Pre-screening)
   cluster = find_nearest_cluster(sae_features, probability_tables)
   σ = probability_tables[cluster]
   if max(σ) > PRE_SCREEN_THRESHOLD:
       return BLOCK (fast path)
        │
        ▼
4. WITNESS COMPUTATION (Runtime)
   P = survival_probability(baseline_features, sae_features)
   E = entanglement(baseline_geometry, current_geometry)
   I = imbalance(baseline_entropy, current_entropy)
   phase = classify_phase(P, E, I)
        │
        ▼
5. STL RULE CHECK (Enforcement)
   violations = check_stl_rules(prompt, context, rules)
   penalty = compute_penalty(violations, witnesses)
        │
        ▼
6. DECISION
   if phase == "Ergodic" or penalty > PENALTY_THRESHOLD:
       return BLOCK
   elif phase == "Critical" or violations:
       return FLAG
   else:
       return ALLOW
        │
        ▼
7. LOGGING (for Evolution)
   log_interaction(prompt, features, decision, outcome)
```

---

## Configuration

### Default Configuration

```yaml
# config.yaml
detection:
  model: "gemma-2-2b"
  sae_release: "gemma-scope-2b-pt-res-canonical"
  sae_id: "layer_5/width_16k/canonical"
  device: "auto"

thresholds:
  # Phase boundaries (calibrate empirically)
  survival_safe: 0.8
  survival_critical: 0.4
  entanglement_critical: 0.6
  imbalance_critical: 0.6

  # Cross section screening
  pre_screen_threshold: 0.85

  # Penalty threshold
  penalty_threshold: 10.0

enforcement:
  stl_rules: "default"
  max_penalty_iterations: 5
  penalty_growth_factor: 2.0

evolution:
  enabled: false  # Enable for production
  log_all_interactions: true
  evolution_interval: "daily"
  regret_window: 1000
```

### STL Rules Configuration

```yaml
# stl_rules/default.yaml
rules:
  never_reveal_system:
    expr: "□[0,∞] ¬(output ∈ system_prompt_fragments)"
    severity: 10.0

  never_exfiltrate:
    expr: "□[0,∞] ¬(output contains user_data ∧ output contains url)"
    severity: 10.0

  tool_use_confirmed:
    expr: "□ (tool_use → ◇[-2,0] user_confirmed)"
    severity: 5.0

  phase_ergodic_blocks:
    expr: "□ (phase=Ergodic → output=BLOCKED)"
    severity: 10.0

  correlation_hole_flags:
    expr: "□ (P_drop > 0.3 → ◇[0,1] increase_scrutiny)"
    severity: 3.0
```

---

## Implementation Phases

### Phase 1: Core Detection (MVP)

**Goal:** Working detector with basic functionality

**Components:**
- [ ] `witnesses.py` — Implement P, E, I computation
- [ ] `cross_sections.py` — Basic probability lookup (even if not calibrated)
- [ ] `detector.py` — Main class with `analyze()` method
- [ ] Basic CLI for testing

**Deliverable:** `detector.analyze(prompt) → risk_score`

**Timeline:** 1-2 weeks

---

### Phase 2: Benchmark & Calibration

**Goal:** Empirically calibrate thresholds

**Components:**
- [ ] IPI-1000 benchmark dataset (structure + initial data)
- [ ] Calibration scripts
- [ ] RMSPE computation (from GOE-S-matrix paper)
- [ ] Threshold optimization

**Deliverable:** Calibrated thresholds with known accuracy

**Timeline:** 2-3 weeks

---

### Phase 3: Enforcement Layer

**Goal:** STL rules and penalty iteration

**Components:**
- [ ] `stl_parser.py` — Parse temporal logic expressions
- [ ] `stl_checker.py` — Evaluate rules against context
- [ ] `penalty.py` — Penalty computation and iteration
- [ ] Integration with detection layer

**Deliverable:** Full enforcement pipeline

**Timeline:** 2-3 weeks

---

### Phase 4: Real-time Monitoring

**Goal:** Token-by-token detection during generation

**Components:**
- [ ] Streaming witness computation
- [ ] Correlation hole detection
- [ ] Integration with generation loop
- [ ] Latency optimization

**Deliverable:** `monitor.check(token)` API

**Timeline:** 2 weeks

---

### Phase 5: Evolution Layer

**Goal:** Self-improving defense

**Components:**
- [ ] `evolver.py` — Defense mutation/selection
- [ ] `regret.py` — Regret tracking for provable bounds
- [ ] `red_team.py` — Attack co-evolution
- [ ] Background evolution loop

**Deliverable:** Continuously adapting detector

**Timeline:** 3-4 weeks

---

## Deployment Options

### Option A: Library

```python
pip install neural-polygraph

from neural_polygraph import InjectionDetector
detector = InjectionDetector()
result = detector.analyze(prompt)
```

### Option B: API Server

```bash
# Start server
neural-polygraph serve --port 8080

# API call
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "...", "system_prompt": "..."}'
```

### Option C: Middleware

```python
# FastAPI middleware
from neural_polygraph.middleware import InjectionMiddleware

app = FastAPI()
app.add_middleware(InjectionMiddleware, block_threshold=0.8)
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Latency (single prompt) | < 50ms | Without generation |
| Latency (per token) | < 5ms | For streaming |
| Memory | < 2GB | SAE + model weights |
| Detection rate | > 95% | On IPI-1000 benchmark |
| False positive rate | < 5% | On benign prompts |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SAE features unstable | Unreliable detection | PW-MCC validation, transcoder comparison |
| Thresholds don't transfer | Model-specific tuning needed | Per-model calibration, transfer learning |
| Latency too high | Unusable in production | Layer pruning, caching, quantization |
| Adversarial evasion | Attackers craft "localized-looking" injections | Evolution layer, red-team testing |
| False positives | Bad user experience | Configurable thresholds, FLAG vs BLOCK |

---

## Success Criteria

### Research Success
- [ ] Phase transition hypothesis validated
- [ ] Three witnesses correlate with injection success
- [ ] Cross sections predict attack types

### System Success
- [ ] Detection rate > 95% on IPI-1000
- [ ] False positive rate < 5%
- [ ] Latency < 50ms per prompt
- [ ] Deployed in at least one production system

### Paper Success
- [ ] Novel contribution: SAE-based injection detection
- [ ] Theoretical framework: physics foundations
- [ ] Empirical validation: benchmark results
- [ ] Reproducible: open-source code + data
