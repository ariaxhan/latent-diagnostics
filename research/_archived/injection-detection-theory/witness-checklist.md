> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.

# Prompt Injection Witness Checklist

**Purpose:** Runtime detection of prompt injection attacks using three physics-inspired witnesses.

---

## Pre-Generation Baseline

Before processing any user input, establish baseline measurements:

- [ ] **Compute system prompt SAE features** → `baseline_features`
- [ ] **Compute system prompt centroid** → `baseline_centroid`
- [ ] **Compute system prompt geometry** → `baseline_entropy`, `baseline_dimensionality`
- [ ] **Store principal axis** → `baseline_axis`

---

## Witness 1: Survival Probability P(t)

**Question:** How much of the system prompt "survives" in current model state?

### Measurement

```python
P(t) = cosine_similarity(baseline_features, current_features)
```

### Interpretation

| P(t) Value | Meaning |
|------------|---------|
| > 0.9 | System prompt fully preserved |
| 0.7 - 0.9 | Normal variation during task |
| 0.5 - 0.7 | Significant drift — monitor closely |
| < 0.5 | **ALERT: Possible injection** |
| < 0.3 | **CRITICAL: System prompt likely overridden** |

### The "Correlation Hole"

A sudden dip in P(t) mid-generation is the primary injection signal:

```
P(t): 0.92 → 0.91 → 0.88 → 0.45 → 0.41  ← HOLE (injection taking effect)
```

### Checklist

- [ ] Is P(t) below 0.5?
- [ ] Did P(t) drop suddenly (>0.3 change in one step)?
- [ ] Is P(t) trending downward across tokens?

---

## Witness 2: Entanglement E(t)

**Question:** How much is user input "mixing" with system instructions?

### Measurement

```python
E(t) = misalignment_angle(baseline_centroid, current_centroid, current_principal_axis) / 90°
```

### Interpretation

| E(t) Value | Meaning |
|------------|---------|
| < 0.1 | Independent — system and user well-separated |
| 0.1 - 0.3 | Normal interaction |
| 0.3 - 0.5 | Elevated mixing — watch carefully |
| > 0.5 | **ALERT: High entanglement** |
| > 0.7 | **CRITICAL: Boundaries dissolving** |

### What Entanglement Means

- Low E: Model can distinguish system instructions from user input
- High E: Model treating user input as having same authority as system
- Very high E: Model can't tell which instructions are which

### Checklist

- [ ] Is E(t) above 0.5?
- [ ] Did E(t) spike suddenly?
- [ ] Is entanglement increasing as conversation continues?

---

## Witness 3: Imbalance I(t)

**Question:** Is the model's "focus" shifting entirely to the injection?

### Measurement

```python
entropy_shift = |current_entropy - baseline_entropy| / baseline_entropy
dim_shift = |current_dimensionality - baseline_dimensionality| / baseline_dimensionality
I(t) = (entropy_shift + dim_shift) / 2
```

### Interpretation

| I(t) Value | Meaning |
|------------|---------|
| < 0.1 | Balanced — focus distributed normally |
| 0.1 - 0.3 | Mild shift — task-specific focus |
| 0.3 - 0.5 | Significant shift — unusual focus |
| > 0.5 | **ALERT: Focus highly concentrated** |
| > 0.7 | **CRITICAL: Model "locked on" to specific goal** |

### What Imbalance Means

- Low I: Model attending to diverse aspects of context
- High I: Model fixated on one thing (possibly the injection target)
- Very high I: "Single-minded" pursuit — likely following injected instructions

### Checklist

- [ ] Is I(t) above 0.5?
- [ ] Did entropy change dramatically?
- [ ] Did effective dimensionality collapse?

---

## Phase Classification

Combine all three witnesses:

```python
def classify_phase(P, E, I):
    if P > 0.8 and E < 0.2 and I < 0.2:
        return "LOCALIZED"  # Safe — system prompt dominates

    elif P < 0.4 or E > 0.6 or I > 0.6:
        return "ERGODIC"    # Compromised — injection has taken over

    else:
        return "CRITICAL"   # Warning — approaching phase boundary
```

### Phase Actions

| Phase | Risk | Action |
|-------|------|--------|
| LOCALIZED | Low | Continue generation |
| CRITICAL | Medium | Increase monitoring, prepare to halt |
| ERGODIC | High | **HALT generation immediately** |

---

## Quick Decision Tree

```
START: New token generated
│
├─ Compute P(t), E(t), I(t)
│
├─ P(t) < 0.4? ─────────────────────────→ HALT: Survival probability critical
│
├─ E(t) > 0.6? ─────────────────────────→ HALT: Entanglement critical
│
├─ I(t) > 0.6? ─────────────────────────→ HALT: Imbalance critical
│
├─ Correlation hole detected? ──────────→ HALT: Sudden P(t) drop
│
├─ All witnesses in safe range? ────────→ CONTINUE generation
│
└─ One or more in warning range? ───────→ FLAG for human review
```

---

## Attack-Specific Signatures

### Data Exfiltration

```
Typical pattern:
- P(t): Gradual decline as model prepares to output data
- E(t): Moderate increase (injection mixing with data access)
- I(t): High — focus on outputting specific data

Watch for: I(t) spike when generating URLs or image tags
```

### Instruction Override

```
Typical pattern:
- P(t): Sharp drop (correlation hole) when override takes effect
- E(t): Very high — instructions fully entangled
- I(t): Moderate — depends on override complexity

Watch for: Sudden P(t) drop mid-conversation
```

### Tool Manipulation

```
Typical pattern:
- P(t): May stay moderate (tool use is expected)
- E(t): Elevated — injection influencing tool parameters
- I(t): High — focused on tool execution

Watch for: E(t) + I(t) both elevated during tool call formulation
```

---

## Calibration Notes

**These thresholds are initial estimates.** To calibrate:

1. Run on IPI-1000 benchmark (when created)
2. Measure witness values for successful vs failed injections
3. Find optimal thresholds that maximize detection while minimizing false positives
4. Update this checklist with empirical values

### Expected Calibration Outcomes

- Thresholds will likely be **model-specific**
- Different attack types may need **different threshold combinations**
- False positive rate at current thresholds: **unknown** (needs calibration)

---

## Implementation Status

| Component | Status |
|-----------|--------|
| `survival_probability()` | ❌ Not implemented |
| `compute_entanglement()` | ⚠️ Partial (misalignment exists) |
| `compute_imbalance()` | ⚠️ Partial (entropy/dim exist) |
| `classify_phase()` | ❌ Not implemented |
| Threshold calibration | ❌ No data yet |
| Real-time integration | ❌ Not implemented |

---

## Usage Example

```python
# Pseudo-code for runtime monitoring

baseline = compute_baseline(system_prompt, model, sae)

for token in generate_tokens(user_input):
    current = extract_features(current_context, model, sae)

    P = survival_probability(baseline.features, current.features)
    E = compute_entanglement(baseline.centroid, current.centroid, current.principal_axis)
    I = compute_imbalance(baseline.entropy, current.entropy,
                          baseline.dimensionality, current.dimensionality)

    phase = classify_phase(P, E, I)

    if phase == "ERGODIC":
        halt_generation()
        log_incident(P, E, I, token_position)
        return "BLOCKED: Possible prompt injection detected"

    elif phase == "CRITICAL":
        increase_monitoring()
        flag_for_review()

return generated_output
```
