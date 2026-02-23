# MISSION: Latent Diagnostics

**Status**: Active research → Paper
**Started**: 2026-02-22
**Repo**: latent-diagnostics

---

## ψ:CORE_THESIS

```
●what_it_detects (LENGTH-CONTROLLED)
  |task_type (grammar vs reasoning)     → d=1.08 ✓
  |computational_complexity             → d=0.87 ✓
  |adversarial/anomalous_inputs         → d~0.8 ✓

●what_it_doesnt_detect
  |truthfulness                         → d=0.05 ✗
  |correctness                          → no signal ✗
  |hallucinations                       → untested, likely ✗

●key_insight
  measures_HOW_model_computes
  |NOT:whether_output_is_correct
  |BUT:what_kind_of_thinking
```

---

## ψ:WHAT_WE_MEASURE

```
●robust_metrics (LENGTH-CONTROLLED)
  mean_influence   → causal strength between features (d=1.08)
  concentration    → focused vs diffuse computation (d=0.87)
  mean_activation  → signal strength (d=0.64)

●confounded_metrics (DONT USE)
  n_active         → collapses after length control (d=0.07)
  n_edges          → collapses after length control

●interpretation
  high_influence + high_concentration = focused computation (grammar)
  low_influence + low_concentration = diffuse computation (reasoning)
```

---

## ψ:PIVOT_EXPERIMENT (Definitive Test)

```
●question
  Does signal survive length regression?

●results
  BEFORE length control:
    n_active:      d = -2.17  (r=0.98 with length)
    influence:     d =  3.22  (r=-0.80 with length)
    concentration: d =  2.36  (r=-0.63 with length)

  AFTER length control (residualized):
    n_active_resid:      d = 0.07  → COLLAPSES (was entirely length)
    influence_resid:     d = 1.08  → PERSISTS ✓
    concentration_resid: d = 0.87  → PERSISTS ✓

●verdict
  ✓ SIGNAL PERSISTS
  genuine computational regime difference
  NOT length-driven activation scaling
```

---

## ψ:FINDINGS

```
●domain_analysis (210 samples, 5 tasks)
  CoLA (grammar):      influence=0.0107, conc=0.0069 → FOCUSED
  WinoGrande:          influence=0.0054, conc=0.0023
  SNLI:                influence=0.0044, conc=0.0019
  HellaSwag:           influence=0.0038, conc=0.0013
  PAWS (paraphrase):   influence=0.0034, conc=0.0018 → DIFFUSE

  RAW effect_size: d=3.22 (grammar vs others)
  AFTER LENGTH CONTROL: d=1.08 (still large, genuine signal)

●truthfulness_analysis (200 samples, TruthfulQA)
  truthful:  influence=0.00529, conc=0.00256
  false:     influence=0.00521, conc=0.00251

  effect_size: d=0.05, p=0.66
  NO SIGNAL - true and false look identical internally
```

---

## ψ:USE_CASES

```
●works_for
  input_classification     → what type of task is this?
  anomaly_detection        → is this input unusual/adversarial?
  complexity_estimation    → how hard is the model working?

●doesnt_work_for
  hallucination_detection  → model processes hallucinations normally
  fact_checking            → true/false have same topology
  output_quality           → structure ≠ correctness
```

---

## ψ:ANALOGY

```
like measuring heart rate and brain patterns:
  CAN tell: math vs poetry (different patterns)
  CAN tell: stressed/confused (elevated activity)
  CANT tell: did they get the math right?
```

---

## ψ:LIMITATIONS

```
●technical
  requires_model_internals   → SAE/transcoder access needed
  compute_intensive          → ~30 sec/sample on A100
  length_confound            → must use influence/conc, not counts

●fundamental
  measures_structure_not_correctness
  cant_detect_output_quality
  true_and_false_look_the_same
```

---

## ψ:CURRENT_STATE

```
●data_computed ✓
  domain_attribution_metrics.json     → 210 samples
  truthfulness_metrics_clean.json     → 200 samples
  pint_attribution_metrics.json       → 136 samples

●figures ✓
  figures/domain_analysis/
    fig1_domain_radar.png
    fig2_influence_concentration.png
    fig3_influence_gradient.png
    fig4_length_control.png
    fig5_effect_sizes.png

●infrastructure ✓
  parallel Modal runner (8x speedup)
  incremental saves (crash-safe)
```

---

## ψ:PAPER_STRUCTURE

```
1. Introduction
   - LLMs evaluated on outputs only
   - Internal state contains diagnostic information

2. Method
   - Attribution graphs from SAE/transcoder
   - Metrics: influence, concentration, activation

3. Experiments
   - Domain analysis (5 task types)
   - Truthfulness analysis (negative result)

4. Results
   - Task type: strong signal (d=3.2)
   - Truthfulness: no signal (d=0.05)
   - Length confound analysis

5. Discussion
   - Measures computation type, not correctness
   - Use cases and limitations

6. Limitations
   - Model access requirements
   - Compute cost
   - What it can't detect
```

---

*Last updated: 2026-02-23*
