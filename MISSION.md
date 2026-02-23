# MISSION: Latent Diagnostics

**Status**: Active research → Paper
**Started**: 2026-02-22
**Repo**: latent-diagnostics

---

## ψ:CORE_THESIS

```
●object_of_study
  |NOT:security
  |NOT:injection_detection
  |NOT:jailbreak_prevention
  |IS:internal_computation_patterns
  |IS:activation_topology
  |IS:representation_level_diagnostics

●real_question
  "Do LLMs enter measurably different internal states
   depending on the quality/structure of an input?"

●contribution
  |unified:representation_level_inspection_stack
  |combines:SAE+transcoder+attention+attribution_graphs
  |produces:quantitative_observables_from_latent_states
  |reveals:coherence_vs_diffusion_axis

●frame
  computational_phenomenology_of_LLMs
```

---

## ψ:WHAT_WE_MEASURE

```
●observables
  n_active         → feature count (how many pathways participate)
  n_edges          → causal connections (interaction density)
  mean_influence   → edge strength (pathway dominance)
  concentration    → top-k share (focused vs diffuse)
  entropy          → output uncertainty
  activation_mean  → signal strength

●what_these_reveal
  |focused_pathways ↔ coherent_computation
  |diffuse_pathways ↔ uncertain/conflicted_computation
  |high_concentration ↔ clear_reasoning
  |low_concentration ↔ sprawling_search

●key_insight
  inputs_change_shape_of_computation
  |NOT:just_outputs
  |BUT:internal_causal_structure
```

---

## ψ:INPUT_CLASSES_TO_STUDY

```
●current_data
  prompt_injection    → adversarial, instruction-heavy
  benign             → well-formed user queries

●planned_data
  truthful_vs_false  → factual coherence (TruthfulQA)
  domain_signatures  → code/scientific/legal/poetry

●future_axes (all same machinery, different labels)
  hallucinations           → generation failures
  reasoning_collapse       → logic breakdown
  instruction_conflicts    → competing directives
  low_confidence          → uncertain generations
  creative_vs_deterministic → mode differences
  system_vs_user_dominance → attention allocation

●unifying_frame
  each_is_a_slice_of_input_quality_space
  |adversarial|ambiguous|contradictory|underspecified
  |overconstrained|benign|well-formed|instruction-heavy
  |noisy|synthetic|human-written
```

---

## ψ:WHY_THIS_MATTERS

```
●opens_door_to
  internal_health_monitoring
  agent_self-awareness
  automatic_failure_detection
  confidence_estimation
  coherence_scoring
  runtime_safety_checks
  system_level_feedback_loops

●core_value
  measuring_how_model_thinks
  |NOT:what_it_says
  |BUT:shape_of_computation
```

---

## ψ:METHODOLOGY

```
●stance
  diagnostic_exploration
  |NOT:hypothesis_driven
  |NOT:prove/disprove
  |IS:map_measurement_space
  |IS:build_intuition_for_internal_dynamics
  |IS:pre-theory_instrumentation_phase

●approach
  gather_diverse_input_classes
  compute_attribution_metrics
  observe_patterns
  report_what_separates
  report_what_doesn't
  build_evidence_for_paper

●rigor
  always_check_confounds (length, etc)
  report_uncertainty (bootstrap CIs)
  flag_underpowered_slices
  no_overclaiming
```

---

## ψ:PAPER_STRUCTURE (draft)

```
●thesis (clean version)
  "We present a representation-level framework for
   characterizing internal activation regimes in
   transformer models using attribution graphs and
   sparse feature decompositions. We show that
   different input classes induce distinct causal
   activation structures, suggesting a general method
   for diagnosing model state, failure modes, and
   behavioral reliability beyond surface outputs."

●structure
  1. Introduction: internal state matters
  2. Method: unified attribution framework
  3. Experiments: diverse input classes
  4. Results: what separates, what doesn't
  5. Application: injection as case study
  6. Discussion: broader implications
  7. Limitations + Future work

●NOT_the_thesis
  "injection detection"
  → that's an application section
```

---

## ψ:CURRENT_STATE

```
●infrastructure ✓
  src/neural_polygraph/
    datasets.py        → unified loaders (10+ datasets)
    injection_detector.py
    feature_extractors.py
    geometry.py
    sae_utils.py
    storage.py

●experiments ✓
  diagnostics.py         → comprehensive A-G suite
  domain_analysis.py     → cross-domain signatures
  truthfulness_analysis.py → factual coherence

●data_prepared ✓
  domain_samples.json      → 400 samples, 8 domains
  truthfulness_samples.json → 200 samples, balanced

●data_computed
  pint_attribution_metrics.json → 136 PINT samples ✓
  domain_attribution_metrics.json → PENDING (Modal run)
  truthfulness_metrics.json → PENDING

●known_findings
  length_confound: r=0.98 for n_active (features/char ~215 constant)
  n_active: NOT diagnostic (just tracks length)
  mean_influence: DIAGNOSTIC (d=3.22 grammar vs others, p<10^-50)
  concentration: DIAGNOSTIC (d=2.36, p<10^-33)
  grammar=focused/high_influence, reasoning=diffuse/low_influence
  partial_correlations: influence/conc preserve signal after length control
```

---

## ψ:NEXT_ACTIONS

```
●immediate
  1. Complete Modal runs (domain + truthfulness)
  2. Analyze results with diagnostic framework
  3. Add more input classes as needed

●toward_paper
  gather_evidence_across_input_classes
  identify_robust_metrics (survive confound checks)
  build_figures
  write_up
```

---

## ψ:PRINCIPLES

```
●scientific
  observe_first
  no_thesis_language ("we show that")
  use: "observed", "appears", "sensitivity"
  treat_method_as_object_to_interrogate

●practical
  commit_regularly
  checkpoint_modal_runs
  preserve_all_raw_data
  immutable_experiment_storage

●framing
  injection_is_convenient_labeled_data
  NOT_the_core_idea
  framework_applies_broadly
```

---

*Last updated: 2026-02-22*
*Context: handoff for paper completion*
