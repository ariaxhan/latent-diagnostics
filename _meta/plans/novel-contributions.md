# Novel Contributions Plan

## Current State (Not Novel)
- Run Anthropic's circuit-tracer on text samples
- Compute basic summary stats (mean, max, concentration)
- Find some metrics distinguish task types after length control

## What Makes This Novel

### Phase 1: Graph-Theoretic Metrics

**Goal:** Extract richer structure from the adjacency matrix than simple mean/max.

**New metrics to compute (during Modal run):**

```python
# Given: adj = graph.adjacency_matrix (features x features)

# 1. DEGREE DISTRIBUTION
# How connected is each feature?
out_degree = (adj.abs() > 0.01).sum(dim=1)  # outgoing edges per feature
in_degree = (adj.abs() > 0.01).sum(dim=0)   # incoming edges per feature
degree_mean = out_degree.float().mean()
degree_std = out_degree.float().std()
degree_skew = ...  # are there hub features?

# 2. EIGENVALUE SPECTRUM
# What's the "shape" of the computation?
# Top eigenvalues capture dominant modes of information flow
eigenvalues = torch.linalg.eigvalsh(adj @ adj.T)  # or SVD
top_eigenvalue = eigenvalues[-1]
eigenvalue_entropy = -sum(p * log(p)) where p = normalized eigenvalues
spectral_gap = eigenvalues[-1] - eigenvalues[-2]  # how "dominant" is top mode?

# 3. CLUSTERING / MODULARITY
# Do features form cliques or is it diffuse?
# Approximate: how often do A->B and A->C imply B connected to C?
# (Full computation is expensive, use sampling)

# 4. PATH STRUCTURE
# How many "hops" from input to output?
# Approximate via matrix powers: adj^k shows k-hop connections

# 5. HUB FEATURES
# Which features have highest centrality?
# PageRank on the adjacency matrix
pagerank = compute_pagerank(adj)
top_hub_indices = pagerank.topk(10)
hub_concentration = pagerank.topk(10).sum() / pagerank.sum()
```

**Hypothesis:** These metrics will capture structure that mean/max misses. Grammar tasks might have:
- Lower spectral entropy (more structured)
- Higher hub concentration (fewer key features)
- Shorter effective path lengths

---

### Phase 2: Intervention Experiments

**Goal:** Prove CAUSAL relationship between topology and behavior.

**Experiment 1: Feature Ablation**

```python
# 1. Run attribution to get graph
graph = attribute(text, model)

# 2. Identify top-k influential features
top_features = get_top_influential_features(graph, k=10)

# 3. Ablate each feature and measure:
for feature_id in top_features:
    # Set feature activation to 0
    model_ablated = ablate_feature(model, feature_id)

    # Run forward pass
    output_ablated = model_ablated(text)

    # Measure impact
    logit_change = kl_divergence(output_original, output_ablated)

    # Record
    ablation_effects[feature_id] = logit_change
```

**Key question:** Do grammar tasks show LARGER ablation effects than reasoning tasks?

If yes → Grammar computation is more "fragile" / concentrated
If no → Both equally distributed

**Experiment 2: Regime Steering**

```python
# Can we STEER a reasoning task to look like grammar?

# 1. Compute "grammar profile" = average feature activations for grammar tasks
grammar_profile = mean([get_activations(text, model) for text in grammar_samples])

# 2. For a reasoning sample, steer toward grammar profile
reasoning_text = "The trophy doesn't fit because it's too big"
steered_output = model.forward_with_steering(
    reasoning_text,
    target_profile=grammar_profile,
    strength=0.5
)

# 3. Measure: Does the output change? Does the attribution topology change?
```

**Key question:** Can we make the model "think like it's doing grammar" on a reasoning task?

---

### Phase 3: What This Proves

**If graph metrics differentiate tasks:**
→ "Attribution topology has richer structure than simple aggregates"
→ "Different tasks have different computational 'shapes'"

**If ablations have different effects:**
→ "Grammar is causally more concentrated than reasoning"
→ "Influence concentration predicts fragility"

**If steering works:**
→ "Computational regime is malleable"
→ "We can control HOW the model thinks, not just WHAT it outputs"

---

## Implementation Plan

### Step 1: Update Modal Script for New Metrics
- Modify `modal_general_attribution.py`
- Add eigenvalue computation, degree stats, etc.
- Re-run on domain samples

### Step 2: Create Intervention Script
- New script: `modal_intervention_experiments.py`
- Implement ablation experiments
- Run on subset of samples (compute-intensive)

### Step 3: Analysis
- Compare new metrics across task types
- Analyze ablation effect sizes
- Update notebook with findings

---

## Compute Estimates

**New metrics:** ~same cost as current runs (metrics computed from existing adj matrix)

**Interventions:** ~10x current cost (multiple forward passes per sample)
- 210 samples × 10 ablations × 30 sec = ~17 GPU-hours
- At Modal A100 rates: ~$30-50

---

## Timeline

1. **Day 1:** Implement new metrics in Modal script, test on 5 samples
2. **Day 2:** Full run with new metrics (210 samples)
3. **Day 3:** Implement intervention script, test on 5 samples
4. **Day 4:** Intervention experiments (subset of samples)
5. **Day 5:** Analysis and notebook update
