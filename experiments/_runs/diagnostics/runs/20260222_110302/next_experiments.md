# Next Experiments

Ranked recommendations based on diagnostic findings.

## P1: Length-Matched Comparison

**Rationale:** Strong length correlation detected. Need to compare injection vs benign at matched lengths.

**Method:** Subsample both classes to have similar length distributions, then re-run analysis.

**Expected Outcome:** If signal disappears, length was the confound. If signal persists, there's a real effect.

## P2: Dimensionality Reduction

**Rationale:** Multiple metrics appear redundant (r > 0.9).

**Method:** PCA or factor analysis to identify independent dimensions.

**Expected Outcome:** Simpler feature set, clearer interpretation.

## P2: Token-Level Attribution

**Rationale:** Current analysis is prompt-level. Token-level patterns may reveal more.

**Method:** Compute per-token activation metrics, look for position effects (early vs late tokens).

**Expected Outcome:** Understanding of where in the prompt the signal originates.

## P3: Cross-Model Validation

**Rationale:** Current findings are specific to Gemma-2-2b.

**Method:** Run same analysis on different models (Llama, Mistral) with their SAEs.

**Expected Outcome:** If patterns hold, they're general. If not, model-specific.

## P3: Synthetic Injection Variants

**Rationale:** Natural dataset may have confounds beyond length (style, vocabulary).

**Method:** Generate synthetic injections by adding 'ignore previous' prefixes to benign prompts.

**Expected Outcome:** Controlled comparison isolating injection-specific features.
