# Self-Generation Probe Benchmark — Experiment Protocol (D-A)

Status: protocol locked 2026-06-12; awaiting GPU run.
Origin: CollabVault/research/sae-hallucination/2026-06-12-review-and-directions.md

## Hypothesis (pre-registered)

On the model's OWN generations (not corrupted input text):

- H1: A logistic probe on residual-stream activations at answer-entity tokens detects
  incorrect answers with AUC >= 0.75 (in-domain).
- H2: A probe restricted to the top-k contrastively-selected GemmaScope SAE features
  reaches >= 90% of the residual probe's AUC (k <= 64).
- H3: The inertia-tensor geometry metrics (sphericity, elongation, eigenvalue entropy,
  misalignment) remain at chance (|AUC - 0.5| < 0.05), as they were on HB-1000.

H1 confirmed + H3 confirmed = the HB-1000/TruthfulQA nulls were a design artifact
(reading vs asserting), not absence of internal signal. H1 refuted at 2B scale is also
informative: pushes the work to larger models before any further internals research.

## Why this design

Every in-house null used third-party text as input. The 2025-2026 literature finds
hallucination signal in self-generations and entity-awareness features:
- Ferrando et al., ICLR 2025 (arXiv:2411.14257) - known/unknown-entity SAE latents, Gemma-2-2B, ~layer 9
- Orgad et al., ICLR 2025 (arXiv:2410.02707) - probes on own generations, exact-answer tokens
- HalluSAE, 2026 (arXiv:2604.16430) - contrastive feature selection then probing
- Kantamneni et al., ICML 2025 (arXiv:2502.16681) - SAE probes must be compared to residual probes

## Design

### Dataset
- 500 entity-centric QA items: 250 high-popularity entities, 250 low-popularity
  (WikiData sitelink count as popularity proxy). Single-fact questions with
  string-checkable answers ("In which year was X founded?", "Who directed Y?").
- Balance check: question template distribution identical across popularity buckets.

### Generation
- Model: google/gemma-2-2b (matches all cached infra). Greedy decoding.
- Label each answer correct/incorrect: normalized string match, then LLM-judge
  fallback for near-misses (reuse CodingVault/vector-native/evaluation/judge.py).
- Discard refusals into a third class (analyzed separately, not dropped silently).

### Feature extraction (Modal, reuse scripts/modal_general_attribution.py plumbing)
Per sample, at the answer-entity token positions (mean-pooled) and at the final
prompt token, capture:
1. Residual stream: layers 9, 12, 20.
2. GemmaScope SAE activations at the same layers (16k width, same release as
   neural-polygraph runs).
3. Geometry metrics on the SAE activation cloud (reuse neural-polygraph
   inertia-tensor code verbatim): sphericity, elongation, eigenvalue entropy,
   effective dimensionality, misalignment angle.

### Detector arms
A. Logistic regression on residual stream (per layer). The baseline to beat/match.
B. Logistic regression on top-k SAE features, k in {16, 64, 256}, selected by
   contrastive mean-activation difference on the TRAIN split only (HalluSAE-lite).
C. Logistic regression on the 5 geometry metrics (control arm).
D. Output-only baselines: max logit prob, logit entropy of the first answer token,
   answer token mean logprob. (Cheap floor every internals method must clear.)

### Evaluation
- 5-fold cross-validation; report mean AUC +/- std.
- Transfer split: train on high-popularity entities, test on low (and vice versa) -
  the Orgad/Azizian generalization concern, measured rather than assumed.
- Length control: residualize every detector input against answer token count
  (house rule; n_active r=0.98 lesson).
- Statistics: DeLong test between arms A and B; bootstrap CIs (1000 resamples).

### Success / kill criteria
- ALIVE: H1 and H2 hold -> write up as "where the signal actually lives" +
  negative-result section from HB-1000/TruthfulQA.
- PIVOT: H1 holds, H2 fails -> SAE features at 2B don't concentrate the signal;
  report residual-probe result, drop SAE arm (consistent with Kantamneni et al.).
- DEAD AT THIS SCALE: H1 fails -> no internals work at 2B; any continuation moves
  to Gemma-2-9B + Gemma Scope 2 or stops.

## Cost & effort
- Claude-Code active time: ~8-10 h (dataset build 2h, generation+labeling 2h,
  Modal extraction 2-3h, probes+stats 2h, writeup 1-2h). No prior in-repo benchmark
  for this exact pipeline - range stated accordingly.
- GPU: ~500 samples x ~30s on A100 for extraction = ~$20-40 Modal.

## Files
- `probe_harness.py` - runnable skeleton (dataset schema, probe arms, eval loop).
- Outputs land in `data/results/self_generation_probe/` (gitignored parquet + JSON
  summary committed).
