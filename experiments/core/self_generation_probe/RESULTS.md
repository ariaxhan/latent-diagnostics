# Self-Generation Probe — Results (local run, 2026-06-15)

Pre-registered protocol: `protocol.md` (locked 2026-06-12). Harness: `probe_harness.py`
(thresholds locked 2026-06-15, before any data). This run honored the registration —
no thresholds moved, no arms dropped post-hoc.

## How it was run (fully local, no Modal / no cloud / no spend / no creds)

- **Model:** `mlx-community/gemma-2-2b-4bit` (base/PT, 4-bit, MLX on MPS) — ungated mirror.
  Greedy decoding, disjoint 5-shot prompt (few-shot entities share nothing with the eval set).
- **SAE:** GemmaScope `gemma-scope-2b-pt-res`, width-16k JumpReLU, layers 9/12/20
  (avg-L0 73/82/71). Ungated `.npz` weights, encoded in numpy/MLX (no sae_lens/torch).
- **Activations:** `blocks.{9,12,20}.hook_resid_post`, captured on a teacher-forced pass
  over prompt+self-answer, mean-pooled over the answer-token span.
- **Dataset:** 160 hand-curated entity-QA items (80 high-pop / 80 low-pop, 5 templates,
  balanced), gold answers hand-verified. `build_dataset_local.py`.
- **Labels:** normalized string match on gold → 129 correct / 31 incorrect / 0 refusal.
- Extraction: `extract_local.py` (~97s for 160 on an M-series Mac).

## Headline numbers (5-fold CV, length-controlled, bootstrap 95% CI, n=160)

| Arm | best AUC | CI95 | notes |
|---|---|---|---|
| **D — output-only (logits)** | **0.899** | [0.761, 0.926] | max_logit_prob / entropy / mean-logprob |
| A — residual stream | 0.883 (L12) | [0.804, 0.935] | the internals baseline-to-beat |
| B — GemmaScope SAE (k≤64) | 0.848 (L12) | [0.772, 0.923] | k=256 ≈ 0.855 |
| C — geometry (control) | 0.683 (L9) | [0.315, 0.550] | unstable; see H3 |

Length control applied to **every** arm (residualize vs answer-token count — the r=0.98 house
rule). Raw length differs by class (incorrect 3.90 tok vs correct 2.66) but is regressed out.

## Pre-registered hypotheses

- **H1 (residual AUC ≥ 0.75): CONFIRMED.** 0.883, CI lower bound 0.804.
  On the model's *own* generations the residual stream detects wrong answers. The
  HB-1000 / TruthfulQA null (d=0.05, p=0.66) **was a design artifact** — reading third-party
  text vs the model asserting. Self-generation flips it.
- **H2 (SAE k≤64 ≥ 90% of residual): CONFIRMED, but hollow.** 0.848 = 96% of 0.883, so the
  *locked* PIVOT_DROP_SAE branch does not fire. BUT the SAE arm is strictly dominated
  (SAE ≤ residual ≤ output) — it clears the bar without ever being the right tool. No reason
  to prefer SAEs; consistent with Kantamneni ICML'25 and the repo's "SAEs are dead weight" prior.
- **H3 (geometry at chance): inconclusive, not a clean confirm.** Mean-fold AUC 0.683 but the
  pooled out-of-fold bootstrap CI sits *below* chance ([0.315, 0.550]) — the geometry arm is
  noisy and fold-unstable at this n, not a reliable detector and not a clean chance result.

## The big question (commission's "honest nag")

**Does anything INSIDE the model beat reading the output probabilities? — NO.**

- Output-only (arm D) = **0.899**, the highest arm.
- Best residual = 0.883 (Δ −0.016 vs D); best SAE = 0.855 (Δ −0.044 vs D).
- All CIs overlap heavily. Internal signal **matches but does not exceed** the free logit baseline.

So: truth *is* internally decodable on self-generations — but at 2B the model's own output
probabilities are an equal-or-better, far cheaper detector. **Interpretability methods (residual
probes, SAEs) do not earn their cost over reading the logits.** Clean negative for
interpretability-for-correctness; clean positive for "self-generation flips the reading-vs-asserting null."

## Confound controls (do not repeat the injection-breakthrough sin)

1. **Length:** controlled (every arm residualized vs answer-token count). Arm D's 0.899 is post-control.
2. **Popularity leakage (the real caveat):** 27/31 incorrect items are low-popularity. In-domain
   probes partly exploit familiarity, not truth per se. The **transfer split measures it**:
   train high-pop → test low-pop AUC ≈ 0.73 (L12 0.761), train low → test high ≈ 0.68 — both
   degrade from 0.88 but stay above chance. Truth-specific signal that generalizes across
   popularity is **~0.73, not 0.88**; the headline is optimistic.
3. **Quantization:** gemma ran 4-bit; GemmaScope was trained on fp gemma → SAE features are
   mildly off-distribution, which can *understate* arm B (so the "SAEs add nothing" read is, if
   anything, conservative — favorable to SAEs and they still lost).
4. **n=160, not 500:** traded for hand-verified gold; CIs are correspondingly wide.

## Verdict

- **Pre-registered locked verdict: `ALIVE`** (H1 ✓ and H2 ✓).
- **Qualified scientific read:** internal truth-signal on self-generations is real and flips the
  old null — but it is **dominated by the output-only baseline**, the SAE arm adds nothing over
  residual, and cross-popularity generalization is ~0.73. The interesting finding is the negative:
  at 2B you do not need internals; logits suffice.

## Next-step gate

The DEAD-AT-2B gate did **not** trigger (H1 held), so no automatic escalation to 9B. The open
question 9B would answer is *not* "is there internal signal" (there is) but "does internal signal
ever **beat** the logit baseline, or pull ahead on the cross-popularity transfer split where the
output baseline also weakens." That is the only run worth the GPU; in-domain 2B is settled.

## Files

- `build_dataset_local.py` · `extract_local.py` · `report.py` (this run's local pipeline)
- `probe_harness.py` · `protocol.md` (pre-registered, unchanged)
- `data/results/self_generation_probe/probe_results.json` (committed; full AUC table)
- `records.json` (22 MB raw activations — gitignored, reproducible via `extract_local.py`)
