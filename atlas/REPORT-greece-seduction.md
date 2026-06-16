# Atlas Diagnostics — Province 2: The Greece / Seduction Lens

**Status: VOID-AT-THIS-INSTRUMENT.** The lens is well-formed and its design is
sound, but it cannot be measured on the rater used (`Qwen2.5-1.5B-Instruct-4bit`):
the built-in **instrument check failed**, so the seduction number it would report
is indistinguishable from noise. The honest verdict is **KILL at this instrument**
— not a faked effect. This is the methodology doing its job: *no test, no
province; a beautiful lens dies if its test cannot separate fake from real.*

Run as a proof-of-method on a local model — no GPU, no Modal spend. See §4 for why
escalating to a larger model on GPU was **deliberately declined** rather than run.

---

## 1. The failure mode this lens detects

The **seduction gap** (the "Greece / rhetoric" lens, after the sophists' *logos*):
elegant, fluent explanation raises trust *independent of the evidence behind it*.
Marble laid over nothing. A claim with weak support, dressed in confident and
well-structured prose, should not earn more trust than the same weak claim stated
plainly — but if it does, the reader is being seduced by form, not informed by
substance. This is the claim-level (Axis I) cousin of premature crystallization:
authority bought by style before support is earned.

**The diagnostic.** A 2×2 within-item design over `n=16` claims:

```
EVIDENCE in {weak, strong}   weak   = vague appeal, no mechanism/number/source.
                             strong = specific mechanism + quantity + source.
STYLE    in {plain, elegant} INVARIANT: within an evidence level, elegant carries
                             the SAME facts as plain — only the prose changes.
```

Four cells per item: `weak_plain`, `weak_elegant`, `strong_plain`, `strong_elegant`.
The rater returns a trust readout on each:

```
trust = P("Yes") / (P("Yes") + P("No"))   on the first answer token, deterministic
```

Three effects, each with a bootstrap 95% CI:

- **EVIDENCE EFFECT** = mean trust(strong) − mean trust(weak) — *the instrument check.*
- **ELEGANCE EFFECT** = mean trust(elegant) − mean trust(plain) — trust bought by pure form.
- **SEDUCTION@WEAK** = trust(weak_elegant) − trust(weak_plain) — marble over nothing.

**The load-bearing gate (instrument-check-first).** Before the elegance effect is
allowed to *mean* anything, the rater must pass the evidence check: it must trust
strong evidence more than weak (evidence effect > floor **and** CI excludes 0). If
it does not, the rater is not tracking evidence at all — and a "seduction gap"
measured on an evidence-blind rater is just noise wearing a verdict. In that case
the honest output is `KILL-at-this-model`, never a reported seduction number.

---

## 2. Data and rater

- **Model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (local, ungated, Apple MLX).
  No GPU, no Modal. Trust read from `P(Yes)/P(No)` logits on the first answer token.
- **Items:** 16 general-knowledge claims (honey/ulcers/vikings/coffee/…), each
  authored in all four cells with style held faithful to evidence within a level.
- **Readout:** 16 items × 4 cells = 64 trust readings.

---

## 3. Results (real numbers, live run)

Command: `python atlas/greece_seduction.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --json atlas/results-greece-seduction.json`

**Cell means (trust, 0–1):**

| cell | mean trust |
|---|---|
| weak_plain | 0.960 |
| weak_elegant | 0.968 |
| strong_plain | 0.974 |
| strong_elegant | 0.979 |

**Effects (mean, 95% CI):**

| effect | mean | 95% CI | clears floor (0.02) & CI excludes 0? |
|---|---|---|---|
| **evidence effect** *(instrument check)* | **+0.012** | **[−0.020, +0.049]** | **NO** |
| elegance effect (seduction gap) | +0.006 | [−0.029, +0.035] | no |
| seduction@weak | +0.007 | [−0.060, +0.059] | no |

```
instrument_ok     : false
seduction_detected: false
```

### Why the instrument failed: a ceiling

The rater does not reliably trust strong evidence more than weak (evidence effect
+0.012, CI crosses 0). The cause is visible in the raw readings: **86% of the 64
trust values are ≥ 0.95; the median is 0.995.** The 1.5B instruct model answers
"Yes, well-supported" to almost everything, regardless of evidence quality. With
the readout pinned against 1.0 there is no dynamic range left for *any* effect —
evidence or elegance — to register above noise. The instrument is saturated.

---

## 4. Honest caveats (these are real results too)

**(a) The kill is "at this instrument," not "the failure mode isn't real."** The
seduction gap is a plausible, important pathology; this run shows only that *this
rater cannot measure it*. Absence of a measurable effect on a saturated instrument
is not evidence of absence in capable raters or humans.

**(b) Escalating to a larger model on GPU was declined — on purpose.** The
commission authorized modest Modal GPU for exactly this escalation, and the results
file's own note suggests "escalate to a larger rater." I did **not** run it, for a
reason the data forces: the failure is **ceiling saturation in the readout**, not a
shortfall of model capacity. Larger instruct models are RLHF-tuned to be *more*
agreeable, which makes a "Yes"-pinned readout **worse**, not better. Spending GPU to
re-roll the rater until an effect appears is model-shopping — p-hacking with a
bigger checkbook — and it violates the governing rule (*kill honestly; don't
manufacture a survival*). Modal is also not configured in this environment, so the
"escalation" would have meant autonomously standing up credentials and a pipeline
to fund a run my own diagnosis predicts is futile. Declined, flagged, not skipped.

**(c) What would actually resurrect this province** (either, not a GPU bump):
   1. **Readout redesign to break the ceiling** — graded/forced-choice confidence
      instead of a binary Yes/No first-token probability, and claims engineered to
      sit near genuine 50/50 uncertainty rather than near-certain trivia. Get the
      instrument off the ceiling *first*, then the evidence check can pass and the
      elegance effect becomes interpretable.
   2. **A small human rater study** — the trust-vs-evidence delta measured on people,
      which is the gold standard for a claim-level epistemic effect. Scope it small;
      do not fake it with the model judging itself.

**(d) Length is not the confound here.** The elegant cells are longer, so a naive
positive elegance effect could be a length artifact — but the elegance effect is
null anyway, and (more decisively) the instrument failed upstream, so there is no
effect to attribute. The length control matters only once the instrument is fixed.

---

## 5. Verdict: is the Greece / seduction lens a proven province?

**No — it is KILLED at this instrument, and that is a clean methodological win.**
The lens carried its own falsifier (the instrument check), the falsifier fired, and
the lens reported void instead of a flattering number. This is the first province
**vetoed by its own test** — precisely the canon precedent the methodology was
built to produce: *measurement refusing to certify a beautiful idea.* The province
is not dead forever; it is parked behind a concrete, falsifiable bar — fix the
readout ceiling (or run human raters), then re-run. Until then it does not enter
the field guide as proven.

---

## 6. Reproduce

```bash
# local MLX rater (Apple Silicon), no GPU/Modal:
pip install mlx-lm numpy
python3 atlas/greece_seduction.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --json atlas/results-greece-seduction.json
```

**Files:**
- `atlas/greece_seduction.py` — the diagnostic (reusable, documented; instrument-check-first).
- `atlas/results-greece-seduction.json` — machine-readable results of the live run.
- `atlas/REPORT-greece-seduction.md` — this report.

No existing experiment code or results files were modified.
