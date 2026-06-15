# Atlas Diagnostics — Province 1: The China / Legibility Lens

**Status: core proof PASSED.** The diagnostic correctly FLAGGED the known length
artifact (`n_active`) and CLEARED the two known real signals (`mean_influence`,
`top_100_concentration`) on the same data with the same threshold.

This is the first province of a planned diagnostics methodology ("Atlas
Diagnostics"), run as a proof-of-method on existing data — no GPU, no new model
runs.

---

## 1. The failure mode this lens detects

A system mistakes a **legible proxy** for reality. A metric looks like signal but
is really tracking a confound that the measurement apparatus made conveniently
readable — *administered reality*, the "Seeing Like a State" / China-legibility
failure. Here the confound is **text length**: longer prompts mechanically light
up more SAE features and produce more attribution edges, so any metric that
merely re-encodes "how long was the input" will look discriminative without
telling you anything about the class you actually care about.

**The diagnostic.** For a binary class comparison, for each candidate metric:

1. **Raw effect** — Cohen's *d* between the two classes.
2. **Confound link** — Pearson *r* between the metric and length.
3. **Residualize** — regress the metric on length (OLS degree 1), recompute *d*
   on the residuals.
4. **Classify** — if the effect survives -> **REAL SIGNAL**; if it collapses ->
   **LEGIBILITY ARTIFACT**.

**Collapse threshold (explicit).** A metric is a LEGIBILITY ARTIFACT if EITHER
the residualized |*d*| < **0.30** (below Cohen's "small" floor — nothing left to
see) OR the effect retains < **30%** of its raw magnitude (it loses > 70% to
length). A metric is REAL SIGNAL only if it clears both bars. The thresholds are
deliberately conservative: we would rather wave a borderline metric through as
real than wrongly condemn genuine signal.

---

## 2. Data and comparison

- **File:** `data/results/domain_attribution_metrics.json` (210 samples, Gemma 2
  2B via circuit-tracer; all metrics pre-computed).
- **Class comparison:** `grammar` (CoLA, n=50) vs `commonsense` (HellaSwag,
  n=50). This pair is the cleanest proof case: it contains a metric we know
  is pure length (`n_active`) AND metrics we know carry real
  computational-focus structure (`mean_influence`, `top_100_concentration`).
- **Confound:** character length of the `text` field.

### A methodological choice that decides whether the lens works: GLOBAL residualization

The length->metric regression is fit on the **entire sample population** (all
domains, full length range), then applied to the two compared classes — **not**
fit on just the two classes.

This matters because the classes are themselves length-stratified:

| domain | n | mean length (chars) |
|---|---|---|
| grammar | 50 | 40 |
| commonsense | 50 | 106 |
| nli | 50 | 141 |
| paraphrase | 13 | 208 |

Grammar and commonsense barely overlap in length (length Cohen's *d* ~ **-3.6**).
If you fit the length regression using only those two classes, length and class
are nearly collinear: the regression soaks up the *real* class difference as if
it were length, and EVERYTHING collapses to ~0 (we measured `mean_influence`
residual *d* dropping to -0.13 and `top_100_concentration` to ~0 under the
within-pair fit). That would make the lens lie — it would flag genuine signal as
artifact.

Fitting the confound law on the full population gives an honest, class-agnostic
length->metric curve, anchored by the wide length spread of all four domains. What
then survives in the residuals is genuine class structure rather than an artifact
of collinearity. This basis choice is stated in the code (`run_diagnostic`
docstring) because it is load-bearing.

---

## 3. Results (real numbers, live run)

Command: `python atlas/china_legibility.py`
Raw output cached to `atlas/results-china-legibility.json`.

| metric | raw *d* | r(len) | residual *d* | % retained | verdict |
|---|---|---|---|---|---|
| **n_active** | -3.19 | +0.98 | **+0.55** | 17% | **LEGIBILITY ARTIFACT** |
| n_edges | -2.85 | +0.96 | +1.77 | 62% | REAL SIGNAL *(watched — see caveat)* |
| **mean_influence** | +2.09 | -0.80 | **+1.25** | 60% | **REAL SIGNAL** |
| max_influence | -0.35 | +0.42 | +0.14 | 41% | LEGIBILITY ARTIFACT *(never had a raw effect)* |
| **top_100_concentration** | +1.54 | -0.63 | **+0.95** | 62% | **REAL SIGNAL** |

### Did the lens flag the fake AND clear the real?

- **`n_active` — FLAGGED.** A huge raw effect (*d* = -3.19) that is 98% correlated
  with length and **perfectly linear** in it; linear residualization removes it
  almost entirely (residual *d* = +0.55, only 17% of the raw effect retained).
  Correctly condemned as administered reality. **PASS.**
- **`mean_influence` — CLEARED.** Raw *d* = +2.09 -> residual *d* = +1.25 (60%
  retained). Most of the effect is independent of length: real computational
  signal. **PASS.**
- **`top_100_concentration` — CLEARED.** Raw *d* = +1.54 -> residual *d* = +0.95
  (62% retained). Survives. **PASS.**

**The core proof holds: the same lens, same threshold, same data both correctly
condemns a length proxy and correctly vindicates genuine signal.** That two-sided
behavior — not merely punishing big raw effects — is what makes it a calibrated
diagnostic rather than a blunt filter.

---

## 4. Honest caveats (these are real results too)

**(a) `n_edges` survives the lens, but only because the confound is super-linear.**
We *expected* `n_edges` to be a length twin of `n_active` and to be flagged. It
isn't (residual *d* = +1.77). The reason is a genuine limitation of the method,
not of the data: `n_active` is linear in length (r=0.98 with length, 0.95 with
length^2), so a degree-1 regression removes it cleanly. `n_edges` grows
**super-linearly** — edges scale roughly with features^2, and it correlates with
length^2 (0.965) just as strongly as with length (0.963). A *linear* length model
structurally cannot absorb a super-linear confound, so a chunk of pure length
leaks through as a fake "residual." Evidence: the commonsense/grammar edge-count
ratio is 4.0x while their length ratio is only 2.6x — exactly the super-linear gap
the linear model misses. **Takeaway:** the lens is only as good as its confound
model. For metrics that scale non-linearly with the confound, the residualization
basis must be non-linear (e.g. regress on length AND length^2, or on log-counts).
This is the natural next refinement of the province, and it does not undermine the
core proof, which used metrics where a linear model is appropriate.

**(b) `max_influence` is flagged, but it never had a raw effect to begin with**
(raw *d* = -0.35). The verdict "artifact" is technically correct (it collapses to
+0.14) but uninformative — there was no signal to defend. The lens should arguably
report a third state, "no raw effect," rather than lumping never-signal in with
killed-signal. Noted for the methodology.

**(c) The within-comparison-vs-global residualization fork is the whole ballgame.**
On this dataset the within-pair basis would have produced the *opposite* verdict
(everything collapses, the lens cries wolf). We chose global residualization for a
principled reason (avoid length/class collinearity), and we are stating it loudly
because a future user running this lens on a dataset where classes are NOT
length-stratified could legitimately use the within-comparison basis. The right
basis is data-dependent; the lens should not hide that choice.

**(d) Sign note.** Verdicts use |*d*|; the sign of raw vs residual *d* can flip
(e.g. `n_active` -3.19 -> +0.55) when residualization over-corrects a near-perfect
length proxy. The flip itself is a tell that almost nothing survives.

---

## 5. Verdict: is the China / legibility lens a proven, runnable diagnostic?

**Yes, for its intended scope.** It is a single reusable script
(`atlas/china_legibility.py`) that runs in under a second on existing results,
needs no GPU, and on real data it does the thing a diagnostic must do to be
trustworthy: it **flags a metric that is secretly just length** (`n_active`) while
**clearing metrics that carry real, length-independent structure**
(`mean_influence`, `top_100_concentration`). A filter that only ever said "fake"
would have failed the clear-the-real half; this one passes both halves with one
threshold.

Its honest boundary: it is only as sharp as its confound model. With a linear
length model it nails linearly-confounded metrics and under-corrects
super-linearly-confounded ones (`n_edges`). That is a known, characterized
limitation with a clear fix (non-linear residualization), not a silent failure —
which is exactly what you want from a first province.

---

## 6. Reproduce

```bash
# one-time analysis env (numpy/pandas/scipy; gitignored, no GPU)
python3 -m venv .venv-atlas && .venv-atlas/bin/pip install numpy pandas scipy

# run the lens
.venv-atlas/bin/python atlas/china_legibility.py --json atlas/results-china-legibility.json
```

**Files:**
- `atlas/china_legibility.py` — the diagnostic (reusable, documented).
- `atlas/results-china-legibility.json` — machine-readable results of the live run.
- `atlas/REPORT-china-legibility.md` — this report.

No existing experiment code or results files were modified.
