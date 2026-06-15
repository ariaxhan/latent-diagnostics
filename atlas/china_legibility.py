"""
Atlas Diagnostics — Province 1: The China / Legibility Lens
============================================================

A reusable diagnostic that asks one question of any candidate metric:

    "Is this metric measuring reality, or is it measuring a *legible proxy*
     that someone (the data-collection pipeline, the model, the world) has
     made easy to read off?"

The named failure mode is *administered reality* (the "China / legibility"
lens, after Scott's *Seeing Like a State*): a system optimizes a metric that
looks like signal but is really tracking a confound the apparatus made
conveniently measurable. Here the confound is **text length**. Longer inputs
mechanically light up more SAE features, produce more attribution edges, etc.
A metric that merely re-encodes "how long was the prompt" is *legible* — it is
not telling you anything about the class you actually care about.

THE DIAGNOSTIC
--------------
For a chosen binary class comparison and a chosen confound (text length):

  1. RAW EFFECT      Cohen's d between the two classes on the raw metric.
  2. CONFOUND LINK   Pearson r between the metric and the confound (length).
  3. RESIDUALIZE     Regress the metric on length (OLS, degree 1), keep the
                     residuals, and recompute Cohen's d on those residuals.
  4. CLASSIFY        If the effect survives residualization -> REAL SIGNAL.
                     If it collapses                       -> LEGIBILITY ARTIFACT.

WHY THIS IS A *PROOF OF METHOD*, NOT JUST A FILTER
--------------------------------------------------
A diagnostic that only ever says "fake" is useless — it would condemn real
signal too. To be trustworthy it must do BOTH:

    * correctly FLAG a metric we already know is a length artifact, and
    * correctly CLEAR a metric we already know carries real, length-independent
      structure.

If both happen on the same data with the same threshold, the lens is calibrated:
it discriminates legibility from reality rather than just punishing big raw
effects.

COLLAPSE THRESHOLD (explicit + justified)
-----------------------------------------
A metric is a LEGIBILITY ARTIFACT if EITHER:

    (a) the residualized |d| falls below ABS_FLOOR (default 0.30), i.e. the
        effect is no longer even a "small" effect by Cohen's conventions
        (0.2 small / 0.5 medium / 0.8 large); once you remove length there is
        essentially nothing left to see; OR

    (b) the effect shrinks by more than RETAIN_DROP of its raw magnitude
        (default 70%), i.e. residualized |d| < 0.30 * raw |d|. This catches
        metrics whose raw effect was almost entirely length even if the raw
        effect was so enormous that 30% of it still clears the absolute floor.

Both conditions are deliberately conservative: we would rather let a borderline
metric through as "real" than wrongly condemn genuine signal. A metric is
REAL SIGNAL only if it clears BOTH bars.

USAGE
-----
    python atlas/china_legibility.py
    python atlas/china_legibility.py --json out.json

This is pure analysis of already-computed attribution results. No GPU, no model
runs, no new data. It does not modify any existing experiment code or results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "results" / "domain_attribution_metrics.json"

# The class comparison. The validated binary contrast in this dataset is
# task-type. grammar (CoLA) vs commonsense (HellaSwag) is the cleanest
# proof case: it contains a metric known to be pure length (n_active) AND
# metrics known to carry real computational-focus structure (influence,
# concentration). See REPORT-china-legibility.md for why this pair.
CLASS_FIELD = "domain"
CLASS_A = "grammar"
CLASS_B = "commonsense"

# The confound we residualize against.
LENGTH_FROM = "text"  # character length of this field

# Candidate metrics to audit.
METRICS: Sequence[str] = (
    "n_active",
    "n_edges",
    "mean_influence",
    "max_influence",
    "top_100_concentration",
)

# Collapse threshold (see module docstring).
ABS_FLOOR = 0.30      # residualized |d| below this -> artifact
RETAIN_DROP = 0.70    # if effect loses more than this fraction -> artifact


# ============================================================
# CORE STATISTICS
# ============================================================

def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Pooled-standard-deviation Cohen's d between two samples.

    Matches the convention used elsewhere in this repo
    (experiments/core/existing_data_analysis.py): pooled SD with (n-1)
    weighting. Sign is (mean(x1) - mean(x2)); only magnitude is used for
    classification.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = (
        (n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1)
    ) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    return float((x1.mean() - x2.mean()) / (pooled_sd + 1e-12))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN-safe for degenerate input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residualize(metric: np.ndarray, confound: np.ndarray) -> np.ndarray:
    """Regress `metric` on `confound` (OLS degree 1), return residuals.

    Uses np.polyfit degree 1, matching experiments/statistics/bootstrap_ci.py.
    Residuals are the part of the metric NOT explained by the confound — i.e.
    what is left once the legible proxy has been subtracted out.
    """
    metric = np.asarray(metric, dtype=float)
    confound = np.asarray(confound, dtype=float)
    coef = np.polyfit(confound, metric, 1)
    predicted = np.polyval(coef, confound)
    return metric - predicted


# ============================================================
# DIAGNOSTIC
# ============================================================

@dataclass
class MetricVerdict:
    metric: str
    raw_d: float
    r_with_length: float
    resid_d: float
    pct_retained: float        # |resid_d| / |raw_d|
    verdict: str               # "LEGIBILITY ARTIFACT" | "REAL SIGNAL"
    reason: str


def classify(metric: str, raw_d: float, r_len: float, resid_d: float) -> MetricVerdict:
    """Apply the collapse threshold to one metric."""
    abs_raw = abs(raw_d)
    abs_resid = abs(resid_d)
    pct_retained = abs_resid / abs_raw if abs_raw > 1e-12 else float("nan")

    below_floor = abs_resid < ABS_FLOOR
    collapsed = (not np.isnan(pct_retained)) and pct_retained < (1.0 - RETAIN_DROP)

    if below_floor or collapsed:
        verdict = "LEGIBILITY ARTIFACT"
        bits = []
        if below_floor:
            bits.append(f"residual |d|={abs_resid:.2f} < floor {ABS_FLOOR}")
        if collapsed:
            bits.append(
                f"retained {pct_retained:.0%} (< {1.0 - RETAIN_DROP:.0%})"
            )
        reason = "collapses after length-residualization: " + "; ".join(bits)
    else:
        verdict = "REAL SIGNAL"
        reason = (
            f"survives: residual |d|={abs_resid:.2f} >= floor {ABS_FLOOR} "
            f"and retained {pct_retained:.0%} of raw effect"
        )

    return MetricVerdict(
        metric=metric,
        raw_d=round(raw_d, 4),
        r_with_length=round(r_len, 4),
        resid_d=round(resid_d, 4),
        pct_retained=round(pct_retained, 4) if not np.isnan(pct_retained) else float("nan"),
        verdict=verdict,
        reason=reason,
    )


def run_diagnostic(
    samples: list[dict],
    metrics: Sequence[str] = METRICS,
    class_field: str = CLASS_FIELD,
    class_a: str = CLASS_A,
    class_b: str = CLASS_B,
    length_from: str = LENGTH_FROM,
) -> dict:
    """Run the China/legibility lens over every candidate metric.

    RESIDUALIZATION BASIS (important — see REPORT for the full argument):
    the length->metric law is fit on the ENTIRE sample population (every
    class present in `samples`), NOT just the two classes being compared.
    Then class differences are measured on those global residuals.

    Why global, not within-pair? In this dataset the classes are themselves
    length-stratified (e.g. grammar prompts are short, commonsense prompts
    are long: length Cohen's d ~3.6 between them). If you fit the length
    regression using only those two classes, length and class are nearly
    collinear and the regression soaks up the *real* class signal as if it
    were length, collapsing everything to ~0 and over-flagging. Fitting the
    confound law on the full population (which spans a wide length range from
    other classes too) gives an honest, class-agnostic length->metric curve,
    so what survives in the residuals is genuine class structure rather than
    an artifact of collinearity. This basis choice is the difference between
    the lens working and the lens lying; it is stated explicitly here.
    """
    rows = [s for s in samples if s.get(class_field) in (class_a, class_b)]
    if not rows:
        raise ValueError(
            f"No samples found for {class_field} in {{{class_a}, {class_b}}}"
        )

    # Global confound basis is fit per-metric below (over the full population).
    labels = np.array([s[class_field] for s in rows])
    mask_a = labels == class_a
    mask_b = labels == class_b
    pair_length = np.array(
        [len(str(s.get(length_from, ""))) for s in rows], dtype=float
    )

    verdicts: list[MetricVerdict] = []
    for m in metrics:
        if not all(m in s for s in rows):
            continue
        # Fit length->metric on the full population, then apply to the pair.
        global_with_metric = [s for s in samples if m in s and s.get(length_from) is not None]
        gl = np.array([len(str(s[length_from])) for s in global_with_metric], dtype=float)
        gv = np.array([float(s[m]) for s in global_with_metric], dtype=float)
        coef = np.polyfit(gl, gv, 1)

        vals = np.array([float(s[m]) for s in rows], dtype=float)
        raw_d = cohens_d(vals[mask_a], vals[mask_b])
        r_len = pearson_r(gl, gv)  # confound strength on full population
        resid = vals - np.polyval(coef, pair_length)
        resid_d = cohens_d(resid[mask_a], resid[mask_b])

        verdicts.append(classify(m, raw_d, r_len, resid_d))

    return {
        "lens": "china_legibility",
        "province": 1,
        "comparison": f"{class_a} vs {class_b}",
        "confound": f"{length_from} length (chars)",
        "n_class_a": int(mask_a.sum()),
        "n_class_b": int(mask_b.sum()),
        "threshold": {
            "abs_floor": ABS_FLOOR,
            "retain_drop": RETAIN_DROP,
            "rule": (
                f"artifact if residual |d| < {ABS_FLOOR} "
                f"OR effect retains < {1.0 - RETAIN_DROP:.0%} of raw magnitude"
            ),
        },
        "verdicts": [asdict(v) for v in verdicts],
    }


# ============================================================
# PROOF CHECK
# ============================================================

def check_proof(result: dict) -> dict:
    """A proof-of-method requires the lens to FLAG a known fake AND CLEAR
    known real metrics on the SAME data with the SAME threshold.

    CORE PROOF (the falsifiable claim):
      - n_active        -> LEGIBILITY ARTIFACT   (the canonical length proxy;
                            r(len)=0.98, perfectly linear -> must collapse)
      - mean_influence  -> REAL SIGNAL           (computational focus; must survive)
      - top_100_concentration -> REAL SIGNAL     (must survive)

    WATCHED (reported, not part of pass/fail — see REPORT for the n_edges
    caveat about linear residualization vs a super-linear confound):
      - n_edges
    """
    by_metric = {v["metric"]: v["verdict"] for v in result["verdicts"]}

    must_flag = ["n_active"]
    must_clear = ["mean_influence", "top_100_concentration"]
    watched = ["n_edges", "max_influence"]

    flagged_ok = {
        m: by_metric.get(m) == "LEGIBILITY ARTIFACT"
        for m in must_flag if m in by_metric
    }
    cleared_ok = {
        m: by_metric.get(m) == "REAL SIGNAL"
        for m in must_clear if m in by_metric
    }
    watched_verdicts = {m: by_metric.get(m) for m in watched if m in by_metric}

    proof_passed = all(flagged_ok.values()) and all(cleared_ok.values())
    return {
        "flagged_fake": flagged_ok,
        "cleared_real": cleared_ok,
        "watched": watched_verdicts,
        "proof_passed": proof_passed,
    }


# ============================================================
# MAIN
# ============================================================

def load_samples(path: Path = DATA_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["samples"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--json", type=Path, default=None,
                        help="optional path to dump full results as JSON")
    args = parser.parse_args()

    samples = load_samples(args.data)
    result = run_diagnostic(samples)
    proof = check_proof(result)
    result["proof_check"] = proof

    # Console table
    print("=" * 78)
    print("ATLAS DIAGNOSTICS — PROVINCE 1: CHINA / LEGIBILITY LENS")
    print("=" * 78)
    print(f"Comparison : {result['comparison']} "
          f"(n={result['n_class_a']} vs {result['n_class_b']})")
    print(f"Confound   : {result['confound']}")
    print(f"Threshold  : {result['threshold']['rule']}")
    print("-" * 78)
    hdr = f"{'metric':<24}{'raw d':>9}{'r(len)':>9}{'resid d':>10}{'kept':>7}  verdict"
    print(hdr)
    print("-" * 78)
    for v in result["verdicts"]:
        kept = "n/a" if v["pct_retained"] != v["pct_retained"] else f"{v['pct_retained']*100:.0f}%"
        print(f"{v['metric']:<24}{v['raw_d']:>+9.2f}{v['r_with_length']:>+9.2f}"
              f"{v['resid_d']:>+10.2f}{kept:>7}  {v['verdict']}")
    print("-" * 78)
    print("PROOF CHECK (core claim)")
    for m, ok in proof["flagged_fake"].items():
        print(f"  flag  {m:<22} {'PASS' if ok else 'FAIL'} (expected LEGIBILITY ARTIFACT)")
    for m, ok in proof["cleared_real"].items():
        print(f"  clear {m:<22} {'PASS' if ok else 'FAIL'} (expected REAL SIGNAL)")
    if proof.get("watched"):
        print("  watched (reported, not pass/fail):")
        for m, verdict in proof["watched"].items():
            print(f"    {m:<22} {verdict}")
    print("-" * 78)
    print(f"PROOF {'PASSED' if proof['proof_passed'] else 'FAILED'}: "
          f"the lens {'correctly flagged the fake AND cleared the real.' if proof['proof_passed'] else 'did NOT reproduce the expected split.'}")
    print("=" * 78)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote: {args.json}")


if __name__ == "__main__":
    main()
