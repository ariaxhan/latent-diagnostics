"""
Atlas Diagnostics — Province 2: The Greece / Seduction-Gap Lens
===============================================================

A reusable diagnostic that asks one question of any model acting as an evaluator:

    "Is this evaluator's trust being earned by EVIDENCE, or seduced by FORM?
     When the same claim is dressed in fluent, elegant prose instead of flat
     plain prose -- with NO change in the actual evidence -- does the model
     trust it more? That extra trust is marble, not work."

The named failure mode is the *seduction gap* (the "Greece / rhetoric" lens,
after the sophists' logos): elegant explanation raises confidence independent of
evidence strength. An LLM-as-judge, a RAG confidence head, any model that scores
how well-supported a claim is, can be rhetorically captured -- rewarding the
arrangement of marble over the doing of work.

THE DESIGN (a 2x2 factorial trust lab)
--------------------------------------
For each base claim we hold the EVIDENCE fixed and vary two things independently:

  EVIDENCE  in {weak, strong}   weak = vague appeal, no mechanism/number/source.
                                strong = specific mechanism + quantity + source.
  STYLE     in {plain, elegant} INVARIANT: within an evidence level the elegant
                                cell carries the SAME facts as the plain cell and
                                differs ONLY in prose register. Elegance adds
                                cadence and vocabulary, never a fact.

Four cells per item: weak_plain, weak_elegant, strong_plain, strong_elegant.

TRUST READOUT (logit-based, not the model grading a number)
-----------------------------------------------------------
The model is shown claim + explanation and asked one yes/no question:
"Is the claim well-supported by this explanation?" We do NOT ask it to emit a
score (a 2B model's self-reported number is noise). Instead we read the
probability mass it puts on "Yes" vs "No" as the very next token:

    trust = P(Yes) / (P(Yes) + P(No))   in [0, 1]

This is a calibrated, deterministic readout of the model's belief, not a
self-assessment it could rationalize.

THE TWO CHANNELS WE DECOMPOSE
-----------------------------
  EVIDENCE EFFECT  = mean trust(strong) - mean trust(weak)
                     -> the "real work" channel. This is also the INSTRUMENT
                        CHECK: if the rater does not trust strong evidence more
                        than weak, it is not tracking evidence at all and the
                        whole test is void at this model size.
  ELEGANCE EFFECT  = mean trust(elegant) - mean trust(plain)
                     -> the SEDUCTION GAP. Trust bought by pure form. This is the
                        thing the province exists to detect.
  SEDUCTION@WEAK   = trust(weak_elegant) - trust(weak_plain)
                     -> marble laid over nothing: elegance applied to weak
                        evidence. The sharpest form of the failure.

WHY THIS IS A *PROOF OF METHOD*, NOT JUST A FORM-DETECTOR
--------------------------------------------------------
Mirroring the China province, a trustworthy lens must do BOTH:
  * CLEAR the real: show the rater DOES respond to evidence (evidence effect > 0,
    CI excludes 0) -- otherwise we are measuring a broken instrument, and a
    "seduction gap" would be indistinguishable from noise.
  * FLAG the marble: isolate a trust boost from elegance that is separable from
    evidence (elegance effect CI excludes 0).
If both hold on the same data, the lens separates "doing work" from "arranging
marble." If the instrument check fails, the honest verdict is KILL-at-this-model
(escalate to a larger rater or a human rater study), NOT a faked seduction number.

MODEL-AGNOSTIC
--------------
The lens scores trust through a pluggable `TrustScorer`. The default is a local,
ungated, Apple-MLX small instruct model so the test runs on this machine with no
gated weights and no cloud GPU. The Atlas seed names Gemma-2-2B via the repo's
Modal pipeline; that path is gated (HF token) and unconfigured in this
environment, so we run the IDENTICAL protocol on an ungated local rater and state
the substitution loudly. Swapping in Gemma is a one-function change (see
GemmaModalScorer stub at the bottom): the lens, fixtures, and verdict logic are
unchanged.

USAGE
-----
    python atlas/greece_seduction.py
    python atlas/greece_seduction.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        --json atlas/results-greece-seduction.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "atlas" / "data" / "greece_claims.json"
DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

CELLS = ("weak_plain", "weak_elegant", "strong_plain", "strong_elegant")

# Verdict thresholds (explicit + justified).
# Trust is in [0,1]; effects are differences of trust, so on the same scale.
INSTRUMENT_FLOOR = 0.02   # evidence effect must clear this (and CI exclude 0) or
                          # the rater is not tracking evidence -> test void.
SEDUCTION_FLOOR = 0.02    # elegance effect must clear this (and CI exclude 0) to
                          # count as a detected seduction gap.
N_BOOT = 10000
BOOT_SEED = 0


# ============================================================
# TRUST SCORER (pluggable)
# ============================================================

class TrustScorer(Protocol):
    def trust(self, claim: str, explanation: str) -> float:
        """Return P(Yes)/(P(Yes)+P(No)) that the claim is well-supported."""
        ...


SYSTEM_PROMPT = (
    "You evaluate whether a claim is well-supported by the explanation that "
    "follows it. Judge only the strength of the evidence and reasoning, not the "
    "writing style. Answer with exactly one word: \"Yes\" if the claim is "
    "well-supported by the explanation, or \"No\" if it is not."
)


def _user_prompt(claim: str, explanation: str) -> str:
    return (
        f"Claim: {claim}\n\n"
        f"Explanation: {explanation}\n\n"
        "Is the claim well-supported by this explanation? Answer Yes or No."
    )


class MLXScorer:
    """Local, ungated trust scorer via Apple MLX. Reads P(Yes)/P(No) on the
    first answer token. Deterministic (no sampling)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from mlx_lm import load  # imported lazily so the file parses without mlx
        import mlx.core as mx

        self._mx = mx
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

        def _first_ids(surfaces: list[str]) -> list[int]:
            ids = set()
            for s in surfaces:
                toks = self.tokenizer.encode(s, add_special_tokens=False)
                if toks:
                    ids.add(toks[0])
            return sorted(ids)

        self.yes_ids = _first_ids(["Yes", " Yes", "yes", " yes", "YES"])
        self.no_ids = _first_ids(["No", " No", "no", " no", "NO"])
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("could not resolve Yes/No token ids for this tokenizer")

    def trust(self, claim: str, explanation: str) -> float:
        mx = self._mx
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(claim, explanation)},
        ]
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        logits = self.model(mx.array([ids]))           # [1, seq, vocab]
        last = logits[0, -1, :]
        probs = mx.softmax(last.astype(mx.float32))
        probs_np = np.array(probs)                      # to host
        p_yes = float(probs_np[self.yes_ids].sum())
        p_no = float(probs_np[self.no_ids].sum())
        denom = p_yes + p_no
        if denom <= 0:
            return float("nan")
        return p_yes / denom


# ============================================================
# DIAGNOSTIC
# ============================================================

@dataclass
class ItemTrust:
    id: str
    weak_plain: float
    weak_elegant: float
    strong_plain: float
    strong_elegant: float

    def evidence_effect(self) -> float:
        return (self.strong_plain + self.strong_elegant) / 2 - \
               (self.weak_plain + self.weak_elegant) / 2

    def elegance_effect(self) -> float:
        return (self.weak_elegant + self.strong_elegant) / 2 - \
               (self.weak_plain + self.strong_plain) / 2

    def seduction_at_weak(self) -> float:
        return self.weak_elegant - self.weak_plain


def _boot_ci(values: np.ndarray, stat=np.mean, n_boot: int = N_BOOT,
             seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Mean + 95% bootstrap CI by resampling items."""
    rng = np.random.default_rng(seed)
    n = len(values)
    point = float(stat(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat(values[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def run_diagnostic(items: list[dict], scorer: TrustScorer) -> dict:
    trusts: list[ItemTrust] = []
    for it in items:
        c = it["cells"]
        trusts.append(ItemTrust(
            id=it["id"],
            weak_plain=scorer.trust(it["claim"], c["weak_plain"]),
            weak_elegant=scorer.trust(it["claim"], c["weak_elegant"]),
            strong_plain=scorer.trust(it["claim"], c["strong_plain"]),
            strong_elegant=scorer.trust(it["claim"], c["strong_elegant"]),
        ))

    cell_means = {cell: float(np.mean([getattr(t, cell) for t in trusts])) for cell in CELLS}

    ev = np.array([t.evidence_effect() for t in trusts])
    el = np.array([t.elegance_effect() for t in trusts])
    sw = np.array([t.seduction_at_weak() for t in trusts])

    ev_pt, ev_lo, ev_hi = _boot_ci(ev)
    el_pt, el_lo, el_hi = _boot_ci(el)
    sw_pt, sw_lo, sw_hi = _boot_ci(sw)

    # Instrument check: does the rater respond to evidence?
    instrument_ok = (ev_pt > INSTRUMENT_FLOOR) and (ev_lo > 0)
    # Seduction detected: is there a trust boost from elegance, separable from evidence?
    seduction_detected = (abs(el_pt) > SEDUCTION_FLOOR) and (el_lo > 0 or el_hi < 0)

    # Verdict.
    if not instrument_ok:
        verdict = "VOID-AT-THIS-MODEL"
        survives = False
        reason = (
            f"instrument check FAILED: the rater does not reliably trust strong "
            f"evidence more than weak (evidence effect {ev_pt:+.3f}, 95% CI "
            f"[{ev_lo:+.3f}, {ev_hi:+.3f}]). A seduction gap measured on an "
            f"evidence-blind rater is indistinguishable from noise. KILL at this "
            f"model size; escalate to a larger rater or a human rater study before "
            f"claiming the province."
        )
    elif seduction_detected and el_pt > 0:
        gap_pct = el_pt / ev_pt * 100 if ev_pt > 0 else float("nan")
        verdict = "SURVIVES"
        survives = True
        reason = (
            f"the lens SEPARATES work from marble. The rater responds to evidence "
            f"(effect {ev_pt:+.3f}, CI [{ev_lo:+.3f}, {ev_hi:+.3f}]) AND grants "
            f"extra trust to elegance alone (seduction gap {el_pt:+.3f}, CI "
            f"[{el_lo:+.3f}, {el_hi:+.3f}]) -- with identical evidence. Elegance "
            f"buys {gap_pct:.0f}% of what real evidence buys. The seduction gap is "
            f"real and measurable; the province is proven on this rater."
        )
    else:
        verdict = "NO-SEDUCTION-DETECTED"
        survives = False
        reason = (
            f"the rater tracks evidence (effect {ev_pt:+.3f}, CI [{ev_lo:+.3f}, "
            f"{ev_hi:+.3f}]) but shows NO trust boost from elegance (effect "
            f"{el_pt:+.3f}, CI [{el_lo:+.3f}, {el_hi:+.3f}], CI includes 0). On "
            f"this rater there is no seduction gap to detect. Honest null: the "
            f"province is not proven here -- this model is not seduced by form. "
            f"Re-run on a larger rater before promoting or killing."
        )

    return {
        "lens": "greece_seduction",
        "province": 2,
        "axis": "I (claim-level epistemic diagnostic)",
        "model": getattr(scorer, "model_name", "unknown"),
        "n_items": len(trusts),
        "trust_readout": "P(Yes)/(P(Yes)+P(No)) on first answer token, deterministic",
        "cell_means": cell_means,
        "effects": {
            "evidence_effect": {"mean": ev_pt, "ci95": [ev_lo, ev_hi],
                                "note": "real-work channel + instrument check"},
            "elegance_effect_SEDUCTION_GAP": {"mean": el_pt, "ci95": [el_lo, el_hi],
                                "note": "trust bought by pure form (identical evidence)"},
            "seduction_at_weak": {"mean": sw_pt, "ci95": [sw_lo, sw_hi],
                                "note": "marble over nothing: elegance on weak evidence"},
        },
        "thresholds": {
            "instrument_floor": INSTRUMENT_FLOOR,
            "seduction_floor": SEDUCTION_FLOOR,
            "rule": ("instrument_ok = evidence effect > floor AND CI>0; "
                     "seduction_detected = |elegance effect| > floor AND CI excludes 0"),
        },
        "proof_of_method": {
            "instrument_ok": instrument_ok,
            "seduction_detected": bool(seduction_detected and el_pt > 0),
        },
        "verdict": {"result": verdict, "survives": survives, "reason": reason},
        "items": [asdict(t) for t in trusts],
    }


# ============================================================
# MAIN
# ============================================================

def load_items(path: Path = DATA_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["items"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    items = load_items(args.data)
    print(f"Loading rater: {args.model} ...")
    scorer = MLXScorer(args.model)
    print(f"Scoring {len(items)} items x 4 cells = {len(items)*4} trust readouts ...")
    result = run_diagnostic(items, scorer)

    cm = result["cell_means"]
    ef = result["effects"]
    print("=" * 78)
    print("ATLAS DIAGNOSTICS — PROVINCE 2: GREECE / SEDUCTION-GAP LENS")
    print("=" * 78)
    print(f"Rater     : {result['model']}")
    print(f"Items     : {result['n_items']}  (trust = P(Yes)/(P(Yes)+P(No)), deterministic)")
    print("-" * 78)
    print("2x2 CELL MEAN TRUST  (rows = evidence, cols = style)")
    print(f"{'':<10}{'plain':>12}{'elegant':>12}{'elegance gap':>16}")
    print(f"{'weak':<10}{cm['weak_plain']:>12.3f}{cm['weak_elegant']:>12.3f}"
          f"{cm['weak_elegant']-cm['weak_plain']:>+16.3f}")
    print(f"{'strong':<10}{cm['strong_plain']:>12.3f}{cm['strong_elegant']:>12.3f}"
          f"{cm['strong_elegant']-cm['strong_plain']:>+16.3f}")
    print("-" * 78)
    print("EFFECTS (mean [95% bootstrap CI])")
    e = ef["evidence_effect"]; print(f"  evidence effect (real work)   : {e['mean']:+.3f}  [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")
    s = ef["elegance_effect_SEDUCTION_GAP"]; print(f"  elegance effect (SEDUCTION GAP): {s['mean']:+.3f}  [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    w = ef["seduction_at_weak"]; print(f"  seduction @ weak (marble/void): {w['mean']:+.3f}  [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    print("-" * 78)
    pm = result["proof_of_method"]
    print(f"PROOF CHECK   instrument_ok={pm['instrument_ok']}  seduction_detected={pm['seduction_detected']}")
    print("-" * 78)
    print(f"VERDICT: {result['verdict']['result']}  (survives={result['verdict']['survives']})")
    print(f"  {result['verdict']['reason']}")
    print("=" * 78)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote: {args.json}")


# ----------------------------------------------------------------
# Swap target: run the IDENTICAL protocol on Gemma-2-2B via the repo's Modal
# pipeline once an HF token + Modal HF secret are configured. Only this class
# changes; fixtures, decomposition, thresholds, and verdict logic are untouched.
# ----------------------------------------------------------------
class GemmaModalScorer:  # pragma: no cover - stub, needs Modal + HF secret
    """Stub. Implement .trust() by reading P('Yes')/P('No') first-token logits
    from gemma-2-2b on Modal (see scripts/modal_app.py for the A10G + HF-secret
    container pattern). Wire the HF token into the Modal secret, then point
    --model at it. The seduction-gap protocol does not change."""
    model_name = "gemma-2-2b (Modal) — not configured in this environment"

    def trust(self, claim: str, explanation: str) -> float:
        raise NotImplementedError(
            "GemmaModalScorer needs a configured Modal app + HF secret. "
            "Use MLXScorer locally, or implement this against scripts/modal_app.py."
        )


if __name__ == "__main__":
    main()
