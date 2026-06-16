"""Self-generation probe benchmark harness (D-A).

Skeleton implementing the evaluation side of protocol.md. The GPU-dependent
extraction step (Modal) plugs in via the SampleRecord schema below; everything
downstream (probe arms, cross-validation, transfer splits, reporting) runs
locally on cached records.

Pipeline:
    1. build_dataset()        -> data/self_generation_probe/questions.json
    2. [Modal] generate + extract -> data/results/self_generation_probe/records.json
    3. run_probes()           -> AUC table per arm/layer + transfer results

Pre-registered hypotheses and kill criteria: see protocol.md.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

LAYERS = (9, 12, 20)
GEOMETRY_METRICS = (
    "sphericity",
    "elongation",
    "eigenvalue_entropy",
    "effective_dimensionality",
    "misalignment_angle",
)
OUTPUT_BASELINES = ("max_logit_prob", "logit_entropy", "answer_mean_logprob")

# ---------------------------------------------------------------------------
# Pre-registered decision thresholds — protocol.md "Success / kill criteria",
# LOCKED 2026-06-12. The analysis layer below (bootstrap CIs + verdict()) was
# added 2026-06-15 as parked-prep: it IMPLEMENTS the locked registration so the
# run is one-command when GPU creds land. It does NOT tune thresholds. Authored
# before any data exists, so it cannot be fit to results (the anti-p-hack point).
# ---------------------------------------------------------------------------
H1_RESIDUAL_AUC = 0.75    # H1: residual-stream probe AUC >= 0.75 (in-domain)
H2_SAE_FRACTION = 0.90    # H2: top-k SAE (k<=64) reaches >= 90% of residual AUC
H2_MAX_K = 64
H3_GEOMETRY_TOL = 0.05    # H3: geometry metrics at chance, |AUC - 0.5| < 0.05
BOOT_RESAMPLES = 1000     # protocol: bootstrap CIs, 1000 resamples
BOOT_SEED = 0             # fixed seed -> deterministic CIs (reproducibility)


@dataclass
class SampleRecord:
    """One generated answer with everything the probe arms need.

    Produced by the Modal extraction script; consumed by run_probes().
    """

    question_id: str
    question: str
    entity: str
    popularity_bucket: str  # "high" | "low"
    generated_answer: str
    label: str  # "correct" | "incorrect" | "refusal"
    answer_token_count: int
    # arm A: residual stream at answer-entity tokens (mean-pooled), per layer
    residual: dict[str, list[float]] = field(default_factory=dict)  # {"9": [...], ...}
    # arm B: sparse SAE activations per layer: {feature_index: activation}
    sae_features: dict[str, dict[str, float]] = field(default_factory=dict)
    # arm C: geometry metrics on the SAE activation cloud, per layer
    geometry: dict[str, dict[str, float]] = field(default_factory=dict)
    # arm D: output-only baselines
    output_baselines: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1 — dataset construction (stub: wire to WikiData dump or static list)
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = [
    "In which year was {entity} founded?",
    "Who directed the film {entity}?",
    "In which country is {entity} located?",
    "Who wrote {entity}?",
    "What is the capital of {entity}?",
]


def build_dataset(
    entities_high: Iterable[tuple[str, str]],
    entities_low: Iterable[tuple[str, str]],
    out_path: Path,
    n_per_bucket: int = 250,
    seed: int = 0,
) -> None:
    """entities_*: iterables of (entity_name, gold_answer) pairs.

    Template distribution is balanced across buckets (protocol balance check).
    """
    rng = random.Random(seed)
    rows = []
    for bucket, entities in (("high", list(entities_high)), ("low", list(entities_low))):
        rng.shuffle(entities)
        for i, (entity, gold) in enumerate(entities[:n_per_bucket]):
            template = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
            rows.append(
                {
                    "question_id": f"{bucket}_{i:04d}",
                    "question": template.format(entity=entity),
                    "entity": entity,
                    "gold_answer": gold,
                    "popularity_bucket": bucket,
                    "template_idx": i % len(QUESTION_TEMPLATES),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} questions -> {out_path}")


# ---------------------------------------------------------------------------
# Shared statistics (no sklearn dependency required for the skeleton;
# swap fit_logistic for sklearn LogisticRegression when running for real)
# ---------------------------------------------------------------------------


def rank_auc(scores: list[float], labels: list[int]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += sum(avg for k in range(i, j) if labels[order[k]] == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def boot_ci(
    scores: list[float], labels: list[int],
    n_resamples: int = BOOT_RESAMPLES, seed: int = BOOT_SEED, alpha: float = 0.05,
) -> list[float]:
    """Percentile bootstrap 95% CI on rank AUC (protocol: 1000 resamples).

    Resamples the pooled out-of-fold (score, label) predictions. Deterministic
    via fixed seed. Resamples that lose a class are skipped.
    """
    if not scores or len(set(labels)) < 2:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(scores)
    stats = []
    for _ in range(n_resamples):
        samp = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in samp]
        y = [labels[i] for i in samp]
        if len(set(y)) < 2:
            continue
        a = rank_auc(s, y)
        if not math.isnan(a):
            stats.append(a)
    if not stats:
        return [float("nan"), float("nan")]
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return [lo, hi]


def _topk_from_key(key: str) -> int | None:
    """Parse k out of a 'B_sae_top{k}_L{layer}' result key."""
    try:
        return int(key.split("B_sae_top")[1].split("_L")[0])
    except (IndexError, ValueError):
        return None


def verdict(results: dict) -> dict:
    """Apply the pre-registered decision rule (protocol.md). READS results only.

    H1: best residual-arm AUC >= 0.75.
    H2: best SAE top-k (k<=64) AUC >= 90% of the best residual AUC.
    H3: every geometry-arm AUC within 0.05 of chance (control; reported, not gating).
    -> ALIVE / PIVOT_DROP_SAE / DEAD_AT_THIS_SCALE.

    NOTE: protocol also specifies a DeLong A-vs-B test; this stdlib skeleton
    reports per-arm bootstrap CIs and the H2 fraction rule instead. A true paired
    DeLong test should be added (scipy) at real-run time — flagged, not silently
    dropped.
    """
    def aucs_with_prefix(prefix):
        return [v["auc"] for kk, v in results.items()
                if kk.startswith(prefix) and isinstance(v, dict) and not math.isnan(v.get("auc", float("nan")))]

    res = aucs_with_prefix("A_residual_")
    best_res = max(res, default=float("nan"))
    sae = [v["auc"] for kk, v in results.items()
           if kk.startswith("B_sae_top") and isinstance(v, dict)
           and (_topk_from_key(kk) or 1e9) <= H2_MAX_K and not math.isnan(v.get("auc", float("nan")))]
    best_sae = max(sae, default=float("nan"))
    geo = aucs_with_prefix("C_geometry_")

    h1 = (not math.isnan(best_res)) and best_res >= H1_RESIDUAL_AUC
    h2 = (not math.isnan(best_sae)) and (not math.isnan(best_res)) and best_res > 0 \
        and best_sae >= H2_SAE_FRACTION * best_res
    h3 = bool(geo) and all(abs(a - 0.5) < H3_GEOMETRY_TOL for a in geo)

    if not h1:
        result = "DEAD_AT_THIS_SCALE"
        reason = (f"H1 refuted: best residual AUC {best_res:.3f} < {H1_RESIDUAL_AUC}. "
                  "No internals work at 2B; continuation moves to Gemma-2-9B or stops.")
    elif not h2:
        result = "PIVOT_DROP_SAE"
        reason = (f"H1 holds (residual {best_res:.3f}) but H2 fails "
                  f"(best SAE k<=64 {best_sae:.3f} < {H2_SAE_FRACTION:.0%} of residual). "
                  "SAEs don't concentrate the signal at 2B (cf. Kantamneni ICML'25); drop the SAE arm.")
    else:
        result = "ALIVE"
        reason = (f"H1 holds (residual {best_res:.3f}) and H2 holds (SAE {best_sae:.3f}). "
                  "Write up 'where the signal lives' + the HB-1000/TruthfulQA negative-result section.")
    return {
        "result": result,
        "H1_residual_ge_0.75": h1,
        "H2_sae_topk_ge_90pct_of_residual": h2,
        "H3_geometry_at_chance": h3,
        "best_residual_auc": best_res,
        "best_sae_topk_le64_auc": best_sae,
        "reason": reason,
    }


def residualize(values: list[float], covariate: list[float]) -> list[float]:
    """Regress out a covariate (length control — house rule from the r=0.98 lesson)."""
    n = len(values)
    mx, my = st.mean(covariate), st.mean(values)
    sxx = sum((x - mx) ** 2 for x in covariate)
    if sxx == 0:
        return [v - my for v in values]
    beta = sum((x - mx) * (y - my) for x, y in zip(covariate, values)) / sxx
    return [y - (my + beta * (x - mx)) for x, y in zip(covariate, values)]


def fit_logistic(
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    epochs: int = 300,
    lr: float = 0.1,
    l2: float = 1e-2,
) -> list[float]:
    """Minimal ridge-logistic via gradient descent. Replace with sklearn for real runs."""
    if not X_train:
        return []
    dim = len(X_train[0])
    # standardize using train stats
    mus = [st.mean(col) for col in zip(*X_train)]
    sds = [st.stdev(col) if len(set(col)) > 1 else 1.0 for col in zip(*X_train)]
    norm = lambda row: [(v - m) / (s or 1.0) for v, m, s in zip(row, mus, sds)]
    Xtr = [norm(r) for r in X_train]
    Xte = [norm(r) for r in X_test]
    w = [0.0] * dim
    b = 0.0
    n = len(Xtr)
    for _ in range(epochs):
        gw = [l2 * wi for wi in w]
        gb = 0.0
        for row, y in zip(Xtr, y_train):
            z = b + sum(wi * xi for wi, xi in zip(w, row))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            err = p - y
            for d in range(dim):
                gw[d] += err * row[d] / n
            gb += err / n
        w = [wi - lr * gi for wi, gi in zip(w, gw)]
        b -= lr * gb
    return [b + sum(wi * xi for wi, xi in zip(w, row)) for row in Xte]


def select_contrastive_features(
    records: list[SampleRecord], layer: int, k: int
) -> list[str]:
    """Top-k SAE features by |mean act difference| (incorrect - correct), TRAIN ONLY."""
    sums: dict[str, list[float]] = {}
    counts = {"correct": 0, "incorrect": 0}
    for r in records:
        if r.label not in counts:
            continue
        counts[r.label] += 1
        for feat, act in r.sae_features.get(str(layer), {}).items():
            sums.setdefault(feat, [0.0, 0.0])
            sums[feat][0 if r.label == "correct" else 1] += act
    diffs = {
        feat: abs(v[1] / max(counts["incorrect"], 1) - v[0] / max(counts["correct"], 1))
        for feat, v in sums.items()
    }
    return sorted(diffs, key=diffs.get, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Step 3 — probe arms and evaluation
# ---------------------------------------------------------------------------


def featurize(record: SampleRecord, arm: str, layer: int, feats: list[str] | None) -> list[float]:
    if arm == "A_residual":
        return list(record.residual[str(layer)])
    if arm == "B_sae_topk":
        layer_acts = record.sae_features.get(str(layer), {})
        return [layer_acts.get(f, 0.0) for f in (feats or [])]
    if arm == "C_geometry":
        g = record.geometry[str(layer)]
        return [g[m] for m in GEOMETRY_METRICS]
    if arm == "D_output":
        return [record.output_baselines[m] for m in OUTPUT_BASELINES]
    raise ValueError(arm)


def run_probes(
    records_path: Path,
    k_values: tuple[int, ...] = (16, 64, 256),
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    raw = json.loads(records_path.read_text())
    records = [SampleRecord(**r) for r in raw]
    # refusals analyzed separately (protocol) — excluded from binary probes
    binary = [r for r in records if r.label in ("correct", "incorrect")]
    print(f"{len(records)} records, {len(binary)} binary, "
          f"{sum(r.label == 'incorrect' for r in binary)} incorrect")

    rng = random.Random(seed)
    idx = list(range(len(binary)))
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]

    results: dict[str, dict] = {}

    def evaluate(arm: str, layer: int, k: int | None = None) -> dict:
        aucs: list[float] = []
        pooled_s: list[float] = []
        pooled_y: list[int] = []
        for f in range(n_folds):
            test_i = set(folds[f])
            train = [binary[i] for i in idx if i not in test_i]
            test = [binary[i] for i in idx if i in test_i]
            feats = (
                select_contrastive_features(train, layer, k)
                if arm == "B_sae_topk"
                else None
            )
            Xtr = [featurize(r, arm, layer, feats) for r in train]
            Xte = [featurize(r, arm, layer, feats) for r in test]
            ytr = [1 if r.label == "incorrect" else 0 for r in train]
            yte = [1 if r.label == "incorrect" else 0 for r in test]
            # length control on every scalar input column
            lens_tr = [float(r.answer_token_count) for r in train]
            Xtr = list(map(list, zip(*[residualize(col, lens_tr) for col in zip(*Xtr)]))) if Xtr and Xtr[0] else Xtr
            scores = fit_logistic(Xtr, ytr, Xte)
            aucs.append(rank_auc(scores, yte))
            pooled_s.extend(scores)
            pooled_y.extend(yte)
        # CI from pooled out-of-fold predictions (protocol: bootstrap, 1000 resamples)
        return {"auc": st.mean(aucs), "ci95": boot_ci(pooled_s, pooled_y), "n": len(pooled_y)}

    for layer in LAYERS:
        results[f"A_residual_L{layer}"] = evaluate("A_residual", layer)
        for k in k_values:
            results[f"B_sae_top{k}_L{layer}"] = evaluate("B_sae_topk", layer, k)
        results[f"C_geometry_L{layer}"] = evaluate("C_geometry", layer)
    results["D_output_baselines"] = evaluate("D_output", LAYERS[0])

    # transfer split: train high-popularity, test low (and reverse)
    for train_bucket, test_bucket in (("high", "low"), ("low", "high")):
        train = [r for r in binary if r.popularity_bucket == train_bucket]
        test = [r for r in binary if r.popularity_bucket == test_bucket]
        for layer in LAYERS:
            Xtr = [featurize(r, "A_residual", layer, None) for r in train]
            Xte = [featurize(r, "A_residual", layer, None) for r in test]
            yte = [1 if r.label == "incorrect" else 0 for r in test]
            scores = fit_logistic(Xtr, [1 if r.label == "incorrect" else 0 for r in train], Xte)
            auc = rank_auc(scores, yte)
            results[f"transfer_{train_bucket}->{test_bucket}_L{layer}"] = {
                "auc": auc, "ci95": boot_ci(scores, yte), "n": len(yte),
            }

    # pre-registered verdict (protocol.md decision rule, locked thresholds)
    results["VERDICT"] = verdict(results)

    out = records_path.parent / "probe_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/results/self_generation_probe/records.json"),
        help="Extraction output (list of SampleRecord dicts).",
    )
    args = parser.parse_args()
    if not args.records.exists():
        raise SystemExit(
            f"{args.records} not found — run the Modal extraction step first "
            "(see protocol.md, step 2)."
        )
    run_probes(args.records)
