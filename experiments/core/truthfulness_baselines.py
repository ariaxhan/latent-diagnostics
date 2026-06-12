"""Truthfulness baselines: AUC of every cached metric on TruthfulQA samples.

Reproduces the 2026-06-12 review analysis: no attribution-graph metric, and no
output-level baseline (max logit prob, logit entropy), exceeds AUC 0.54 for
truthful-vs-false classification on this dataset/model (Gemma-2-2B, statements
as input). Confirms the d=0.05 null from MISSION.md with rank statistics.

Usage:
    python experiments/core/truthfulness_baselines.py \
        [--data data/results/truthfulness_metrics_clean.json]

No dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

METRICS = [
    "n_active",
    "mean_activation",
    "max_activation",
    "mean_influence",
    "max_influence",
    "top_100_concentration",
    "max_logit_prob",
    "logit_entropy",
]


def rank_auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC with tie handling (average ranks)."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        raise ValueError("Need both classes present.")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            if labels[order[k]] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def cohens_d(scores: list[float], labels: list[int]) -> float:
    a = [s for s, l in zip(scores, labels) if l == 1]
    b = [s for s, l in zip(scores, labels) if l == 0]
    pooled = math.sqrt(
        (st.variance(a) * (len(a) - 1) + st.variance(b) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    return (st.mean(a) - st.mean(b)) / pooled if pooled > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/results/truthfulness_metrics_clean.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.data.read_text())
    samples = payload["samples"]
    labels = [1 if s["label"] == "truthful" else 0 for s in samples]
    print(f"n={len(samples)}  truthful={sum(labels)}  false={len(labels) - sum(labels)}")
    print(f"model={payload['metadata'].get('model')}")
    print()
    print(f"{'metric':24s} {'AUC':>6s} {'|AUC-0.5|':>9s} {'d':>7s}")

    for metric in METRICS:
        scores = [float(s[metric]) for s in samples]
        auc = rank_auc(scores, labels)
        print(f"{metric:24s} {auc:6.3f} {abs(auc - 0.5):9.3f} {cohens_d(scores, labels):+7.3f}")

    lengths = [float(len(s["text"].split())) for s in samples]
    print(f"{'text_length_words':24s} {rank_auc(lengths, labels):6.3f}")


if __name__ == "__main__":
    main()
