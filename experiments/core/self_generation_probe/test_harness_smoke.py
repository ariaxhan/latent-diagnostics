"""Smoke test for probe_harness — analysis path only, NO GPU / model / creds.

This does NOT run the real experiment (which needs Modal-extracted activations).
It feeds SYNTHETIC SampleRecords through the full downstream path — arms A–D,
5-fold CV, length control, bootstrap CIs, and the pre-registered verdict() — to
prove the harness is one-command the instant real records.json lands.

The numbers here are MEANINGLESS (planted synthetic signal). Only the control
flow + schema + verdict wiring are under test.

Run:  python3 experiments/core/self_generation_probe/test_harness_smoke.py
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import probe_harness as H


def _make_records(n: int = 80, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        incorrect = i % 2  # balanced
        bucket = "high" if i % 4 < 2 else "low"
        # planted signal in residual (separable); geometry ~ chance; sae moderate
        sig = rng.gauss(incorrect * 1.5, 1.0)
        rec = H.SampleRecord(
            question_id=f"{bucket}_{i:04d}",
            question="In which year was X founded?",
            entity=f"E{i}",
            popularity_bucket=bucket,
            generated_answer="1999",
            label="incorrect" if incorrect else "correct",
            answer_token_count=rng.randint(1, 12),
            residual={str(L): [sig + rng.gauss(0, 0.5) for _ in range(8)] for L in H.LAYERS},
            sae_features={str(L): {str(f): max(0.0, rng.gauss(incorrect * 0.3, 1.0)) for f in range(40)} for L in H.LAYERS},
            geometry={str(L): {m: rng.gauss(0, 1) for m in H.GEOMETRY_METRICS} for L in H.LAYERS},
            output_baselines={m: rng.gauss(incorrect * 0.2, 1.0) for m in H.OUTPUT_BASELINES},
        )
        rows.append(asdict(rec))
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        recs = Path(d) / "records.json"
        recs.write_text(json.dumps(_make_records()))
        results = H.run_probes(recs)

    # --- assertions: the path produced what the pre-registration needs ---
    assert "VERDICT" in results, "verdict() not wired into run_probes"
    v = results["VERDICT"]
    assert v["result"] in {"ALIVE", "PIVOT_DROP_SAE", "DEAD_AT_THIS_SCALE"}, v["result"]

    # every probe arm present, each with an auc + a bootstrap CI
    arm_keys = [k for k in results if k.startswith(("A_residual_", "B_sae_top", "C_geometry_", "D_output", "transfer_"))]
    assert any(k.startswith("A_residual_") for k in arm_keys), "arm A missing"
    assert any(k.startswith("B_sae_top") for k in arm_keys), "arm B missing"
    assert any(k.startswith("C_geometry_") for k in arm_keys), "arm C missing"
    assert "D_output_baselines" in results, "arm D missing"
    assert any(k.startswith("transfer_") for k in arm_keys), "transfer split missing"
    for k in arm_keys:
        cell = results[k]
        assert "auc" in cell and "ci95" in cell, f"{k} lacks auc/ci95"
        lo, hi = cell["ci95"]
        assert math.isnan(lo) or (0.0 <= lo <= hi <= 1.0), f"{k} CI malformed: {cell['ci95']}"

    # verdict reflects the locked thresholds (sanity, not a result claim)
    assert isinstance(v["H1_residual_ge_0.75"], bool)
    assert isinstance(v["H2_sae_topk_ge_90pct_of_residual"], bool)
    assert isinstance(v["H3_geometry_at_chance"], bool)

    print("\nSMOKE TEST PASSED — analysis path runs end-to-end on synthetic data.")
    print(f"  arms exercised : {len(arm_keys)} result cells, all with auc + bootstrap CI")
    print(f"  verdict wired  : {v['result']}  (synthetic numbers — NOT a real result)")
    print("  GPU/model/creds: NONE used. Real run only needs records.json from Modal extraction.")


if __name__ == "__main__":
    main()
