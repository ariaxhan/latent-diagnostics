"""Summarize probe_results.json: pre-registered verdict + the arm-D comparison
(the commission's 'honest nag' — does anything INSIDE the model beat reading the
output probabilities?) + the popularity-confound check.

Reads only; computes no thresholds. Run after probe_harness.run_probes().
"""
import json, math
from pathlib import Path

R = Path("data/results/self_generation_probe/probe_results.json")


def best(results, prefix, kmax=None):
    def topk(k):
        try: return int(k.split("B_sae_top")[1].split("_L")[0])
        except Exception: return None
    items = []
    for k, v in results.items():
        if not (k.startswith(prefix) and isinstance(v, dict)): continue
        if math.isnan(v.get("auc", float("nan"))): continue
        if kmax is not None and (topk(k) or 1e9) > kmax: continue
        items.append((k, v))
    if not items: return None, None
    return max(items, key=lambda kv: kv[1]["auc"])


def main():
    results = json.loads(R.read_text())
    v = results["VERDICT"]
    print("=" * 64)
    print("SELF-GENERATION PROBE — RESULTS")
    print("=" * 64)

    ak, av = best(results, "A_residual_")
    bk, bv = best(results, "B_sae_top", kmax=64)
    bk_any, bv_any = best(results, "B_sae_top")
    ck, cv = best(results, "C_geometry_")
    dv = results.get("D_output_baselines", {})

    def fmt(k, val):
        if not val: return f"  {k:34} n/a"
        ci = val.get("ci95", [float('nan')]*2)
        return f"  {k:34} AUC={val['auc']:.3f}  CI95=[{ci[0]:.3f},{ci[1]:.3f}]  n={val.get('n','?')}"

    print("\n-- best arm per family --")
    print(fmt(ak or "A_residual", av))
    print(fmt(bk or "B_sae(k<=64)", bv))
    print(fmt(bk_any or "B_sae(any k)", bv_any))
    print(fmt(ck or "C_geometry", cv))
    print(fmt("D_output_baselines", dv))

    print("\n-- pre-registered hypotheses --")
    print(f"  H1 residual AUC >= 0.75 .......... {v['H1_residual_ge_0.75']}  (best {v['best_residual_auc']:.3f})")
    print(f"  H2 SAE(k<=64) >= 90% of residual . {v['H2_sae_topk_ge_90pct_of_residual']}  (best {v['best_sae_topk_le64_auc']:.3f})")
    print(f"  H3 geometry at chance ............ {v['H3_geometry_at_chance']}")
    print(f"\n  VERDICT: {v['result']}")
    print(f"  {v['reason']}")

    # The honest nag: internal vs output-only
    print("\n-- arm-D comparison (does internal beat reading the logits?) --")
    d_auc = dv.get("auc", float("nan"))
    a_auc = av["auc"] if av else float("nan")
    b_auc = bv_any["auc"] if bv_any else float("nan")
    print(f"  output-only (arm D) AUC ......... {d_auc:.3f}")
    print(f"  best residual (arm A) AUC ....... {a_auc:.3f}   delta vs D = {a_auc - d_auc:+.3f}")
    print(f"  best SAE (arm B) AUC ............ {b_auc:.3f}   delta vs D = {b_auc - d_auc:+.3f}")
    a_ci = av["ci95"] if av else [float('nan')]*2
    beats = (not math.isnan(a_auc)) and a_auc > d_auc and a_ci[0] > d_auc
    print(f"  -> internal beats output baseline (A>D and A's CI lower bound > D): {beats}")

    # transfer (generalization across popularity)
    print("\n-- transfer (train high-pop / test low-pop and reverse) --")
    for k, val in results.items():
        if k.startswith("transfer_") and isinstance(val, dict):
            print(fmt(k, val))


if __name__ == "__main__":
    main()
