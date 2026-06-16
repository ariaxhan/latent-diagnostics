"""Local (no-Modal) extraction for the self-generation probe.

Runs gemma-2-2b (4-bit, MLX, MPS) on the model's OWN generations and produces
records.json in the SampleRecord schema probe_harness.run_probes() consumes.

Per question:
  1. few-shot greedy generation  -> self-answer  (base model, disjoint few-shot)
  2. teacher-forced forward over prompt+answer:
       - residual stream at blocks {9,12,20}.hook_resid_post  (arm A)
       - GemmaScope JumpReLU SAE features at those layers      (arm B)
       - inertia-tensor geometry on the SAE activation cloud   (arm C)
       - output-only baselines from the capped logits          (arm D)
  3. label correct/incorrect/refusal by normalized string match on gold.

Pure local compute. No cloud, no creds, no spend. gemma 4-bit from the ungated
mlx-community mirror; GemmaScope SAEs are ungated .npz weights.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

LAYERS = (9, 12, 20)
SAE_L0 = {9: 73, 12: 82, 20: 71}
MODEL_ID = "mlx-community/gemma-2-2b-4bit"
GEOM_CLOUD_K = 8          # last-K token positions form the per-sample SAE cloud (arm C)
MAX_NEW_TOKENS = 12

GEOMETRY_METRICS = (
    "sphericity", "elongation", "eigenvalue_entropy",
    "effective_dimensionality", "misalignment_angle",
)

# Few-shot preamble — entities DISJOINT from the eval set (no leakage).
FEWSHOT = (
    "Answer each question with a short factual answer.\n\n"
    "Q: What is the capital of Portugal?\nA: Lisbon\n\n"
    "Q: Who wrote Romeo and Juliet?\nA: William Shakespeare\n\n"
    "Q: In which year was Twitter founded?\nA: 2006\n\n"
    "Q: Who directed the film Inglourious Basterds?\nA: Quentin Tarantino\n\n"
    "Q: In which country is the Brandenburg Gate located?\nA: Germany\n\n"
)


# --------------------------------------------------------------------------
# Geometry (vendored from experiments/core/geometric_analysis.py — numpy only)
# --------------------------------------------------------------------------
def compute_geometry(points: np.ndarray) -> dict[str, float]:
    n, d = points.shape
    if n < 3:
        return {m: 0.0 for m in GEOMETRY_METRICS}
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    L1 = max(eigvals[0], 1e-12)
    L2 = eigvals[1] if len(eigvals) > 1 else 0.0
    L3 = eigvals[2] if len(eigvals) > 2 else 0.0
    a, b, c = np.sqrt(L1), np.sqrt(max(L2, 0)), np.sqrt(max(L3, 0))
    c_over_a = c / a if a > 0 else 0.0   # sphericity
    b_over_a = b / a if a > 0 else 0.0   # elongation
    total = eigvals.sum()
    eff_dim = (total ** 2) / ((eigvals ** 2).sum() + 1e-12)
    probs = eigvals[eigvals > 0] / total if total > 0 else np.array([1.0])
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    # misalignment: angle between centroid and the major axis
    w, V = np.linalg.eigh(cov)
    major = V[:, int(np.argmax(w))]
    cn = np.linalg.norm(centroid)
    if cn > 1e-9:
        cos = abs(float(centroid @ major)) / (cn * np.linalg.norm(major) + 1e-9)
        misalign = float(np.degrees(np.arccos(np.clip(cos, 0, 1))))
    else:
        misalign = 0.0
    return {
        "sphericity": float(c_over_a), "elongation": float(b_over_a),
        "eigenvalue_entropy": entropy, "effective_dimensionality": float(eff_dim),
        "misalignment_angle": misalign,
    }


# --------------------------------------------------------------------------
# GemmaScope JumpReLU SAE (official encode: no b_dec subtraction on encode)
# --------------------------------------------------------------------------
class JumpReLUSAE:
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.W_enc = mx.array(d["W_enc"].astype(np.float32))      # (d_model, d_sae)
        self.b_enc = mx.array(d["b_enc"].astype(np.float32))      # (d_sae,)
        self.threshold = mx.array(d["threshold"].astype(np.float32))

    def encode(self, x: mx.array) -> mx.array:
        pre = x @ self.W_enc + self.b_enc
        mask = pre > self.threshold
        return mx.where(mask, mx.maximum(pre, 0.0), 0.0)


# --------------------------------------------------------------------------
# Model forward with residual capture
# --------------------------------------------------------------------------
def forward_capture(model, ids: mx.array):
    """Return (residuals {L: (seq,d_model) np}, logits (seq,vocab) np) for one seq."""
    m = model.model
    h = m.embed_tokens(ids) * (m.args.hidden_size ** 0.5)
    mask = create_attention_mask(h, None, return_array=True)
    caps = {}
    for i, layer in enumerate(m.layers):
        h = layer(h, mask, None)
        if i in LAYERS:
            caps[i] = h
    final = m.norm(h)
    logits = m.embed_tokens.as_linear(final)
    logits = mx.tanh(logits / model.final_logit_softcapping) * model.final_logit_softcapping
    mx.eval(logits, *caps.values())
    return {i: np.array(caps[i][0]) for i in caps}, np.array(logits[0])


def greedy_generate(model, tokenizer, prompt_ids: list[int]) -> tuple[list[int], str]:
    """Greedy-decode, then trim to the first answer line (gemma emits a '\\n\\n'
    token that a bare newline-id check misses). Returns (answer_ids, answer_text)."""
    ids = list(prompt_ids)
    gen = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(mx.array(ids)[None])[:, -1, :]
        mx.eval(logits)
        nxt = int(np.array(logits[0]).argmax())
        gen.append(nxt)
        ids.append(nxt)
        if "\n" in tokenizer.decode(gen):
            break
    # trim to the tokens whose decode stays on the first line
    answer_ids = []
    for j in range(len(gen)):
        if "\n" in tokenizer.decode(gen[: j + 1]):
            answer_ids = gen[:j]            # exclude the newline-introducing token
            break
        answer_ids = gen[: j + 1]
    text = tokenizer.decode(answer_ids).strip()
    return answer_ids, text


# --------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(the|a|an|in|of|city|film|movie)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

REFUSAL_MARKERS = ("i don't know", "i do not know", "unknown", "not sure",
                   "cannot", "can't", "no idea", "unsure")

def label(generated: str, gold: str) -> str:
    g = generated.strip()
    if not g:
        return "refusal"
    gl = g.lower()
    if any(mk in gl for mk in REFUSAL_MARKERS):
        return "refusal"
    ng, ngold = _norm(g), _norm(gold)
    if not ngold:
        return "incorrect"
    # year answers: exact 4-digit match anywhere
    if re.fullmatch(r"\d{4}", gold.strip()):
        return "correct" if re.search(rf"\b{re.escape(gold.strip())}\b", g) else "incorrect"
    # else: gold tokens all present, or gold substring present
    if ngold in ng or all(tok in ng.split() for tok in ngold.split()):
        return "correct"
    return "incorrect"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    qpath = Path("data/self_generation_probe/questions.json")
    questions = json.loads(qpath.read_text())
    print(f"loaded {len(questions)} questions", flush=True)

    print("loading gemma-2-2b-4bit (MLX)...", flush=True)
    model, tokenizer = load(MODEL_ID)
    saes = {L: JumpReLUSAE(f"data/gemmascope/layer_{L}/width_16k/average_l0_{SAE_L0[L]}/params.npz")
            for L in LAYERS}

    records = []
    t0 = time.time()
    for qi, q in enumerate(questions):
        prompt = FEWSHOT + f"Q: {q['question']}\nA:"
        prompt_ids = tokenizer.encode(prompt)
        gen_ids, gen_text = greedy_generate(model, tokenizer, prompt_ids)
        lab = label(gen_text, q["gold_answer"])

        full_ids = prompt_ids + gen_ids
        resid, logits = forward_capture(model, mx.array(full_ids)[None])
        p_len, n_ans = len(prompt_ids), len(gen_ids)

        rec = {
            "question_id": q["question_id"], "question": q["question"],
            "entity": q["entity"], "popularity_bucket": q["popularity_bucket"],
            "generated_answer": gen_text, "label": lab,
            "answer_token_count": max(n_ans, 1),
            "residual": {}, "sae_features": {}, "geometry": {}, "output_baselines": {},
        }

        # answer-token slice (residual capture is for blocks output at those positions)
        ans_slice = slice(p_len, p_len + n_ans) if n_ans > 0 else slice(p_len - 1, p_len)
        for L in LAYERS:
            acts = resid[L]                                  # (seq, d_model)
            ans_acts = acts[ans_slice]                       # (n_ans, d_model)
            rec["residual"][str(L)] = ans_acts.mean(axis=0).tolist()
            # SAE encode answer tokens (mean-pooled sparse dict for arm B)
            sae_ans = np.array(saes[L].encode(mx.array(ans_acts)))   # (n_ans, d_sae)
            mean_feat = sae_ans.mean(axis=0)
            nz = np.nonzero(mean_feat)[0]
            rec["sae_features"][str(L)] = {str(int(i)): float(mean_feat[i]) for i in nz}
            # geometry on the SAE activation cloud over last-K positions (arm C)
            k0 = max(0, len(full_ids) - GEOM_CLOUD_K)
            cloud_acts = acts[k0:]                            # (<=K, d_model)
            cloud_sae = np.array(saes[L].encode(mx.array(cloud_acts)))   # (<=K, d_sae)
            active = np.nonzero(cloud_sae.sum(axis=0))[0]     # union of active feats
            cloud = cloud_sae[:, active] if active.size else cloud_sae[:, :1]
            rec["geometry"][str(L)] = compute_geometry(cloud)

        # arm D: output-only baselines over answer positions
        if n_ans > 0:
            maxp, ent, lp = [], [], []
            for j in range(n_ans):
                pos = p_len + j - 1                          # logits here predict gen token j
                z = logits[pos].astype(np.float64)           # fp16 -> fp64 (avoid underflow)
                z = z - z.max()
                e = np.exp(z); sm = e / e.sum()
                nz = sm[sm > 0]
                maxp.append(float(sm.max()))
                ent.append(float(-(nz * np.log(nz)).sum()))
                lp.append(float(np.log(max(sm[gen_ids[j]], 1e-300))))
            rec["output_baselines"] = {
                "max_logit_prob": float(np.mean(maxp)),
                "logit_entropy": float(np.mean(ent)),
                "answer_mean_logprob": float(np.mean(lp)),
            }
        else:
            rec["output_baselines"] = {"max_logit_prob": 0.0, "logit_entropy": 0.0,
                                       "answer_mean_logprob": -20.0}

        records.append(rec)
        if (qi + 1) % 10 == 0 or qi == len(questions) - 1:
            nc = sum(r["label"] == "correct" for r in records)
            ni = sum(r["label"] == "incorrect" for r in records)
            nr = sum(r["label"] == "refusal" for r in records)
            print(f"[{qi+1}/{len(questions)}] correct={nc} incorrect={ni} refusal={nr} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = Path("data/results/self_generation_probe/records.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    nc = sum(r["label"] == "correct" for r in records)
    ni = sum(r["label"] == "incorrect" for r in records)
    nr = sum(r["label"] == "refusal" for r in records)
    print(f"\nwrote {len(records)} records -> {out}")
    print(f"correct={nc} incorrect={ni} refusal={nr}")


if __name__ == "__main__":
    main()
