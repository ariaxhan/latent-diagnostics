"""
Injection Detection Server

Model loads once at startup. Stays in memory.
Send requests via HTTP - no reload penalty.
"""

import os
import sys
import time
from typing import List, Optional
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, "src")

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Globals - loaded once at startup
MODEL = None
SAE_EXTRACTOR = None
TC_EXTRACTOR = None
DEVICE = None


class AnalyzeRequest(BaseModel):
    texts: List[str]
    baseline: Optional[str] = "You are a helpful AI assistant."


class FeatureResult(BaseModel):
    text: str
    sae_l0: int
    sae_energy: float
    tc_l0: Optional[int] = None
    tc_energy: Optional[float] = None
    sae_similarity: Optional[float] = None
    tc_similarity: Optional[float] = None


class AnalyzeResponse(BaseModel):
    results: List[FeatureResult]
    baseline_sae_l0: int
    baseline_tc_l0: Optional[int] = None
    load_time_ms: float


def load_models():
    """Load all models once at startup."""
    global MODEL, SAE_EXTRACTOR, TC_EXTRACTOR, DEVICE

    # Device selection
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"[startup] Device: {DEVICE}")

    # Load model
    print("[startup] Loading Gemma-2-2B...")
    t0 = time.time()
    from transformer_lens import HookedTransformer
    MODEL = HookedTransformer.from_pretrained("gemma-2-2b", device=DEVICE)
    print(f"[startup] Model loaded in {time.time() - t0:.1f}s")

    # Load SAE
    print("[startup] Loading SAE...")
    t0 = time.time()
    from hallucination_detector.feature_extractors import SAEExtractor, TranscoderExtractor
    SAE_EXTRACTOR = SAEExtractor(device=DEVICE, layer=5)
    print(f"[startup] SAE loaded in {time.time() - t0:.1f}s")

    # Load Transcoder
    print("[startup] Loading Transcoder...")
    t0 = time.time()
    try:
        TC_EXTRACTOR = TranscoderExtractor(device=DEVICE, layer=5)
        print(f"[startup] Transcoder loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[startup] Transcoder failed: {e}")
        TC_EXTRACTOR = None

    print("[startup] Ready!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    yield
    # Cleanup if needed
    pass


app = FastAPI(
    title="Injection Detection API",
    description="SAE + Transcoder feature extraction. Model stays loaded.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": MODEL is not None,
        "sae_loaded": SAE_EXTRACTOR is not None,
        "transcoder_loaded": TC_EXTRACTOR is not None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """Extract features from texts and compare to baseline."""
    import torch.nn.functional as F

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()

    def extract(text, extractor):
        tokens = MODEL.to_tokens(text)
        _, cache = MODEL.run_with_cache(tokens)
        acts = cache[extractor.hook_point][0, -1, :]
        return extractor.encode(acts)

    # Baseline features
    baseline_sae = extract(request.baseline, SAE_EXTRACTOR)
    baseline_tc = extract(request.baseline, TC_EXTRACTOR) if TC_EXTRACTOR else None

    results = []
    for text in request.texts:
        # SAE
        sae_feat = extract(text, SAE_EXTRACTOR)
        sae_l0 = (sae_feat > 0).sum().item()
        sae_energy = sae_feat.sum().item()
        sae_sim = F.cosine_similarity(
            baseline_sae.unsqueeze(0), sae_feat.unsqueeze(0)
        ).item()

        # Transcoder
        tc_l0, tc_energy, tc_sim = None, None, None
        if TC_EXTRACTOR:
            tc_feat = extract(text, TC_EXTRACTOR)
            tc_l0 = (tc_feat > 0).sum().item()
            tc_energy = tc_feat.sum().item()
            tc_sim = F.cosine_similarity(
                baseline_tc.unsqueeze(0), tc_feat.unsqueeze(0)
            ).item()

        results.append(FeatureResult(
            text=text[:100],  # Truncate for response
            sae_l0=sae_l0,
            sae_energy=sae_energy,
            tc_l0=tc_l0,
            tc_energy=tc_energy,
            sae_similarity=sae_sim,
            tc_similarity=tc_sim,
        ))

    return AnalyzeResponse(
        results=results,
        baseline_sae_l0=(baseline_sae > 0).sum().item(),
        baseline_tc_l0=(baseline_tc > 0).sum().item() if baseline_tc is not None else None,
        load_time_ms=(time.time() - t0) * 1000,
    )


@app.post("/compare")
def compare(request: AnalyzeRequest):
    """
    Compare injection vs benign texts.
    First half of texts treated as injection, second half as benign.
    """
    response = analyze(request)

    n = len(response.results)
    mid = n // 2

    injection_sims = [r.sae_similarity for r in response.results[:mid]]
    benign_sims = [r.sae_similarity for r in response.results[mid:]]

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    return {
        "injection_mean_similarity": mean(injection_sims),
        "benign_mean_similarity": mean(benign_sims),
        "gap": mean(benign_sims) - mean(injection_sims),
        "discrimination": "YES" if mean(benign_sims) > mean(injection_sims) else "NO",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
