## CONTEXT HANDOFF

**Summary**: Explored using SAE spectroscopy to detect prompt injection attacks, built extensive research docs, then tore it down to identify what's actually useful vs metaphor overhead.

**Goal**: Build a working prompt injection detector using sparse feature analysis (SAE or transcoders) with probability tables for fast lookup.

**Current state**:
- Created 9 research docs with 6 physics frameworks mapped to AI security
- Critical teardown revealed: 70% metaphor, 30% actionable
- The useful part: **probability tables** (pre-computed risk by feature cluster)
- Hypothesis test running: Does cosine similarity of SAE features discriminate injection vs benign?
- Decision: Shift to **transcoders** (better interpretability per recent research)

**Decisions made**:
- Cut all physics terminology from implementation (Wannier-Stark, GOE-S-matrix, etc. are just metaphors)
- Probability tables are the novel/useful idea (cluster prompts → pre-compute injection success rate → fast lookup)
- Use [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) for transcoder support
- Skip STL parser, evolution layer, regret minimization (overkill for MVP)

**Artifacts created**:
- `_meta/research/injection-detection-framework.md` — Two-stage detection architecture
- `_meta/research/physics-foundations.md` — All 6 frameworks mapped (reference only)
- `_meta/research/attack-cross-sections.md` — 5 attack types with signatures
- `_meta/research/witness-checklist.md` — Operational decision tree
- `_meta/research/enforcement-evolution.md` — STL + AlphaEvolve (deferred)
- `_meta/research/system-design.md` — Full system architecture spec
- `_meta/reviews/injection-detection-teardown.md` — **Critical review: what to keep/cut**
- `test_injection_hypothesis.py` — Running test: injection vs benign SAE similarity

**Open threads**:
- Hypothesis test still running (check task b2f73fd output)
- If cosine similarity doesn't discriminate → rethink approach
- Transcoder availability for Gemma-2-2B not confirmed yet

**Next steps**:
1. Check hypothesis test results (`cat /private/tmp/claude/.../b2f73fd.output`)
2. If positive: Build minimal detector (~50 lines) with probability tables
3. Switch from SAE-Lens to [sparsify](https://github.com/EleutherAI/sparsify) for transcoder support
4. Create IPI-100 benchmark (50 injection + 50 benign prompts)

**Warnings**:
- Existing hallucination data (HB-1000) showed NO discrimination (Cohen's d < 0.1) — but hallucinations were 1-word diffs, not structural changes like injection
- Don't implement full 3-layer system until hypothesis validated
- Physics metaphors are intellectually satisfying but add zero computational value

**Key findings from teardown**:
```
KEEP (actionable):
- cosine_similarity(baseline, current) — simple drift detection
- Probability tables — cluster prompts, pre-compute risk
- Geometric metrics (c/a, entropy) — might work, needs validation

CUT (overhead):
- All 6 physics framework names
- STL parser
- Evolution layer
- Regret minimization
- 8 research documents of prose
```

**The minimal viable detector** (~50 lines):
```python
class InjectionDetector:
    def __init__(self, extractor, probability_table):
        self.extractor = extractor  # SAE or Transcoder
        self.table = probability_table
        self.baseline = None

    def set_baseline(self, system_prompt):
        self.baseline = self.extractor.encode(system_prompt)

    def analyze(self, prompt):
        features = self.extractor.encode(prompt)
        cluster = find_nearest_cluster(features, self.table.centroids)
        table_risk = self.table[cluster].injection_rate
        similarity = cosine_similarity(self.baseline, features)
        risk = 0.6 * table_risk + 0.4 * (1 - similarity)
        return {"risk": risk, "action": "BLOCK" if risk > 0.7 else "ALLOW"}
```

**File paths to read**:
- `_meta/reviews/injection-detection-teardown.md` — The critical review (most important)
- `_meta/research/system-design.md` — Full system spec if needed
- `test_injection_hypothesis.py` — The running experiment

**Continuation prompt for new session**:
> We're building a prompt injection detector using sparse feature analysis. After extensive research, we did a critical teardown and found the useful core: probability tables (cluster prompts in feature space → pre-compute injection success rate → fast lookup). The hypothesis test (injection vs benign cosine similarity) may be running or finished — check results. Next: If hypothesis validates, build minimal 50-line detector and switch to transcoders via EleutherAI/sparsify library.
