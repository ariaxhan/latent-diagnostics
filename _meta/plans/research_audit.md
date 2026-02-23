# Research Folder Audit Plan

**Contract:** CR-20260223132602-68611-C
**Date:** 2026-02-23
**Purpose:** Determine what stays vs archives for clean Medium presentation

---

## Decision Framework Applied

- **KEEP** if: Directly explains methodology, reproducible, has concrete results
- **ARCHIVE** if: Speculative theory, "vector native" not grounded in data, session notes
- **DELETE** if: Empty or redundant

---

## File-by-File Decisions

### 1. README.md
**Decision:** ARCHIVE

**Reasoning:** This is the index for a "Neural Polygraph" prompt injection detection system that was largely theoretical. The implementation status shows most components "Not implemented". It references physics frameworks (Wannier-Stark, GOE-S-Matrix, etc.) that were never empirically validated.

**Specific issues:**
- Lists "Six Frameworks" but none were implemented
- "Three Witnesses" (P, E, I) never built
- "Cross section tables" never computed
- Paints a picture of a system that doesn't exist

**Archive location:** `research/_archived/injection-detection-theory/`

---

### 2. physics-foundations.md
**Decision:** ARCHIVE

**Reasoning:** Pure speculative analogy. Maps physics concepts to AI security with no empirical validation. Contains pseudo-code that was never run, formulas that were never tested.

**Specific issues:**
- "Wannier-Stark Localization" analogy - never tested
- "GOE-S-Matrix" cross sections - never computed
- "Hyperon Puzzle" layered defense - never implemented
- "Knot Theory y-ification" - no data
- Research roadmap shows all items "[ ] Not done"

**Archive location:** `research/_archived/injection-detection-theory/`

---

### 3. system-design.md
**Decision:** ARCHIVE

**Reasoning:** Detailed API design and architecture for a system that was never built. Speculative design document, not documentation of working code.

**Specific issues:**
- `InjectionDetector` class - never implemented
- Streaming API - never built
- STL rules configuration - never created
- Performance targets - never measured
- All "Implementation Phases" marked pending

**Archive location:** `research/_archived/injection-detection-theory/`

---

### 4. vector_native_empirical_validation.md
**Decision:** ARCHIVE

**Reasoning:** While this contains empirical data ("VN won 10/10 tests"), it validates "vector native" prompt notation, NOT the latent diagnostics system. This is about prompt engineering aesthetics, not mechanistic interpretability.

**Specific issues:**
- Claims symbols activate "4-9x more features per character"
- Results not reproduced by independent verification
- "Vector native" is personal notation preference, not scientific finding
- Does not help explain how latent diagnostics works

**Archive location:** `research/_archived/vector-native-experiments/`

---

### 5. enforcement-evolution.md
**Decision:** ARCHIVE

**Reasoning:** More speculative theory about multi-agent STL rules and AlphaEvolve evolutionary defense. Implementation status shows everything "to build".

**Specific issues:**
- STL rule checker - never implemented
- Penalty functions - never tested
- Defense evolver - never built
- Regret tracker - never created
- References papers but doesn't apply them

**Archive location:** `research/_archived/injection-detection-theory/`

---

### 6. witness-checklist.md
**Decision:** ARCHIVE

**Reasoning:** Operational checklist for a detection system that was never built. Implementation status explicitly shows:
- `survival_probability()` - "Not implemented"
- `compute_entanglement()` - "Partial"
- `classify_phase()` - "Not implemented"
- Threshold calibration - "No data yet"

**Archive location:** `research/_archived/injection-detection-theory/`

---

### 7. open-questions.md
**Decision:** ARCHIVE

**Reasoning:** Research hypotheses that were never tested. Every item in "Next Actions" table shows "Not started". This is a wishlist, not documentation of work done.

**Specific issues:**
- "Phase transition hypothesis" - Not started
- "Stiffness effect" - Not started
- "Topological signatures" - Not started
- "Threshold calibration" - Blocked (never unblocked)

**Archive location:** `research/_archived/injection-detection-theory/`

---

### 8. 2026-02-20-theory-benchmarks.md
**Decision:** PARTIAL KEEP / PARTIAL ARCHIVE

**Reasoning:** Mixed content. Some useful benchmark survey information (PINT, Open-Prompt-Injection) worth keeping. Physics paper analysis is speculative.

**Keep (extract to separate file):**
- Part 3: Prompt Injection Benchmarks - useful survey
- Part 2: Transcoder vs SAE Comparison - relevant literature review

**Archive:**
- Part 1: Physics Paper Analysis - speculative mappings
- Part 4: New Research Direction - never executed

**Recommendation:** Extract useful portions to `research/literature/` or keep inline in main docs. Archive the speculative parts.

---

### 9. session-2026-02-22.md
**Decision:** ARCHIVE

**Reasoning:** Session notes, not methodology documentation. Contains valuable insights but format is not appropriate for external readers.

**Key insights to preserve elsewhere:**
- "Length confound confirmed: r=0.96 correlation"
- "Best metric: mean_activation (AUC=0.830)"
- "Shifted focus from classification to mechanistic exploration"
- "Truthfulness NOT detectable via activation topology (d=0.05)"

**Recommendation:** Extract findings to a "Lessons Learned" or "What We Discovered" section in the main notebook/docs. Archive the raw session notes.

**Archive location:** `research/_archived/session-notes/`

---

## Archive Structure

```
research/
  _archived/
    README.md                    # Explains what's here and why
    injection-detection-theory/  # The "Neural Polygraph" theory docs
      README.md
      physics-foundations.md
      system-design.md
      enforcement-evolution.md
      witness-checklist.md
      open-questions.md
    vector-native-experiments/
      vector_native_empirical_validation.md
    session-notes/
      session-2026-02-22.md
```

---

## Files to KEEP in research/

After this audit, research/ should contain:

1. **A new README.md** that honestly describes what the latent diagnostics system actually does (not what was speculated)
2. **Extracted benchmark literature** (if worth keeping separately)
3. **Any methodology docs that reflect implemented code**

---

## Recommended Disclaimers for Archive

Each archived file should have a header added:

```markdown
> **ARCHIVED**: This document represents speculative research directions
> that were not empirically validated. The "Neural Polygraph" injection
> detection system described here was never fully implemented. For what
> actually works, see the main experiments/ and notebooks/ directories.
```

---

## Summary Table

| File | Decision | Reason |
|------|----------|--------|
| README.md | ARCHIVE | Index for unbuilt system |
| physics-foundations.md | ARCHIVE | Speculative physics analogies |
| system-design.md | ARCHIVE | Design for unbuilt system |
| vector_native_empirical_validation.md | ARCHIVE | VN validation, not diagnostics |
| enforcement-evolution.md | ARCHIVE | Speculative enforcement theory |
| witness-checklist.md | ARCHIVE | Checklist for unbuilt features |
| open-questions.md | ARCHIVE | Untested hypotheses |
| 2026-02-20-theory-benchmarks.md | PARTIAL | Extract benchmark survey, archive rest |
| session-2026-02-22.md | ARCHIVE | Session notes (preserve insights elsewhere) |

---

## Key Insight

The research/ folder documents an ambitious theoretical framework ("Neural Polygraph") that was never implemented. The actual working code in experiments/ tells a different, more honest story: we can extract SAE attribution metrics, but classification doesn't beat baselines.

**The honest narrative:**
1. We tried to build a prompt injection detector using SAE features
2. The physics analogies were intellectually interesting but never validated
3. What actually works: metric extraction and visualization
4. What doesn't work: classification (length confounds, low sample sizes)

This audit recommends archiving the theoretical speculation and keeping only what reflects the actual implemented system.
