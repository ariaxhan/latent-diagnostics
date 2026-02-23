# Experiment 07: Complete Workflow Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 07: GHOST CLASSIFIER                  │
│              Testing Thesis B: Predictive Validity                  │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: HB-1000 Benchmark (200 samples × 4 domains)                 │
│ • Entity swaps (geo_001, geo_002, ...)                             │
│ • Temporal shifts (time_001, time_002, ...)                        │
│ • Logical inversions (logic_001, logic_002, ...)                   │
│ • Adversarial traps (adv_001, adv_002, ...)                        │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 1: Extract Ghost Features Per Sample                        │
│                                                                     │
│ For each sample:                                                    │
│   1. Get activations for fact text                                 │
│   2. Get activations for hallucination text                        │
│   3. Compute: F_ghost = F_hall \ F_fact                            │
│   4. Store: ghost_feature_ids, ghost_magnitudes                    │
│                                                                     │
│ Output: 200 sample records with raw feature IDs                    │
│         Global frequency counter (2,467 unique ghosts)             │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 2: Identify Universal Ghosts                                │
│                                                                     │
│ 1. Filter ghosts by frequency (≥10 samples)                        │
│ 2. Select top 100 by frequency                                     │
│ 3. Decode to vocabulary space:                                     │
│    • feature_direction = sae.W_dec[feature_id]                     │
│    • logits = model.unembed(feature_direction)                     │
│    • top_words = model.to_str_tokens(logits.argsort()[:10])       │
│ 4. Analyze domain distribution                                     │
│                                                                     │
│ Output: 100 UniversalGhost objects with semantics                  │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 3: Mechanistic Antagonism Analysis                          │
│                                                                     │
│ For each universal ghost:                                          │
│   1. Get ghost direction: W_dec[feature_id]                        │
│   2. For samples where ghost appears:                              │
│      • Get fact token embedding                                    │
│      • Compute dot product: ghost · fact_token                     │
│   3. Average dot products across samples                           │
│   4. Antagonism score = -avg_dot                                   │
│                                                                     │
│ Hypothesis: Negative scores = ghost opposes truth                  │
│                                                                     │
│ Output: 100 MechanisticAnalysis objects                            │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 4: Binary Classification                                    │
│                                                                     │
│ 1. Build feature matrix:                                           │
│    • Rows: 200 samples                                             │
│    • Cols: 100 universal ghosts                                    │
│    • Values: 1 if ghost present, 0 otherwise                       │
│                                                                     │
│ 2. Train 3 classifiers:                                            │
│    • Logistic Regression (interpretable baseline)                  │
│    • Random Forest (feature importance)                            │
│    • Gradient Boosting (best performance)                          │
│                                                                     │
│ 3. 5-fold cross-validation                                         │
│                                                                     │
│ 4. Compute metrics:                                                │
│    • Accuracy, Precision, Recall, F1                               │
│    • ROC-AUC (KEY METRIC for Thesis B)                             │
│    • Confusion matrix                                              │
│    • Feature importance                                            │
│                                                                     │
│ Output: 3 ClassifierResults objects                                │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 5: Feature Importance Analysis                              │
│                                                                     │
│ Extract from each classifier:                                      │
│   • Random Forest: clf.feature_importances_                        │
│   • Gradient Boosting: clf.feature_importances_                    │
│   • Logistic Regression: abs(clf.coef_[0])                         │
│                                                                     │
│ Rank features by importance                                        │
│ Identify top 20 most diagnostic ghosts                             │
│                                                                     │
│ Output: Feature importance rankings                                │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT FILES                                                        │
│                                                                     │
│ Data Files:                                                         │
│ • manifest.json              - Experiment metadata                 │
│ • metrics_*.parquet          - Per-sample ghost features           │
│ • universal_ghosts.json      - Top 100 universal ghosts            │
│ • antagonism_analysis.json   - Mechanistic scores                  │
│ • classifier_results.json    - Performance metrics                 │
│                                                                     │
│ Visualization Files (7 PNGs):                                      │
│ • 00_comprehensive_dashboard.png                                   │
│ • 01_ghost_frequency.png                                           │
│ • 02_semantic_interpretation.png                                   │
│ • 03_antagonism_analysis.png                                       │
│ • 04_classifier_performance.png                                    │
│ • 05_feature_importance.png                                        │
│ • 06_confusion_matrices.png                                        │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ THESIS B VALIDATION                                                 │
│                                                                     │
│ IF ROC-AUC > 0.8:                                                   │
│   ✓ STRONG SUPPORT                                                 │
│   → Ghost features are predictive biomarkers                       │
│   → Can build working hallucination detector                       │
│   → Thesis B becomes much stronger                                 │
│                                                                     │
│ IF ROC-AUC 0.7-0.8:                                                 │
│   ○ MODERATE SUPPORT                                               │
│   → Some predictive power                                          │
│   → May need feature engineering                                   │
│                                                                     │
│ IF ROC-AUC < 0.7:                                                   │
│   ✗ WEAK SUPPORT                                                   │
│   → Ghost features may not be predictive                           │
│   → Need alternative approaches                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│   Sample     │
│  (prompt,    │
│   fact,      │
│   halluc)    │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Get Activations                     │
│  • fact_act = get_activations(fact)  │
│  • hall_act = get_activations(hall)  │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Extract Ghost Features              │
│  • fact_set = set(fact_act.indices)  │
│  • hall_set = set(hall_act.indices)  │
│  • ghost_set = hall_set - fact_set   │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Store Per-Sample Record             │
│  {                                   │
│    "ghost_feature_ids": [8383, ...], │
│    "ghost_magnitudes": [15.3, ...],  │
│    "ghost_count": 10                 │
│  }                                   │
└──────┬───────────────────────────────┘
       │
       ↓ (aggregate across all samples)
       │
┌──────────────────────────────────────┐
│  Global Frequency Counter            │
│  {                                   │
│    8383: 45,  # Appears in 45 samples│
│    7475: 38,                         │
│    1234: 32,                         │
│    ...                               │
│  }                                   │
└──────┬───────────────────────────────┘
       │
       ↓ (filter by frequency ≥ 10)
       │
┌──────────────────────────────────────┐
│  Universal Ghosts (top 100)          │
│  [                                   │
│    {feature_id: 8383, freq: 45, ...},│
│    {feature_id: 7475, freq: 38, ...},│
│    ...                               │
│  ]                                   │
└──────┬───────────────────────────────┘
       │
       ↓ (decode to vocabulary)
       │
┌──────────────────────────────────────┐
│  Semantic Interpretation             │
│  Feature #8383:                      │
│    top_words: [" Charles", " Robert",│
│                " George", ...]       │
└──────┬───────────────────────────────┘
       │
       ↓ (compute antagonism)
       │
┌──────────────────────────────────────┐
│  Mechanistic Antagonism              │
│  Feature #8383:                      │
│    avg_dot_with_fact: -0.1234        │
│    antagonism_score: 0.1234          │
└──────┬───────────────────────────────┘
       │
       ↓ (build feature matrix)
       │
┌──────────────────────────────────────┐
│  Feature Matrix (200 × 100)          │
│                                      │
│        F8383  F7475  F1234  ...      │
│  S1     1      0      1     ...      │
│  S2     0      1      1     ...      │
│  S3     1      1      0     ...      │
│  ...   ...    ...    ...    ...      │
└──────┬───────────────────────────────┘
       │
       ↓ (train classifiers)
       │
┌──────────────────────────────────────┐
│  Binary Classifiers                  │
│  • Logistic Regression               │
│  • Random Forest                     │
│  • Gradient Boosting                 │
└──────┬───────────────────────────────┘
       │
       ↓ (evaluate)
       │
┌──────────────────────────────────────┐
│  Classification Metrics              │
│  • Accuracy: 0.85                    │
│  • F1 Score: 0.85                    │
│  • ROC-AUC: 0.89 ← KEY METRIC        │
│  • Feature Importance: {...}         │
└──────────────────────────────────────┘
```

## Timeline

```
Time    Module                          Output
─────   ──────────────────────────────  ────────────────────────────
0:00    Load benchmark + model          HB_Benchmark initialized
        
2:00    MODULE 1: Extract ghosts        200 sample records
                                        2,467 unique ghosts
        
5:00    MODULE 2: Universal ghosts      100 UniversalGhost objects
                                        Semantic interpretations
        
7:00    MODULE 3: Antagonism            100 MechanisticAnalysis
                                        Antagonism scores
        
10:00   MODULE 4: Classification        3 ClassifierResults
                                        ROC-AUC scores
        
12:00   Save results                    5 JSON/Parquet files
        
14:00   Generate visualizations         7 PNG files
        
DONE    ✓ Complete
```

## Decision Tree

```
                    Run Experiment
                          ↓
                    Check ROC-AUC
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
       ROC-AUC > 0.8              ROC-AUC < 0.8
            ↓                           ↓
    ✓ STRONG SUPPORT              Check if > 0.7
    • Write paper section               ↓
    • Run causal intervention   ┌───────┴───────┐
    • Test on other models      ↓               ↓
                           ROC-AUC > 0.7   ROC-AUC < 0.7
                                ↓               ↓
                         ○ MODERATE      ✗ WEAK SUPPORT
                         • Feature eng   • Try geometry
                         • Ensemble      • Alternative
```

## Parallel Processing Opportunities

```
MODULE 1: Extract Ghosts
├── Sample 1 ──┐
├── Sample 2 ──┤
├── Sample 3 ──┼─→ Can process in parallel
├── ...        │
└── Sample 200 ┘

MODULE 2: Decode Ghosts
├── Ghost 1 ──┐
├── Ghost 2 ──┤
├── Ghost 3 ──┼─→ Can decode in parallel
├── ...       │
└── Ghost 100 ┘

MODULE 3: Antagonism
├── Ghost 1 ──┐
├── Ghost 2 ──┤
├── Ghost 3 ──┼─→ Can compute in parallel
├── ...       │
└── Ghost 100 ┘

MODULE 4: Classification
├── Logistic ──┐
├── RF ────────┼─→ Can train in parallel
└── GBM ───────┘
```

## Memory Usage

```
Component                   Memory      Notes
─────────────────────────   ─────────   ──────────────────────
Model (gemma-2-2b)          ~8 GB       Loaded once
SAE (layer 5, 16k)          ~2 GB       Loaded once
Sample records              ~10 MB      200 samples
Universal ghosts            ~1 MB       100 ghosts
Feature matrix              ~80 KB      200×100 binary
Classifiers                 ~5 MB       3 models
─────────────────────────   ─────────   ──────────────────────
TOTAL                       ~10 GB      Peak usage
```

## Error Handling

```
Try:
    Load model
    ↓
    Load SAE
    ↓
    Extract ghosts
    ↓
    Identify universal
    ↓
    Compute antagonism
    ↓
    Train classifiers
    ↓
    Save results

Except ImportError:
    → Install missing packages

Except OutOfMemoryError:
    → Reduce batch size
    → Reduce top_k

Except RuntimeError:
    → Check CUDA/MPS availability
    → Fall back to CPU
```

## Quality Checks

```
After MODULE 1:
✓ All samples processed?
✓ Ghost counts > 0?
✓ Feature IDs valid?

After MODULE 2:
✓ Universal ghosts identified?
✓ Semantic decoding successful?
✓ Top words make sense?

After MODULE 3:
✓ Antagonism scores computed?
✓ Dot products valid?
✓ No NaN values?

After MODULE 4:
✓ Classifiers trained?
✓ ROC-AUC > 0.5 (better than random)?
✓ Feature importance extracted?

After Visualization:
✓ All 7 figures generated?
✓ No rendering errors?
✓ Figures readable?
```

## Success Metrics

```
Metric                  Target      Interpretation
─────────────────────   ─────────   ────────────────────────
ROC-AUC                 > 0.8       Strong support
Feature sparsity        > 80%       Binary features
Cross-val std           < 0.05      Stable performance
Top feature importance  > 0.10      Clear diagnostic ghosts
Antagonism (top 10)     < 0         Oppose truth
Universal ghost freq    ≥ 10        Robust across samples
```



