# Experiment 07: Ghost Feature Classifier

**Deep Mechanistic Analysis + Predictive Validity**

## Thesis B

> "Ghost features are not just present—they are **predictive biomarkers** of hallucination."

## Overview

This experiment goes **deep** on the ghost feature angle, building on Experiment 03's discovery of features unique to hallucinations. We move beyond mere identification to:

1. **Mechanistic Understanding**: How do ghost features interact with truth?
2. **Predictive Validity**: Can we build a classifier using ghost features?
3. **Feature Importance**: Which ghosts are most diagnostic?

## Experimental Logic

### 1. Ghost Feature Extraction (Per-Sample)
- For each sample, compute: `F_ghost = F_hallucination \ F_fact`
- Store raw feature IDs and magnitudes
- Build global frequency counter

### 2. Universal Ghost Identification
- Identify features appearing in ≥10 samples (configurable)
- Decode to vocabulary space: What do these features represent?
- Analyze domain distribution: Are ghosts domain-specific or universal?

### 3. Mechanistic Antagonism Analysis
- For each ghost feature, get its direction in model space (`W_dec`)
- Compute dot product with fact token embeddings
- **Antagonism Score**: Negative = opposes truth, Positive = aligns
- Hypothesis: Ghosts should show negative correlation with truth

### 4. Binary Classification
- **Input**: Binary feature vector (is ghost X present?)
- **Output**: Hallucination prediction (0/1)
- **Models**:
  - Logistic Regression (interpretable baseline)
  - Random Forest (feature importance)
  - Gradient Boosting (best performance)

### 5. Cross-Validation & Feature Importance
- 5-fold stratified cross-validation
- Extract feature importance scores
- Identify most diagnostic ghost features

## Key Metrics

### Classification Performance
- **Accuracy**: Overall correctness
- **Precision**: When we predict hallucination, how often are we right?
- **Recall**: Of all hallucinations, how many do we catch?
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)

### Thesis B Validation Criteria
- **Strong Support**: ROC-AUC > 0.8
- **Moderate Support**: ROC-AUC > 0.7
- **Weak Support**: ROC-AUC ≤ 0.7

## Expected Outcomes

### If Thesis B is Correct:
1. Classifier achieves ROC-AUC > 0.8
2. Feature importance reveals specific diagnostic ghosts
3. Antagonism analysis shows ghosts oppose truth (negative dot products)
4. Universal ghosts appear across multiple domains

### If Thesis B is Weak:
1. Classifier performs near random (ROC-AUC ≈ 0.5)
2. Feature importance is flat (no clear diagnostic features)
3. Antagonism scores are random (no systematic opposition)
4. Ghost features are sample-specific noise

## Usage

### Run Experiment
```bash
python experiments/07_ghost_classifier.py
```

### Visualize Results
```bash
python experiments/visualize_ghost_classifier.py
```

## Output Structure

```
07_ghost_classifier/
└── runs/
    └── YYYYMMDD_HHMMSS/
        ├── manifest.json              # Experiment metadata
        ├── metrics_*.parquet          # Per-sample ghost features
        ├── universal_ghosts.json      # Top universal ghosts with semantics
        ├── antagonism_analysis.json   # Mechanistic antagonism scores
        ├── classifier_results.json    # Classification performance
        └── figures/
            ├── 00_comprehensive_dashboard.png
            ├── 01_ghost_frequency.png
            ├── 02_semantic_interpretation.png
            ├── 03_antagonism_analysis.png
            ├── 04_classifier_performance.png
            ├── 05_feature_importance.png
            └── 06_confusion_matrices.png
```

## Interpretation Guide

### Universal Ghosts
- **High Frequency**: Appears in many samples → robust biomarker
- **Domain Distribution**: Cross-domain → general hallucination signal
- **Semantic Interpretation**: What concepts do ghosts represent?

### Antagonism Analysis
- **Negative Scores**: Ghost opposes truth → mechanistic antagonism
- **Positive Scores**: Ghost aligns with truth → spurious correlation
- **Zero Scores**: Ghost is orthogonal → independent signal

### Feature Importance
- **High Importance**: Ghost is diagnostic for classification
- **Low Importance**: Ghost is present but not predictive
- **Top Features**: Most reliable biomarkers

## Connection to Paper

This experiment directly supports **Thesis B** in the paper:

> "While aggregate geometric metrics fail to distinguish hallucinations, a different signal emerges: hallucinations consistently activate features absent from factual completions. These 'ghost features' constitute a novel hallucination biomarker."

### Paper Sections:
- **Methods**: Ghost feature extraction methodology
- **Results Part 2**: Ghost feature analysis (main focus)
  - Distribution across domains
  - Semantic decoding
  - Feature clustering
  - **NEW**: Predictive validity (classifier results)
  - **NEW**: Mechanistic antagonism
- **Discussion**: Ghost features as "activation leakage"
- **Proposed Detection Mechanism**: Binary feature masking

### Strengthening the Thesis:
- **Before**: "Ghost features are present in hallucinations"
- **After**: "Ghost features are **predictive** biomarkers with **mechanistic antagonism** to truth"

## Future Directions

1. **Causal Intervention**: Suppress ghost features during generation
2. **Layer Analysis**: Do ghosts appear at specific layers?
3. **Temporal Dynamics**: When do ghosts activate during generation?
4. **Transfer Learning**: Do ghosts generalize to other models?
5. **Semantic Clustering**: Do ghosts form coherent semantic clusters?

## Notes

- All samples are hallucinations (label=1) by design
- To test on factual samples, need to add negative class
- Current classifier tests: "Can we predict hallucination from ghost presence?"
- Future extension: "Can we distinguish fact vs. hallucination?"



