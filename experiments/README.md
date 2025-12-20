# Experiments: Hallucination Detection Research

This directory contains the research experiments demonstrating SAE-based hallucination detection.

## Purpose

These experiments implement the methodology described in the Medium article series, providing:
- Reproducible research code
- Real experimental results
- Validation of the hallucination detection approach

## Main Experiment

### `hallucination_biopsy.py`

The core experiment that demonstrates hallucination detection using SAE feature analysis.

**What it does:**
1. Loads Gemma-2-2b model and GemmaScope SAE
2. Defines fact/hallucination test pairs
3. Extracts and compares feature activations
4. Identifies unique "hallucination biomarkers"
5. Decodes features to interpret their meaning
6. Saves results to JSON

**Test cases:**
- Geography: "Eiffel Tower in Paris" vs "Eiffel Tower in Rome"
- (Single example for speed - see full protocol for more)

## Running the Experiment

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run

```bash
# From repository root
python experiments/hallucination_biopsy.py
```

### Expected Output

```
============================================================
HALLUCINATION BIOPSY EXPERIMENT
============================================================

STEP 1: Loading Model and SAE
------------------------------------------------------------
Loading instruments on device: mps
  Loading SAE microscope...
  Loading Gemma-2-2b model...
  ✓ Instruments ready

STEP 2: Defining Test Case
------------------------------------------------------------
FACT:          'The Eiffel Tower is located in Paris'
HALLUCINATION: 'The Eiffel Tower is located in Rome'

STEP 3: Running Differential Diagnosis
------------------------------------------------------------
Spectral Metrics:
  Control entropy (fact):       245
  Sample entropy (hallucination): 238
  Energy difference:            +116.143

Biomarkers:
  Unique to hallucination: 73
  Missing from hallucination: 80

STEP 4: Identifying Loudest Hallucination Signatures
------------------------------------------------------------
Found 5 unique features (sorted by activation strength):

STEP 5: Decoding Feature Meanings
------------------------------------------------------------
  1. Feature #9958 → RB, RSD, RCS
  2. Feature #891  → ...
  ...

STEP 6: Saving Results
------------------------------------------------------------
Results saved to: results/biopsy_20241219_150000.json

============================================================
EXPERIMENT COMPLETE
============================================================
```

### Runtime

- **First run:** 5-10 minutes (includes model download)
- **Subsequent runs:** 1-2 minutes
- **Memory:** ~8GB RAM
- **Storage:** ~5GB for models (cached)

## Results

Results are saved to `results/` directory in JSON format:

```json
{
  "experiment_type": "hallucination_biopsy",
  "timestamp": {...},
  "setup": {
    "device": "mps",
    "model": "gemma-2-2b",
    "sae": "gemma-scope-2b-pt-res-canonical/layer_5/width_16k/canonical"
  },
  "test_case": {
    "fact": "The Eiffel Tower is located in Paris",
    "hallucination": "The Eiffel Tower is located in Rome"
  },
  "diagnosis": {
    "spectral_metrics": {...},
    "biomarkers": {...}
  },
  "loudest_features": {
    "indices": [9958, 891, ...],
    "translations": [...]
  }
}
```

## Understanding the Results

### Spectral Metrics

- **Control entropy:** Number of active features in factual text
- **Sample entropy:** Number of active features in hallucination
- **Energy difference:** Total activation magnitude difference

### Biomarkers

- **Unique to hallucination:** Features that only activate for false information
- **Missing from hallucination:** Grounding features absent in hallucination
- **Top features:** Strongest hallucination-specific activations

### Feature Translations

Each feature is decoded by projecting onto the vocabulary:
- Shows top 3-5 words the feature promotes
- Reveals semantic meaning of the feature
- Helps interpret what the model is "thinking"

## Extending the Experiments

Want to try more test cases? Modify `hallucination_biopsy.py`:

```python
# Add your own test cases
test_cases = [
    {
        "fact": "Your factual statement",
        "hallucination": "Your hallucinated version"
    },
    # Add more...
]
```

Suggested categories:
- **Geography:** Location errors
- **History:** Temporal anachronisms
- **Biology:** Impossible capabilities
- **Mathematics:** Logical inversions
- **Science:** Physical impossibilities

## Reference: Full Protocol

For the complete batch analysis with 5+ test cases, see:
- `/experiments/specimens/2024_12_19_hallucination_biopsy_gemma2/protocol.py` (in experiments repo)

This simplified version focuses on a single example for clarity and speed.

## System Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- 10GB disk space
- CPU (slow but works)

### Recommended
- Python 3.10+
- 16GB RAM
- 10GB disk space
- Apple Silicon (MPS) or NVIDIA GPU (CUDA)

## Troubleshooting

**"Model download failed"**
- Check internet connection
- Verify Hugging Face Hub access
- Try: `huggingface-cli login`

**"Out of memory"**
- Close other applications
- Use CPU instead of GPU
- Reduce batch size (already minimal)

**"Results not saving"**
- Check `results/` directory exists
- Verify write permissions
- Check disk space

## Citation

If you use this experiment in your research:

```bibtex
@misc{han2024hallucination,
  author = {Han, Aria},
  title = {Hallucination Biopsy: SAE-Based Detection},
  year = {2024},
  url = {https://github.com/yourusername/neural-polygraph}
}
```

## Related Work

- **SAE Lens:** https://github.com/jbloomAus/SAELens
- **TransformerLens:** https://github.com/neelnanda-io/TransformerLens
- **Neuronpedia:** https://neuronpedia.org/gemma-2b
- **GemmaScope:** https://huggingface.co/google/gemma-scope

---

**Questions?** See the main README or open an issue on GitHub.

