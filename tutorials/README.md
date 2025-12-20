# Tutorials: Learning SAE-Based Hallucination Detection

This directory contains educational Jupyter notebooks that teach the fundamentals of using Sparse Autoencoders (SAEs) for hallucination detection.

## Purpose

These tutorials are designed to:
- Introduce SAEs and their role in interpretability
- Teach practical feature extraction techniques
- Build intuition for hallucination detection methodology
- Provide hands-on experience with real models

## Prerequisites

- Basic Python knowledge
- Familiarity with machine learning concepts (helpful but not required)
- Curiosity about how language models work internally

## Notebook Order

### 1. `01_sae_basics.ipynb` - SAE Basics: The Prism for Language Models

**What you'll learn:**
- What Sparse Autoencoders are and why they matter
- The "prism metaphor" for understanding SAEs
- How to load a pre-trained model and SAE
- How to extract and decode feature activations

**Estimated time:** 10-15 minutes

**Key concepts:**
- Dense vs sparse representations
- Feature activation patterns
- Vocabulary projection for feature interpretation

---

### 2. `02_feature_extraction.ipynb` - Feature Extraction and Comparison

**What you'll learn:**
- How to use the `hallucination_detector` package
- Techniques for comparing features between texts
- How to identify unique and shared features
- The foundation of differential diagnosis

**Estimated time:** 15-20 minutes

**Key concepts:**
- Feature set comparison
- Unique vs shared features
- Spectral signatures
- Hallucination biomarkers

---

## Running the Notebooks

### Setup

Make sure you've installed the package and dependencies:

```bash
# From the repository root
pip install -r requirements.txt
pip install -e .
```

### Launch Jupyter

```bash
jupyter notebook tutorials/
```

Or use JupyterLab:

```bash
jupyter lab tutorials/
```

### First Run Notes

- **Model Download:** The first run will download ~5GB of models (Gemma-2-2b + SAE)
- **Device:** Automatically uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- **Runtime:** Each notebook takes 10-20 minutes to complete
- **Memory:** Requires ~8GB RAM minimum

## After Completing Tutorials

Once you've finished both notebooks, you're ready to:

1. **Run the full experiment:** `python experiments/hallucination_biopsy.py`
2. **Explore the codebase:** Check out `src/hallucination_detector/`
3. **Read the articles:** See the Medium series for detailed analysis
4. **Experiment:** Try your own fact/hallucination pairs!

## Troubleshooting

### Common Issues

**"Model download is slow"**
- First run downloads ~5GB, subsequent runs use cache
- Check your internet connection
- Models cache in `~/.cache/huggingface/`

**"Out of memory"**
- Close other applications
- Use a smaller batch size (already optimized in tutorials)
- Consider using a machine with more RAM

**"Import errors"**
- Make sure you ran `pip install -e .` from the repo root
- Verify virtual environment is activated
- Check Python version is 3.10+

**"MPS/CUDA not working"**
- The code automatically falls back to CPU
- MPS requires macOS 12.3+ with Apple Silicon
- CUDA requires NVIDIA GPU with proper drivers

## Feedback

Found an issue or have suggestions? Please:
- Open an issue on GitHub
- Suggest improvements via PR
- Share your experiments!

---

**Next Steps:** Start with `01_sae_basics.ipynb` and work through sequentially.

