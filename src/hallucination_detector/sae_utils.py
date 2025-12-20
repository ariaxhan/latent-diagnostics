"""
SAE Utilities for Hallucination Detection

Core functions for extracting and analyzing feature activations using
Sparse Autoencoders (SAEs) to identify hallucination signatures.
"""

from typing import Dict, List, Tuple
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


def initialize_model_and_sae(device: str = None) -> Tuple[HookedTransformer, SAE, str]:
    """
    Load the language model and SAE analyzer.
    
    Args:
        device: Device to use ('mps', 'cuda', or 'cpu'). Auto-detects if None.
    
    Returns:
        Tuple of (model, sae, device)
    """
    # Auto-detect device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading instruments on device: {device}")
    
    # Load SAE (the "microscope")
    print("  Loading SAE microscope...")
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_5/width_16k/canonical"
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device
    )
    
    # Load model (the "subject")
    print("  Loading Gemma-2-2b model...")
    model_name = "gemma-2-2b"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    print("  ✓ Instruments ready")
    return model, sae, device


def extract_features(text: str, model: HookedTransformer, sae: SAE) -> Dict:
    """
    Extract SAE feature activations from text (the "biopsy").
    
    This function runs the model on input text and extracts the sparse
    feature activations from the SAE. It returns both the active feature
    indices and their magnitudes.
    
    Args:
        text: Input text to analyze
        model: Language model
        sae: Sparse autoencoder
    
    Returns:
        Dictionary with indices, magnitudes, counts, and energy
    """
    # Run model to get activations
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    
    # Extract activation from last token (where prediction happens)
    activations = cache["blocks.5.hook_resid_post"][0, -1, :]
    
    # Apply SAE to get feature activations
    activations = activations.unsqueeze(0)
    feature_acts = sae.encode(activations).squeeze()
    
    # Filter for active features
    active_indices = torch.nonzero(feature_acts > 0).squeeze()
    if active_indices.dim() == 0:  # Handle single element case
        active_indices = active_indices.unsqueeze(0)
    magnitudes = feature_acts[active_indices]
    
    return {
        "indices": active_indices.tolist() if len(active_indices) > 0 else [],
        "magnitudes": magnitudes.tolist() if len(magnitudes) > 0 else [],
        "total_active": len(active_indices),
        "energy": magnitudes.sum().item() if len(magnitudes) > 0 else 0.0
    }


def decode_feature(feature_id: int, model: HookedTransformer, sae: SAE, top_k: int = 5) -> Dict:
    """
    Translate a feature ID into the words it promotes.
    
    This function decodes what a feature "means" by projecting its direction
    in the model's representation space onto the vocabulary. The top-k words
    with highest logits reveal what the feature is detecting.
    
    Args:
        feature_id: SAE feature index
        model: Language model
        sae: Sparse autoencoder
        top_k: Number of top words to return
    
    Returns:
        Dictionary with words, logits, and feature_id
    """
    # Get feature direction in model space
    feature_direction = sae.W_dec[feature_id]
    
    # Project to vocabulary
    logits = model.unembed(feature_direction)
    
    # Get top words
    top_token_ids = logits.argsort(descending=True)[:top_k]
    top_words = model.to_str_tokens(top_token_ids)
    top_logits = logits[top_token_ids].tolist()
    
    return {
        "feature_id": feature_id,
        "words": top_words,
        "logits": top_logits,
    }


def get_loudest_unique_features(
    fact_text: str, 
    hall_text: str, 
    model: HookedTransformer, 
    sae: SAE,
    top_k: int = 5
) -> List[int]:
    """
    Find features that are active in hallucination but not in fact, sorted by magnitude.
    
    This is the core hallucination detection method: we identify features that
    "light up" only in the hallucinated text, not in the factual control. These
    unique features are sorted by activation strength (loudest first).
    
    Args:
        fact_text: Ground truth text
        hall_text: Hallucinated text
        model: Language model
        sae: Sparse autoencoder
        top_k: Number of top features to return
    
    Returns:
        List of feature indices (loudest first)
    """
    # Get activations for both texts
    tokens_fact = model.to_tokens(fact_text)
    _, cache_fact = model.run_with_cache(tokens_fact)
    act_fact = cache_fact["blocks.5.hook_resid_post"][0, -1, :]
    feat_fact = sae.encode(act_fact.unsqueeze(0)).squeeze()
    
    tokens_hall = model.to_tokens(hall_text)
    _, cache_hall = model.run_with_cache(tokens_hall)
    act_hall = cache_hall["blocks.5.hook_resid_post"][0, -1, :]
    feat_hall = sae.encode(act_hall.unsqueeze(0)).squeeze()
    
    # Find unique features (active in hallucination, zero in fact)
    hall_active = (feat_hall > 0)
    fact_inactive = (feat_fact == 0)
    unique_mask = hall_active & fact_inactive
    
    unique_indices = torch.nonzero(unique_mask).squeeze()
    if unique_indices.dim() == 0:
        unique_indices = unique_indices.unsqueeze(0)
    
    if len(unique_indices) == 0:
        return []
    
    unique_magnitudes = feat_hall[unique_indices]
    
    # Sort by magnitude (loudest first)
    sorted_indices = unique_indices[torch.argsort(unique_magnitudes, descending=True)]
    
    return sorted_indices[:top_k].tolist()


def run_differential_diagnosis(
    fact_text: str, 
    hall_text: str, 
    model: HookedTransformer, 
    sae: SAE
) -> Dict:
    """
    Compare feature activations between fact and hallucination.
    
    This function performs a "differential diagnosis" by comparing the
    spectral signatures of factual vs hallucinated text. It identifies
    unique features, missing features, and energy differences.
    
    Args:
        fact_text: Ground truth text
        hall_text: Hallucinated text
        model: Language model
        sae: Sparse autoencoder
    
    Returns:
        Diagnosis dictionary with comparative metrics and biomarkers
    """
    # Get signatures
    sig_fact = extract_features(fact_text, model, sae)
    sig_hall = extract_features(hall_text, model, sae)
    
    # Compare
    set_fact = set(sig_fact["indices"])
    set_hall = set(sig_hall["indices"])
    
    unique_to_hallucination = list(set_hall - set_fact)
    missing_from_hallucination = list(set_fact - set_hall)
    
    return {
        "spectral_metrics": {
            "control_entropy": sig_fact["total_active"],
            "sample_entropy": sig_hall["total_active"],
            "energy_diff": sig_hall["energy"] - sig_fact["energy"]
        },
        "biomarkers": {
            "unique_to_hallucination_count": len(unique_to_hallucination),
            "missing_grounding_count": len(missing_from_hallucination),
            "top_hallucination_features": unique_to_hallucination[:5]
        },
        "signatures": {
            "fact": sig_fact,
            "hallucination": sig_hall,
        }
    }

