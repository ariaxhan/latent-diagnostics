"""Hallucination Detector: SAE-based spectral signature analysis"""

from .sae_utils import (
    initialize_model_and_sae,
    extract_features,
    decode_feature,
    get_loudest_unique_features,
    run_differential_diagnosis,
)

__version__ = "0.1.0"

__all__ = [
    "initialize_model_and_sae",
    "extract_features",
    "decode_feature",
    "get_loudest_unique_features",
    "run_differential_diagnosis",
]

