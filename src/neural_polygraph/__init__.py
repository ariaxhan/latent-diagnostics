"""
Hallucination Detector Package

A package for detecting hallucinations in language models using
Sparse Autoencoder (SAE) spectral signatures and geometric analysis.

Components:
- sae_utils: Core SAE feature extraction and analysis
- geometry: Geometric/topological analysis of feature activations
- data_loader: HB-1000 benchmark suite loader
- storage: Robust experiment storage with immutable runs
"""

# Core SAE utilities
from .sae_utils import (
    initialize_model_and_sae,
    extract_features,
    decode_feature,
    get_loudest_unique_features,
    run_differential_diagnosis,
)

# Geometric analysis
from .geometry import (
    GeometricMetrics,
    compute_inertia_tensor,
)

# Data loading
from .data_loader import (
    BenchmarkSample,
    ActivationResult,
    HB_Benchmark,
)

# Storage
from .storage import ExperimentStorage

# Feature extractors (injection detection)
from .feature_extractors import (
    SAEExtractor,
    TranscoderExtractor,
    AttentionExtractor,
    extract_combined_features,
    compute_injection_score,
    load_model,
)

# Injection detector
from .injection_detector import (
    AttributionInjectionDetector,
    InjectionMetrics,
    DetectionResult,
    Thresholds,
)

# PINT benchmark loader
from .pint_loader import (
    PINTBenchmark,
    PINTSample,
)

__all__ = [
    # SAE utilities
    "initialize_model_and_sae",
    "extract_features",
    "decode_feature",
    "get_loudest_unique_features",
    "run_differential_diagnosis",
    # Geometry
    "GeometricMetrics",
    "compute_inertia_tensor",
    # Data loading
    "BenchmarkSample",
    "ActivationResult",
    "HB_Benchmark",
    # Storage
    "ExperimentStorage",
    # Feature extractors (injection detection)
    "SAEExtractor",
    "TranscoderExtractor",
    "AttentionExtractor",
    "extract_combined_features",
    "compute_injection_score",
    "load_model",
    # Injection detector
    "AttributionInjectionDetector",
    "InjectionMetrics",
    "DetectionResult",
    "Thresholds",
    # PINT benchmark
    "PINTBenchmark",
    "PINTSample",
]

