"""
Data Loader for HB-1000 Benchmark Suite
Hallucination Benchmark for SAE Analysis

This module provides infrastructure for loading the benchmark datasets
and interfacing with the Gemma-2-2B model and GemmaScope SAEs.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE


@dataclass
class BenchmarkSample:
    """A single test case from the benchmark."""
    id: str
    domain: str
    complexity: int
    prompt: str
    fact: str
    hallucination: str
    
    def get_fact_text(self) -> str:
        """Get the complete factual text."""
        return f"{self.prompt} {self.fact}"
    
    def get_hallucination_text(self) -> str:
        """Get the complete hallucinated text."""
        return f"{self.prompt} {self.hallucination}"


@dataclass
class ActivationResult:
    """Results from SAE feature extraction."""
    residual_stream: torch.Tensor
    feature_indices: List[int]
    feature_magnitudes: List[float]
    reconstruction_error: float
    l0_norm: int
    l2_norm: float


class HB_Benchmark:
    """
    Hallucination Benchmark Suite Loader
    
    Manages the HB-1000 benchmark dataset and provides utilities for
    extracting SAE activations from the Gemma-2-2B model.
    """
    
    DATASET_FILES = {
        "entity": "bench_entity_swaps.json",
        "temporal": "bench_temporal_shifts.json",
        "logical": "bench_logical_inversions.json",
        "adversarial": "bench_adversarial_traps.json",
    }
    
    def __init__(self, data_dir: str = "experiments/data", device: Optional[str] = None):
        """
        Initialize the benchmark loader.
        
        Args:
            data_dir: Directory containing the benchmark JSON files
            device: Device to use ('mps', 'cuda', or 'cpu'). Auto-detects if None.
        """
        self.data_dir = Path(data_dir)
        self.device = self._get_device(device)
        self.samples: Dict[str, List[BenchmarkSample]] = {}
        self.model: Optional[HookedTransformer] = None
        self.sae: Optional[SAE] = None
        self.sae_layer: int = 5  # Layer 5 for initial experiments
        
    def _get_device(self, device: Optional[str]) -> str:
        """Auto-detect or validate device."""
        if device is None:
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_datasets(self, domains: Optional[List[str]] = None) -> None:
        """
        Load benchmark datasets from JSON files.
        
        Args:
            domains: List of domains to load. If None, loads all domains.
        """
        if domains is None:
            domains = list(self.DATASET_FILES.keys())
        
        print(f"Loading benchmark datasets from {self.data_dir}")
        
        for domain in domains:
            if domain not in self.DATASET_FILES:
                raise ValueError(f"Unknown domain: {domain}. Valid domains: {list(self.DATASET_FILES.keys())}")
            
            filepath = self.data_dir / self.DATASET_FILES[domain]
            
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.samples[domain] = [
                BenchmarkSample(**item) for item in data
            ]
            
            print(f"  ✓ Loaded {len(self.samples[domain])} samples from {domain}")
        
        total = sum(len(samples) for samples in self.samples.values())
        print(f"Total samples loaded: {total}")
    
    def load_model_and_sae(
        self, 
        model_name: str = "gemma-2-2b",
        sae_release: str = "gemma-scope-2b-pt-res-canonical",
        layer: int = 5,
        width: str = "16k"
    ) -> None:
        """
        Load Gemma-2-2B model and corresponding GemmaScope SAE.
        
        Args:
            model_name: Name of the transformer model
            sae_release: SAE release identifier
            layer: Layer number for SAE (5, 12, or 20)
            width: SAE width ('16k' or '65k')
        """
        print(f"Loading model and SAE on device: {self.device}")
        
        # Load SAE first
        print(f"  Loading SAE (layer {layer}, width {width})...")
        sae_id = f"layer_{layer}/width_{width}/canonical"
        self.sae = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=self.device
        )
        self.sae_layer = layer
        
        # Load model
        print(f"  Loading {model_name}...")
        self.model = HookedTransformer.from_pretrained(
            model_name, 
            device=self.device
        )
        
        print("  ✓ Model and SAE ready")
    
    def get_activations(self, text: str) -> ActivationResult:
        """
        Extract SAE feature activations for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ActivationResult with all activation metrics
        """
        if self.model is None or self.sae is None:
            raise RuntimeError("Model and SAE must be loaded first. Call load_model_and_sae().")
        
        # Run model to get residual stream activations
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(tokens)
        
        # Extract activation from last token position
        hook_name = f"blocks.{self.sae_layer}.hook_resid_post"
        residual_stream = cache[hook_name][0, -1, :]  # (d_model,)
        
        # Apply SAE encoder to get feature activations
        residual_stream_batched = residual_stream.unsqueeze(0)  # (1, d_model)
        feature_acts = self.sae.encode(residual_stream_batched).squeeze()  # (n_features,)
        
        # Reconstruct and compute error
        reconstructed = self.sae.decode(feature_acts.unsqueeze(0)).squeeze()
        reconstruction_error_vec = residual_stream - reconstructed
        reconstruction_error = torch.norm(reconstruction_error_vec, p=2).item()
        
        # Filter for active features (threshold at 0)
        active_mask = feature_acts > 0
        active_indices = torch.nonzero(active_mask).squeeze()
        
        if active_indices.dim() == 0:  # Single element
            active_indices = active_indices.unsqueeze(0)
        
        if len(active_indices) > 0:
            magnitudes = feature_acts[active_indices]
            indices_list = active_indices.tolist()
            magnitudes_list = magnitudes.tolist()
            l0_norm = len(indices_list)
            l2_norm = torch.norm(magnitudes, p=2).item()
        else:
            indices_list = []
            magnitudes_list = []
            l0_norm = 0
            l2_norm = 0.0
        
        return ActivationResult(
            residual_stream=residual_stream,
            feature_indices=indices_list,
            feature_magnitudes=magnitudes_list,
            reconstruction_error=reconstruction_error,
            l0_norm=l0_norm,
            l2_norm=l2_norm
        )
    
    def get_all_samples(self) -> List[Tuple[str, BenchmarkSample]]:
        """
        Get all samples across all loaded domains.
        
        Returns:
            List of (domain, sample) tuples
        """
        all_samples = []
        for domain, samples in self.samples.items():
            for sample in samples:
                all_samples.append((domain, sample))
        return all_samples
    
    def get_samples_by_domain(self, domain: str) -> List[BenchmarkSample]:
        """Get all samples for a specific domain."""
        if domain not in self.samples:
            raise ValueError(f"Domain '{domain}' not loaded. Available: {list(self.samples.keys())}")
        return self.samples[domain]
    
    def __len__(self) -> int:
        """Total number of samples across all domains."""
        return sum(len(samples) for samples in self.samples.values())
    
    def __repr__(self) -> str:
        domains_info = ", ".join(f"{k}: {len(v)}" for k, v in self.samples.items())
        return f"HB_Benchmark(domains=[{domains_info}], device={self.device})"

