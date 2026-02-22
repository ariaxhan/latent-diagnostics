"""
Feature Extractors for Injection Detection

Unified interface for:
1. SAE features (existing)
2. Transcoder features (new)
3. Attention head patterns (new - Attention Tracker style)
"""

from typing import Dict, Protocol, Tuple, List, Optional
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_lens import SAE


class FeatureExtractor(Protocol):
    """Protocol for feature extractors (SAE or Transcoder)."""
    def encode(self, activations: torch.Tensor) -> torch.Tensor: ...


class SAEExtractor:
    """Extract features using Sparse Autoencoder."""

    def __init__(
        self,
        device: str = None,
        layer: int = 5,
        width: str = "16k"
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.layer = layer
        self.hook_point = f"blocks.{layer}.hook_resid_post"

        # Load SAE
        sae_release = "gemma-scope-2b-pt-res-canonical"
        sae_id = f"layer_{layer}/width_{width}/canonical"
        self.sae = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        return self.sae.encode(activations).squeeze()

    @property
    def name(self) -> str:
        return f"SAE-L{self.layer}"


class TranscoderExtractor:
    """
    Extract features using Transcoder (MLP input -> MLP output).

    Transcoders from Gemma Scope are trained on:
    - Input: activations just AFTER the pre-MLP RMSNorm
    - Output: MLP sublayer output

    The pre-MLP RMSNorm gains are folded into the transcoder weights,
    so we use hook_resid_mid (residual stream after attention, before MLP).
    """

    def __init__(
        self,
        device: str = None,
        layer: int = 5,
        width: str = "16k",
        l0: str = "average_l0_153"
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.layer = layer

        # Transcoders operate on MLP input
        # hook_resid_mid = residual stream after attention, before MLP
        # The RMSNorm is folded into the transcoder weights
        self.hook_point = f"blocks.{layer}.hook_resid_mid"

        # Load Transcoder (same API as SAE via sae-lens)
        tc_release = "gemma-scope-2b-pt-transcoders"
        tc_id = f"layer_{layer}/width_{width}/{l0}"

        print(f"  Loading transcoder: {tc_release}/{tc_id}")
        self.transcoder = SAE.from_pretrained(
            release=tc_release,
            sae_id=tc_id,
            device=device
        )

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode MLP input activations to sparse features."""
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        return self.transcoder.encode(activations).squeeze()

    @property
    def name(self) -> str:
        return f"Transcoder-L{self.layer}"


class AttentionExtractor:
    """
    Extract attention patterns for injection detection.

    Based on Attention Tracker (NAACL 2025):
    - Identifies "important heads" that shift attention from system to injected instructions
    - Measures attention disagreement across heads
    """

    def __init__(self, model: HookedTransformer, layers: List[int] = None):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads

        # Default: analyze middle layers (where most processing happens)
        if layers is None:
            mid = self.n_layers // 2
            self.layers = list(range(max(0, mid - 3), min(self.n_layers, mid + 4)))
        else:
            self.layers = layers

    def extract_attention_patterns(
        self,
        cache: dict,
        system_token_range: Optional[Tuple[int, int]] = None,
        user_token_range: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Extract attention-based features from model cache.

        Args:
            cache: TransformerLens cache from run_with_cache
            system_token_range: (start, end) indices for system prompt tokens
            user_token_range: (start, end) indices for user input tokens

        Returns:
            Dict with attention coherence metrics
        """
        all_patterns = []
        head_entropies = []

        for layer in self.layers:
            # Shape: [batch, n_heads, seq_len, seq_len]
            attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

            if attn_pattern.dim() == 4:
                attn_pattern = attn_pattern[0]  # Remove batch dim

            all_patterns.append(attn_pattern)

            # Compute per-head entropy (measure of attention spread)
            # High entropy = diffuse attention, low entropy = focused attention
            for head in range(self.n_heads):
                head_attn = attn_pattern[head, -1, :]  # Last token attending to all
                entropy = -(head_attn * (head_attn + 1e-10).log()).sum().item()
                head_entropies.append(entropy)

        # Compute head agreement/disagreement
        # Stack all final-token attention patterns: [n_layers * n_heads, seq_len]
        final_attns = torch.stack([
            cache[f"blocks.{l}.attn.hook_pattern"][0, h, -1, :]
            for l in self.layers
            for h in range(self.n_heads)
        ])

        # Pairwise cosine similarity between heads
        similarities = F.cosine_similarity(
            final_attns.unsqueeze(0),
            final_attns.unsqueeze(1),
            dim=2
        )

        # Coherence = mean pairwise similarity (high = heads agree, low = disagreement)
        coherence = similarities.mean().item()
        coherence_std = similarities.std().item()

        # Attention to system vs user regions (if ranges provided)
        system_attention = None
        user_attention = None
        attention_ratio = None

        if system_token_range and user_token_range:
            sys_start, sys_end = system_token_range
            usr_start, usr_end = user_token_range

            # Average attention to each region across all heads
            system_attention = final_attns[:, sys_start:sys_end].mean().item()
            user_attention = final_attns[:, usr_start:usr_end].mean().item()

            # Ratio: high = more attention to user, low = more attention to system
            if system_attention > 0:
                attention_ratio = user_attention / system_attention

        return {
            "coherence": coherence,
            "coherence_std": coherence_std,
            "mean_entropy": sum(head_entropies) / len(head_entropies),
            "entropy_std": torch.tensor(head_entropies).std().item(),
            "system_attention": system_attention,
            "user_attention": user_attention,
            "attention_ratio": attention_ratio,
            "n_heads_analyzed": len(head_entropies),
        }

    @property
    def name(self) -> str:
        return f"Attention-L{self.layers}"


def load_model(device: str = None) -> Tuple[HookedTransformer, str]:
    """Load Gemma-2-2B model."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading Gemma-2-2B on {device}...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    return model, device


def extract_combined_features(
    text: str,
    model: HookedTransformer,
    sae_extractor: Optional[FeatureExtractor] = None,
    transcoder_extractor: Optional[FeatureExtractor] = None,
    attention_extractor: Optional[AttentionExtractor] = None,
    system_token_range: Optional[Tuple[int, int]] = None,
    user_token_range: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Extract all feature types from text.

    Returns combined features from SAE, transcoder, and attention patterns.
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)

    results = {
        "tokens": tokens.shape[1],
    }

    # SAE features (residual stream)
    if sae_extractor:
        sae_acts = cache[sae_extractor.hook_point][0, -1, :]
        sae_features = sae_extractor.encode(sae_acts)
        results["sae_features"] = sae_features
        results["sae_l0"] = (sae_features > 0).sum().item()
        results["sae_energy"] = sae_features.sum().item()

    # Transcoder features (MLP path)
    if transcoder_extractor:
        tc_acts = cache[transcoder_extractor.hook_point][0, -1, :]
        tc_features = transcoder_extractor.encode(tc_acts)
        results["transcoder_features"] = tc_features
        results["tc_l0"] = (tc_features > 0).sum().item()
        results["tc_energy"] = tc_features.sum().item()

    # Attention patterns
    if attention_extractor:
        attn_metrics = attention_extractor.extract_attention_patterns(
            cache,
            system_token_range=system_token_range,
            user_token_range=user_token_range
        )
        results["attention"] = attn_metrics

    return results


def compute_injection_score(
    baseline_features: Dict,
    prompt_features: Dict,
    weights: Dict = None
) -> Dict:
    """
    Compute injection risk score from feature comparison.

    Combines:
    - SAE/Transcoder cosine similarity to baseline
    - Attention coherence metrics

    Args:
        baseline_features: Features from system prompt
        prompt_features: Features from user input
        weights: Optional weights for combining signals

    Returns:
        Dict with risk score and component scores
    """
    if weights is None:
        weights = {
            "sae_similarity": 0.3,
            "tc_similarity": 0.3,
            "attention_coherence": 0.2,
            "attention_ratio": 0.2,
        }

    scores = {}

    # SAE similarity (lower = more different from baseline = higher risk)
    if "sae_features" in baseline_features and "sae_features" in prompt_features:
        sae_sim = F.cosine_similarity(
            baseline_features["sae_features"].unsqueeze(0),
            prompt_features["sae_features"].unsqueeze(0)
        ).item()
        scores["sae_similarity"] = sae_sim
        scores["sae_risk"] = 1 - sae_sim

    # Transcoder similarity
    if "transcoder_features" in baseline_features and "transcoder_features" in prompt_features:
        tc_sim = F.cosine_similarity(
            baseline_features["transcoder_features"].unsqueeze(0),
            prompt_features["transcoder_features"].unsqueeze(0)
        ).item()
        scores["tc_similarity"] = tc_sim
        scores["tc_risk"] = 1 - tc_sim

    # Attention coherence (lower = more head disagreement = higher risk)
    if "attention" in prompt_features:
        coherence = prompt_features["attention"]["coherence"]
        scores["attention_coherence"] = coherence
        scores["attention_risk"] = 1 - coherence

        # Attention ratio (if available)
        ratio = prompt_features["attention"].get("attention_ratio")
        if ratio is not None:
            # High ratio = attention shifted to user input = higher risk
            scores["attention_ratio"] = min(ratio, 5.0) / 5.0  # Cap and normalize

    # Combined risk score
    risk_components = []

    if "sae_risk" in scores:
        risk_components.append(scores["sae_risk"] * weights.get("sae_similarity", 0.3))

    if "tc_risk" in scores:
        risk_components.append(scores["tc_risk"] * weights.get("tc_similarity", 0.3))

    if "attention_risk" in scores:
        risk_components.append(scores["attention_risk"] * weights.get("attention_coherence", 0.2))

    if "attention_ratio" in scores:
        risk_components.append(scores["attention_ratio"] * weights.get("attention_ratio", 0.2))

    # Normalize by actual weights used
    total_weight = sum(weights.get(k, 0) for k in ["sae_similarity", "tc_similarity", "attention_coherence", "attention_ratio"])

    if risk_components and total_weight > 0:
        scores["combined_risk"] = sum(risk_components) / total_weight
    else:
        scores["combined_risk"] = 0.5  # Default uncertain

    scores["recommendation"] = "BLOCK" if scores["combined_risk"] > 0.6 else "ALLOW"

    return scores
