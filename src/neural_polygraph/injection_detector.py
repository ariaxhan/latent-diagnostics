"""
Attribution-Based Injection Detector

Detects prompt injection using attribution graph metrics.

Key insight: Injection creates DIFFUSE causal graphs (many features, scattered influence).
Benign prompts have FOCUSED causal pathways.

Discriminative metrics (Cohen's d > 1.0 from pilot study):
- top_100_concentration: Injection LOWER (influence is scattered)
- n_active: Injection HIGHER (more features activate)
- n_edges: Injection HIGHER (more causal connections)
- mean_influence: Injection LOWER (per-edge influence is weaker)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class InjectionMetrics:
    """Metrics extracted from attribution graph."""
    # Core metrics (from circuit-tracer)
    n_active: int = 0
    n_edges: int = 0
    mean_influence: float = 0.0
    max_influence: float = 0.0
    top_100_concentration: float = 0.0

    # Activation metrics
    mean_activation: float = 0.0
    max_activation: float = 0.0

    # Logit metrics
    logit_entropy: float = 0.0
    max_logit_prob: float = 0.0

    # Raw text for reference
    text: str = ""

    def to_dict(self) -> Dict:
        return {
            "n_active": self.n_active,
            "n_edges": self.n_edges,
            "mean_influence": self.mean_influence,
            "max_influence": self.max_influence,
            "top_100_concentration": self.top_100_concentration,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "logit_entropy": self.logit_entropy,
            "max_logit_prob": self.max_logit_prob,
            "text": self.text[:100],  # Truncate for storage
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "InjectionMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DetectionResult:
    """Result of injection detection."""
    is_injection: bool
    confidence: float  # 0-1 score
    scores: Dict[str, float] = field(default_factory=dict)
    metrics: Optional[InjectionMetrics] = None

    def to_dict(self) -> Dict:
        return {
            "is_injection": self.is_injection,
            "confidence": self.confidence,
            "scores": self.scores,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


@dataclass
class Thresholds:
    """
    Classification thresholds derived from calibration.

    Default values from pilot study (n=6).
    Should be recalibrated on larger dataset.
    """
    # Injection tends to have MORE active features
    n_active_high: float = 20000
    n_active_weight: float = 1.0

    # Injection tends to have LOWER concentration
    concentration_low: float = 0.002
    concentration_weight: float = 1.5  # Most discriminative

    # Injection tends to have LOWER mean influence
    mean_influence_low: float = 0.005
    mean_influence_weight: float = 1.0

    # Injection tends to have MORE edges
    n_edges_high: float = 40_000_000
    n_edges_weight: float = 0.5

    # Classification threshold (sum of weighted scores)
    decision_threshold: float = 2.0

    def to_dict(self) -> Dict:
        return {
            "n_active_high": self.n_active_high,
            "n_active_weight": self.n_active_weight,
            "concentration_low": self.concentration_low,
            "concentration_weight": self.concentration_weight,
            "mean_influence_low": self.mean_influence_low,
            "mean_influence_weight": self.mean_influence_weight,
            "n_edges_high": self.n_edges_high,
            "n_edges_weight": self.n_edges_weight,
            "decision_threshold": self.decision_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Thresholds":
        return cls(**d)


class AttributionInjectionDetector:
    """
    Injection detector using attribution graph analysis.

    Usage:
        detector = AttributionInjectionDetector()

        # From pre-computed metrics (fast, no GPU needed)
        result = detector.classify(metrics)

        # Or with threshold calibration
        detector.calibrate(labeled_metrics)
        result = detector.classify(metrics)
    """

    def __init__(self, thresholds: Optional[Thresholds] = None):
        self.thresholds = thresholds or Thresholds()
        self._calibrated = False

    def classify(self, metrics: InjectionMetrics) -> DetectionResult:
        """
        Classify a single sample based on its metrics.

        Returns DetectionResult with is_injection, confidence, and component scores.
        """
        scores = {}

        # Score 1: High activation count suggests injection
        if metrics.n_active > self.thresholds.n_active_high:
            scores["n_active"] = self.thresholds.n_active_weight
        else:
            scores["n_active"] = 0.0

        # Score 2: Low concentration suggests injection (most discriminative)
        if metrics.top_100_concentration < self.thresholds.concentration_low:
            scores["concentration"] = self.thresholds.concentration_weight
        else:
            scores["concentration"] = 0.0

        # Score 3: Low mean influence suggests injection
        if metrics.mean_influence < self.thresholds.mean_influence_low:
            scores["mean_influence"] = self.thresholds.mean_influence_weight
        else:
            scores["mean_influence"] = 0.0

        # Score 4: High edge count suggests injection
        if metrics.n_edges > self.thresholds.n_edges_high:
            scores["n_edges"] = self.thresholds.n_edges_weight
        else:
            scores["n_edges"] = 0.0

        # Total score
        total_score = sum(scores.values())
        max_possible = sum([
            self.thresholds.n_active_weight,
            self.thresholds.concentration_weight,
            self.thresholds.mean_influence_weight,
            self.thresholds.n_edges_weight,
        ])

        # Decision
        is_injection = total_score >= self.thresholds.decision_threshold
        confidence = total_score / max_possible

        return DetectionResult(
            is_injection=is_injection,
            confidence=confidence,
            scores=scores,
            metrics=metrics,
        )

    def classify_batch(self, metrics_list: List[InjectionMetrics]) -> List[DetectionResult]:
        """Classify multiple samples."""
        return [self.classify(m) for m in metrics_list]

    def calibrate(
        self,
        metrics_list: List[InjectionMetrics],
        labels: List[bool],
        optimize_for: str = "f1",
    ) -> Dict:
        """
        Calibrate thresholds using labeled data.

        Args:
            metrics_list: List of InjectionMetrics from attribution analysis
            labels: True = injection, False = benign
            optimize_for: "f1", "precision", "recall", or "balanced_accuracy"

        Returns:
            Dict with calibration results and optimal thresholds
        """
        import numpy as np

        # Extract metric arrays
        n_active = np.array([m.n_active for m in metrics_list])
        concentration = np.array([m.top_100_concentration for m in metrics_list])
        mean_influence = np.array([m.mean_influence for m in metrics_list])
        n_edges = np.array([m.n_edges for m in metrics_list])
        labels_arr = np.array(labels)

        # Find optimal thresholds via grid search
        best_score = 0
        best_thresholds = None

        # Grid search over percentiles
        for n_active_pct in [50, 60, 70, 80]:
            for conc_pct in [20, 30, 40, 50]:
                for inf_pct in [20, 30, 40, 50]:
                    # Set thresholds at percentiles
                    t = Thresholds(
                        n_active_high=float(np.percentile(n_active, n_active_pct)),
                        concentration_low=float(np.percentile(concentration, conc_pct)),
                        mean_influence_low=float(np.percentile(mean_influence, inf_pct)),
                        n_edges_high=float(np.percentile(n_edges, n_active_pct)),
                    )

                    # Test these thresholds
                    self.thresholds = t
                    predictions = [self.classify(m).is_injection for m in metrics_list]

                    score = self._compute_score(predictions, labels, optimize_for)

                    if score > best_score:
                        best_score = score
                        best_thresholds = t

        # Set best thresholds (use default if grid search failed)
        if best_thresholds is not None:
            self.thresholds = best_thresholds
        else:
            self.thresholds = Thresholds()
        self._calibrated = True

        # Final evaluation
        predictions = [self.classify(m).is_injection for m in metrics_list]
        final_metrics = self._compute_all_metrics(predictions, labels)

        return {
            "thresholds": best_thresholds.to_dict(),
            "optimized_for": optimize_for,
            "best_score": best_score,
            "metrics": final_metrics,
        }

    def _compute_score(
        self,
        predictions: List[bool],
        labels: List[bool],
        metric: str,
    ) -> float:
        """Compute optimization metric."""
        tp = sum(p and l for p, l in zip(predictions, labels))
        fp = sum(p and not l for p, l in zip(predictions, labels))
        fn = sum(not p and l for p, l in zip(predictions, labels))
        tn = sum(not p and not l for p, l in zip(predictions, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if metric == "f1":
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == "precision":
            return precision
        elif metric == "recall":
            return recall
        elif metric == "balanced_accuracy":
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return (sensitivity + specificity) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _compute_all_metrics(
        self,
        predictions: List[bool],
        labels: List[bool],
    ) -> Dict:
        """Compute all classification metrics."""
        tp = sum(p and l for p, l in zip(predictions, labels))
        fp = sum(p and not l for p, l in zip(predictions, labels))
        fn = sum(not p and l for p, l in zip(predictions, labels))
        tn = sum(not p and not l for p, l in zip(predictions, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "pint_score": accuracy * 100,  # PINT benchmark uses percentage
        }

    def evaluate(
        self,
        metrics_list: List[InjectionMetrics],
        labels: List[bool],
    ) -> Dict:
        """
        Evaluate detector on labeled data.

        Returns dict with precision, recall, F1, accuracy, and PINT score.
        """
        predictions = [self.classify(m).is_injection for m in metrics_list]
        return self._compute_all_metrics(predictions, labels)

    def save(self, path: Path) -> None:
        """Save detector configuration (thresholds) to JSON."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump({
                "thresholds": self.thresholds.to_dict(),
                "calibrated": self._calibrated,
            }, f, indent=2)
        print(f"✓ Detector saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "AttributionInjectionDetector":
        """Load detector from JSON."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        detector = cls(thresholds=Thresholds.from_dict(data["thresholds"]))
        detector._calibrated = data.get("calibrated", False)
        return detector

    def __repr__(self) -> str:
        status = "calibrated" if self._calibrated else "default"
        return f"AttributionInjectionDetector({status})"
