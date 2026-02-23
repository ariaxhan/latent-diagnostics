#!/usr/bin/env python3
"""
Prompt Injection Detection via Attribution Graph Geometry

Core thesis: Injection prompts create DIFFUSE causal graphs.
Benign prompts have FOCUSED causal pathways.

Discriminative metrics (Cohen's d > 1.0):
- top_100_concentration: Injection LOWER (scattered influence)
- n_active: Injection HIGHER (more features active)
- n_edges: Injection HIGHER (more connections)
- mean_influence: Injection LOWER (weaker per-connection)

Two-phase workflow:
1. Modal GPU computes attribution metrics → data/results/pint_attribution_metrics.json
2. This script loads metrics, calibrates, evaluates, saves results

Usage:
    # First: Run attribution on Modal
    modal run scripts/modal_pint_benchmark.py

    # Then: Analyze locally
    python experiments/detection.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import ExperimentStorage
from neural_polygraph.injection_detector import (
    AttributionInjectionDetector,
    InjectionMetrics,
    Thresholds,
)


def load_attribution_metrics(metrics_path: Path) -> tuple:
    """
    Load pre-computed attribution metrics from Modal run.

    Returns:
        (metrics_list, labels, categories, raw_data)
    """
    with open(metrics_path, 'r') as f:
        data = json.load(f)

    metrics_list = []
    labels = []
    categories = []

    for sample in data["samples"]:
        m = InjectionMetrics(
            n_active=sample.get("n_active", 0),
            n_edges=sample.get("n_edges", 0),
            mean_influence=sample.get("mean_influence", 0),
            max_influence=sample.get("max_influence", 0),
            top_100_concentration=sample.get("top_100_concentration", 0),
            mean_activation=sample.get("mean_activation", 0),
            max_activation=sample.get("max_activation", 0),
            logit_entropy=sample.get("logit_entropy", 0),
            text=sample.get("text", "")[:100],
        )
        metrics_list.append(m)
        labels.append(sample["label"])
        categories.append(sample.get("category", "unknown"))

    return metrics_list, labels, categories, data


def run_injection_detection_experiment(metrics_path: Path = None):
    """
    Run injection detection experiment.

    If metrics_path is None, looks for latest Modal output.
    """
    print("=" * 80)
    print("EXPERIMENT 08: PROMPT INJECTION DETECTION")
    print("Target: Beat Lakera Guard's 95.22% PINT Score")
    print("=" * 80)
    print()

    # Initialize storage
    experiment_path = Path(__file__).parent / "injection_detection"
    storage = ExperimentStorage(experiment_path)

    # Find metrics file
    if metrics_path is None:
        # Look in standard locations
        possible_paths = [
            Path(__file__).parent.parent / "data" / "results" / "pint_attribution_metrics.json",
            Path(__file__).parent.parent / "pint_metrics.json",  # fallback
        ]
        for p in possible_paths:
            if p.exists():
                metrics_path = p
                break

        if metrics_path is None:
            print("ERROR: No metrics file found.")
            print()
            print("Run attribution analysis first:")
            print("  modal run scripts/modal_pint_benchmark.py")
            print()
            print("Expected output: pint_metrics.json")
            return None

    print("STEP 1: Loading Attribution Metrics")
    print("-" * 80)
    print(f"Metrics file: {metrics_path}")

    metrics_list, labels, categories, raw_data = load_attribution_metrics(metrics_path)

    n_injection = sum(labels)
    n_benign = len(labels) - n_injection
    print(f"✓ Loaded {len(metrics_list)} samples")
    print(f"  Injection: {n_injection} ({100*n_injection/len(labels):.1f}%)")
    print(f"  Benign: {n_benign} ({100*n_benign/len(labels):.1f}%)")
    print()

    # Split into train/test (70/30)
    print("STEP 2: Train/Test Split")
    print("-" * 80)

    import random
    random.seed(42)
    indices = list(range(len(metrics_list)))
    random.shuffle(indices)

    split_idx = int(0.7 * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_metrics = [metrics_list[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_metrics = [metrics_list[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"Train: {len(train_metrics)} samples")
    print(f"Test: {len(test_metrics)} samples")
    print()

    # Calibrate detector on training set
    print("STEP 3: Calibrating Detector")
    print("-" * 80)

    detector = AttributionInjectionDetector()
    calibration_result = detector.calibrate(
        train_metrics,
        train_labels,
        optimize_for="f1",
    )

    print("Optimal thresholds:")
    for k, v in calibration_result["thresholds"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Training F1: {calibration_result['best_score']:.4f}")
    print()

    # Evaluate on test set
    print("STEP 4: Evaluating on Test Set")
    print("-" * 80)

    test_results = detector.evaluate(test_metrics, test_labels)

    print("Test Set Results:")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print()
    print(f"  PINT SCORE: {test_results['pint_score']:.2f}%")
    print()

    # Compare to baselines
    print("STEP 5: Comparison to Baselines")
    print("-" * 80)

    baselines = {
        "Lakera Guard": 95.22,
        "AWS Bedrock Guardrails": 89.24,
        "Azure AI Prompt Shield": 89.12,
        "protectai/deberta-v3": 79.14,
        "Llama Prompt Guard 2": 78.76,
        "Google Model Armor": 70.07,
    }

    our_score = test_results['pint_score']

    for name, score in baselines.items():
        delta = our_score - score
        status = "✅ BEAT" if delta > 0 else "❌ BEHIND"
        print(f"  {name}: {score:.2f}% ({status} by {abs(delta):.2f}%)")

    print()

    # Evaluate by category (if available)
    if categories:
        print("STEP 6: Per-Category Analysis")
        print("-" * 80)

        unique_cats = sorted(set(categories))
        for cat in unique_cats:
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            if not cat_indices:
                continue

            cat_metrics = [metrics_list[i] for i in cat_indices]
            cat_labels = [labels[i] for i in cat_indices]

            cat_results = detector.evaluate(cat_metrics, cat_labels)
            print(f"  {cat}: {cat_results['pint_score']:.2f}% (n={len(cat_indices)})")
        print()

    # Save results
    print("STEP 7: Saving Results")
    print("-" * 80)

    # Prepare metrics for Parquet
    results_data = {
        "sample_id": list(range(len(metrics_list))),
        "label": labels,
        "category": categories,
        "n_active": [m.n_active for m in metrics_list],
        "n_edges": [m.n_edges for m in metrics_list],
        "mean_influence": [m.mean_influence for m in metrics_list],
        "top_100_concentration": [m.top_100_concentration for m in metrics_list],
        "prediction": [detector.classify(m).is_injection for m in metrics_list],
        "confidence": [detector.classify(m).confidence for m in metrics_list],
        "correct": [
            detector.classify(m).is_injection == l
            for m, l in zip(metrics_list, labels)
        ],
    }

    # Add train/test split indicator
    split_indicator = ["test"] * len(metrics_list)
    for i in train_indices:
        split_indicator[i] = "train"
    results_data["split"] = split_indicator

    # Save manifest
    manifest = {
        "experiment_type": "injection_detection",
        "experiment_name": "08_injection_detection",
        "description": "Prompt injection detection using attribution graph analysis",
        "method": "attribution_graph_metrics",
        "model": "gemma-2-2b",
        "transcoder_set": "gemma",
        "total_samples": len(metrics_list),
        "train_samples": len(train_metrics),
        "test_samples": len(test_metrics),
        "metrics": [
            "n_active", "n_edges", "mean_influence",
            "top_100_concentration", "prediction", "confidence"
        ],
        "thresholds": calibration_result["thresholds"],
        "test_results": test_results,
        "baselines": baselines,
        "timestamp": datetime.now().isoformat(),
    }
    storage.write_manifest(manifest)

    # Save metrics as Parquet
    storage.write_metrics(results_data)

    # Save detector
    detector_path = storage.run_path / "detector.json"
    detector.save(detector_path)

    # Save detailed results
    storage.write_results_json({
        "calibration": calibration_result,
        "test_results": test_results,
        "per_category": {
            cat: detector.evaluate(
                [metrics_list[i] for i, c in enumerate(categories) if c == cat],
                [labels[i] for i, c in enumerate(categories) if c == cat]
            )
            for cat in set(categories) if categories
        },
    }, "detailed_results.json")

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: {storage.run_path}")
    print()
    print("Next Steps:")
    print("  1. Visualize: python experiments/visualize_injection_detection.py")
    print("  2. Run on full PINT benchmark for official score")
    print()

    return storage


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to pint_metrics.json from Modal run"
    )
    args = parser.parse_args()

    run_injection_detection_experiment(metrics_path=args.metrics)
