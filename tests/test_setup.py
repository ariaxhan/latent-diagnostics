#!/usr/bin/env python3
"""
Test Setup and Import Chain

Verifies that all dependencies are installed and imports work correctly.
Run this before executing experiments to catch any issues early.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test all critical imports."""
    print("=" * 80)
    print("TESTING IMPORT CHAIN")
    print("=" * 80)
    print()
    
    tests = []
    
    # Test 1: Core dependencies
    print("Test 1: Core Dependencies")
    print("-" * 80)
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
        tests.append(("torch", True))
    except ImportError as e:
        print(f"❌ torch: {e}")
        tests.append(("torch", False))
    
    try:
        import transformer_lens
        print(f"✅ transformer_lens")
        tests.append(("transformer_lens", True))
    except ImportError as e:
        print(f"❌ transformer_lens: {e}")
        tests.append(("transformer_lens", False))
    
    try:
        import sae_lens
        print(f"✅ sae_lens")
        tests.append(("sae_lens", True))
    except ImportError as e:
        print(f"❌ sae_lens: {e}")
        tests.append(("sae_lens", False))
    
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
        tests.append(("numpy", True))
    except ImportError as e:
        print(f"❌ numpy: {e}")
        tests.append(("numpy", False))
    
    try:
        import polars as pl
        print(f"✅ polars {pl.__version__}")
        tests.append(("polars", True))
    except ImportError as e:
        print(f"❌ polars: {e}")
        tests.append(("polars", False))
    
    try:
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__}")
        tests.append(("matplotlib", True))
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        tests.append(("matplotlib", False))
    
    try:
        import seaborn as sns
        print(f"✅ seaborn {sns.__version__}")
        tests.append(("seaborn", True))
    except ImportError as e:
        print(f"❌ seaborn: {e}")
        tests.append(("seaborn", False))
    
    print()
    
    # Test 2: Package imports
    print("Test 2: Hallucination Detector Package")
    print("-" * 80)
    try:
        from neural_polygraph import (
            initialize_model_and_sae,
            extract_features,
            decode_feature,
            get_loudest_unique_features,
            run_differential_diagnosis,
        )
        print("✅ SAE utilities imported")
        tests.append(("sae_utils", True))
    except ImportError as e:
        print(f"❌ SAE utilities: {e}")
        tests.append(("sae_utils", False))
    
    try:
        from neural_polygraph import (
            GeometricMetrics,
            compute_inertia_tensor,
        )
        print("✅ Geometry module imported")
        tests.append(("geometry", True))
    except ImportError as e:
        print(f"❌ Geometry module: {e}")
        tests.append(("geometry", False))
    
    try:
        from neural_polygraph import (
            BenchmarkSample,
            ActivationResult,
            HB_Benchmark,
        )
        print("✅ Data loader imported")
        tests.append(("data_loader", True))
    except ImportError as e:
        print(f"❌ Data loader: {e}")
        tests.append(("data_loader", False))
    
    try:
        from neural_polygraph import ExperimentStorage
        print("✅ Storage module imported")
        tests.append(("storage", True))
    except ImportError as e:
        print(f"❌ Storage module: {e}")
        tests.append(("storage", False))
    
    print()
    
    # Test 3: Data files
    print("Test 3: Benchmark Data Files")
    print("-" * 80)
    data_dir = Path("experiments/data")
    
    required_files = [
        "bench_entity_swaps.json",
        "bench_temporal_shifts.json",
        "bench_logical_inversions.json",
        "bench_adversarial_traps.json",
    ]
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            import json
            with open(filepath) as f:
                data = json.load(f)
            print(f"✅ {filename} ({len(data)} samples)")
            tests.append((filename, True))
        else:
            print(f"❌ {filename} not found")
            tests.append((filename, False))
    
    print()
    
    # Test 4: Device detection
    print("Test 4: Device Detection")
    print("-" * 80)
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"✅ MPS (Apple Silicon) available")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"✅ CUDA available")
        else:
            device = "cpu"
            print(f"⚠️  CPU only (no GPU acceleration)")
        print(f"   Device: {device}")
        tests.append(("device", True))
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        tests.append(("device", False))
    
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    print(f"Tests passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        print()
        print("Ready to run experiments:")
        print("  python run_experiment.py 01_spectroscopy")
        print("  python run_experiment.py --list")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Failed tests:")
        for name, success in tests:
            if not success:
                print(f"  - {name}")
        print()
        print("Fix issues before running experiments.")
        print()
        print("Install missing dependencies:")
        print("  pip install -e .")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

