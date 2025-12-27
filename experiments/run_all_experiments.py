#!/usr/bin/env python3
"""
Master Script: Run All Experiments

Executes the complete experimental pipeline for the paper:
"The Shape of Errors: Spectroscopic and Geometric Analysis of SAE Features"

Experiments:
  A. Pure Spectroscopy (01_spectroscopy.py)
  B. Geometric Topology (02_geometry.py)
  C. Ghost Features (03_ghost_features.py)
  D. Layer Sensitivity (04_layer_sensitivity.py)

Usage:
  python experiments/run_all_experiments.py [--quick]
  
  --quick: Run with reduced sample size for testing
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_experiment(script_name: str, description: str):
    """
    Run a single experiment script.
    
    Args:
        script_name: Name of the experiment script
        description: Human-readable description
    """
    print()
    print("=" * 80)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("=" * 80)
    print()
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print()
        print(f"✓ {description} COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ {description} FAILED")
        print(f"Error: {e}")
        return False


def run_visualization(script_name: str, description: str):
    """
    Run a visualization script.
    
    Args:
        script_name: Name of the visualization script
        description: Human-readable description
    """
    print()
    print("-" * 80)
    print(f"Generating: {description}")
    print("-" * 80)
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    """Run all experiments in sequence."""
    parser = argparse.ArgumentParser(
        description="Run all experiments for the hallucination detection paper"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced sample size for testing"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("=" * 80)
    print("HALLUCINATION DETECTION EXPERIMENTAL PIPELINE")
    print("The Shape of Errors: Spectroscopic and Geometric Analysis")
    print("=" * 80)
    print()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        print("Mode: QUICK (reduced samples)")
    else:
        print("Mode: FULL")
    
    print()
    
    # Track results
    results = {}
    
    # Experiment A: Spectroscopy
    results['A'] = run_experiment(
        "01_spectroscopy.py",
        "Experiment A: Pure Spectroscopy"
    )
    
    if not results['A']:
        print("\n⚠ Experiment A failed. Stopping pipeline.")
        return 1
    
    # Experiment B: Geometry
    results['B'] = run_experiment(
        "02_geometry.py",
        "Experiment B: Geometric Topology"
    )
    
    # Experiment C: Ghost Features
    results['C'] = run_experiment(
        "03_ghost_features.py",
        "Experiment C: Ghost Feature Finder"
    )
    
    # Experiment D: Layer Sensitivity
    results['D'] = run_experiment(
        "04_layer_sensitivity.py",
        "Experiment D: Layer Sensitivity Analysis"
    )
    
    # Generate visualizations
    if not args.skip_viz:
        print()
        print("=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        viz_results = {}
        
        viz_results['spectroscopy'] = run_visualization(
            "visualize_spectroscopy.py",
            "Spectroscopy Visualizations"
        )
        
        if results['B']:
            viz_results['geometry'] = run_visualization(
                "visualize_geometry.py",
                "Geometry Visualizations (Figure 2)"
            )
        
        if results['C']:
            viz_results['ghost'] = run_visualization(
                "visualize_ghost_features.py",
                "Ghost Features Visualizations (Figure 3)"
            )
        
        if results['D']:
            viz_results['layer'] = run_visualization(
                "visualize_layer_sensitivity.py",
                "Layer Sensitivity Visualizations (Figure 4)"
            )
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()
    
    print("Experiment Results:")
    for exp, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  Experiment {exp}: {status}")
    
    print()
    
    # Count successes
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    if successful == total:
        print(f"🎉 All {total} experiments completed successfully!")
        print()
        print("Next Steps:")
        print("  1. Review figures in each experiment's figures/ directory")
        print("  2. Key figures for paper:")
        print("     - Figure 1: experiments/01_spectroscopy/figures/")
        print("     - Figure 2: experiments/02_geometry/figures/fig2_topology_phase_plot.png")
        print("     - Figure 3: experiments/03_ghost_features/figures/fig3_feature_prism.png")
        print("     - Figure 4: experiments/04_layer_sensitivity/figures/fig4_layer_sensitivity.png")
        print()
        return 0
    else:
        print(f"⚠ {total - successful} experiment(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

