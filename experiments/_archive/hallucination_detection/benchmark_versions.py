#!/usr/bin/env python3
"""
Benchmark Script: Compare Sequential vs Parallel Performance

This script helps you determine the optimal batch size and version for your hardware.
"""

import sys
import time
import subprocess
from pathlib import Path
import argparse


def run_benchmark(script_name: str, batch_size: int = None, num_samples: int = 20) -> dict:
    """
    Run a single benchmark and measure performance.
    
    Args:
        script_name: Name of the experiment script
        batch_size: Batch size (for parallel version)
        num_samples: Number of samples to process (for quick test)
        
    Returns:
        Dictionary with timing results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {script_name}")
    if batch_size:
        print(f"Batch Size: {batch_size}")
    print(f"Samples: {num_samples} (quick test)")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [sys.executable, f"experiments/{script_name}"]
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    
    # Run and time
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Parse output for samples/sec if available
        samples_per_sec = None
        for line in result.stdout.split('\n'):
            if 'samples/sec' in line.lower():
                try:
                    # Extract number before "samples/sec"
                    parts = line.split('samples/sec')[0].split()
                    samples_per_sec = float(parts[-1])
                except:
                    pass
        
        return {
            'success': result.returncode == 0,
            'time': elapsed_time,
            'samples_per_sec': samples_per_sec,
            'error': None if result.returncode == 0 else result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'time': 600,
            'samples_per_sec': None,
            'error': 'Timeout (>10 minutes)'
        }
    except Exception as e:
        return {
            'success': False,
            'time': 0,
            'samples_per_sec': None,
            'error': str(e)
        }


def print_results(results: dict):
    """Print benchmark results in a nice table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print()
    
    # Table header
    print(f"{'Version':<30} {'Batch Size':<12} {'Time (s)':<12} {'Samples/sec':<15} {'Status':<10}")
    print("-"*80)
    
    # Table rows
    for key, result in results.items():
        version, batch_size = key if isinstance(key, tuple) else (key, "N/A")
        
        time_str = f"{result['time']:.1f}" if result['success'] else "FAILED"
        rate_str = f"{result['samples_per_sec']:.1f}" if result['samples_per_sec'] else "N/A"
        status = "✓ OK" if result['success'] else "✗ FAIL"
        
        print(f"{version:<30} {str(batch_size):<12} {time_str:<12} {rate_str:<15} {status:<10}")
    
    print()
    
    # Find best performer
    successful = {k: v for k, v in results.items() if v['success']}
    if successful:
        best = min(successful.items(), key=lambda x: x[1]['time'])
        best_key, best_result = best
        
        print("Recommendation:")
        if isinstance(best_key, tuple):
            version, batch_size = best_key
            print(f"  Use: {version} with batch size {batch_size}")
            print(f"  Command: python experiments/{version} --batch-size {batch_size}")
        else:
            print(f"  Use: {best_key}")
            print(f"  Command: python experiments/{best_key}")
        print(f"  Expected time: {best_result['time']:.1f}s for 20 samples")
        print(f"  Estimated full run: {best_result['time'] * 10:.0f}s (~{best_result['time'] * 10 / 60:.1f} minutes)")
    
    print()


def main():
    """Run comprehensive benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark experiment versions")
    parser.add_argument("--quick", action="store_true", help="Quick test (20 samples only)")
    parser.add_argument("--full", action="store_true", help="Full test (all samples)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16],
                       help="Batch sizes to test (default: 4 8 16)")
    args = parser.parse_args()
    
    num_samples = 20 if args.quick else None
    
    print("="*80)
    print("EXPERIMENT VERSION BENCHMARK")
    print("="*80)
    print()
    print("This script will test different versions to find the fastest for your hardware.")
    print()
    
    if args.quick:
        print("Mode: QUICK TEST (20 samples)")
        print("Note: Use --full for complete benchmark with all samples")
    else:
        print("Mode: FULL TEST (all samples)")
        print("Warning: This will take a while!")
    
    print()
    input("Press Enter to start benchmark...")
    
    results = {}
    
    # Test sequential version
    print("\n" + "="*80)
    print("TEST 1: Sequential Version (Baseline)")
    print("="*80)
    
    result = run_benchmark("06_comprehensive_analysis.py", num_samples=num_samples)
    results["06_comprehensive_analysis.py"] = result
    
    if result['success']:
        print(f"✓ Sequential: {result['time']:.1f}s")
    else:
        print(f"✗ Sequential FAILED: {result['error']}")
    
    # Test parallel version with different batch sizes
    print("\n" + "="*80)
    print("TEST 2: Parallel Version (Multiple Batch Sizes)")
    print("="*80)
    
    for batch_size in args.batch_sizes:
        result = run_benchmark(
            "06_comprehensive_analysis_parallel.py",
            batch_size=batch_size,
            num_samples=num_samples
        )
        results[("06_comprehensive_analysis_parallel.py", batch_size)] = result
        
        if result['success']:
            speedup = results["06_comprehensive_analysis.py"]['time'] / result['time']
            print(f"✓ Parallel (batch={batch_size}): {result['time']:.1f}s (speedup: {speedup:.2f}x)")
        else:
            print(f"✗ Parallel (batch={batch_size}) FAILED: {result['error']}")
    
    # Print final results
    print_results(results)
    
    # Save results
    results_file = Path(__file__).parent / "benchmark_results.txt"
    with open(results_file, 'w') as f:
        f.write("BENCHMARK RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for key, result in results.items():
            if isinstance(key, tuple):
                version, batch_size = key
                f.write(f"{version} (batch_size={batch_size}):\n")
            else:
                f.write(f"{key}:\n")
            
            f.write(f"  Time: {result['time']:.1f}s\n")
            f.write(f"  Samples/sec: {result['samples_per_sec']}\n")
            f.write(f"  Success: {result['success']}\n")
            if result['error']:
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()



