#!/usr/bin/env python3
"""
Quick script to load and explore experiment results.

Usage:
    python experiments/load_results.py 01_spectroscopy
    python experiments/load_results.py 01_spectroscopy --run 20251228_210046
    python experiments/load_results.py 01_spectroscopy --all-runs
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_polygraph import ExperimentStorage
import polars as pl


def explore_data(df: pl.DataFrame, run_id: str = None):
    """Print summary statistics for the data."""
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\nSchema:")
    print(df.schema)
    
    # Domain breakdown
    if "domain" in df.columns:
        print(f"\nSamples by domain:")
        domain_counts = df.group_by("domain").agg(pl.count().alias("count"))
        print(domain_counts)
    
    # Condition breakdown
    if "condition" in df.columns:
        print(f"\nSamples by condition:")
        condition_counts = df.group_by("condition").agg(pl.count().alias("count"))
        print(condition_counts)
        
        # Compare fact vs hallucination
        print(f"\nComparison: Fact vs Hallucination")
        comparison = df.group_by("condition").agg([
            pl.mean("l0_norm").alias("mean_l0"),
            pl.mean("l2_norm").alias("mean_l2"),
            pl.mean("gini_coefficient").alias("mean_gini"),
            pl.mean("reconstruction_error").alias("mean_recon_error"),
        ])
        print(comparison)
    
    # Domain × Condition breakdown
    if "domain" in df.columns and "condition" in df.columns:
        print(f"\nSamples by domain × condition:")
        cross_tab = df.group_by(["domain", "condition"]).agg(pl.count().alias("count"))
        print(cross_tab)
    
    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head(5))
    
    # Statistical summary
    numeric_cols = [col for col, dtype in df.schema.items() 
                    if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    if numeric_cols:
        print(f"\nStatistical summary:")
        print(df.select(numeric_cols).describe())


def main():
    parser = argparse.ArgumentParser(
        description="Load and explore experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/load_results.py 01_spectroscopy
  python experiments/load_results.py 01_spectroscopy --run 20251228_210046
  python experiments/load_results.py 01_spectroscopy --all-runs
  python experiments/load_results.py 01_spectroscopy --export results.csv
        """
    )
    
    parser.add_argument(
        "experiment",
        help="Experiment ID (e.g., 01_spectroscopy)"
    )
    parser.add_argument(
        "--run",
        metavar="RUN_ID",
        help="Specific run ID to load (default: latest)"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Load all runs combined"
    )
    parser.add_argument(
        "--export",
        metavar="FILENAME",
        help="Export to CSV file"
    )
    parser.add_argument(
        "--pandas",
        action="store_true",
        help="Convert to pandas DataFrame (for interactive use)"
    )
    
    args = parser.parse_args()
    
    # Initialize storage
    experiment_path = Path("experiments") / args.experiment
    
    if not experiment_path.exists():
        print(f"Error: Experiment '{args.experiment}' not found at {experiment_path}")
        return
    
    try:
        storage = ExperimentStorage(experiment_path, run_id=args.run)
        
        # Get summary
        summary = storage.get_summary()
        print("=" * 80)
        print(f"EXPERIMENT: {args.experiment}")
        print("=" * 80)
        print(f"Experiment Path: {summary['experiment_path']}")
        print(f"Total Runs: {summary['total_runs']}")
        print(f"Latest Run: {summary['latest_run']}")
        
        if summary['total_runs'] > 0:
            print(f"\nAll Runs:")
            for run_id in summary['all_runs']:
                size = summary.get('run_sizes', {}).get(run_id, 'unknown')
                marker = " ←" if run_id == storage.run_id else ""
                print(f"  - {run_id} ({size} rows){marker}")
        
        # Load data
        if args.all_runs:
            print(f"\nLoading all runs combined...")
            df = storage.read_all_runs()
            print(f"✓ Loaded {len(df)} rows from {summary['total_runs']} runs")
        else:
            print(f"\nLoading run: {storage.run_id}")
            df = storage.read_metrics()
            print(f"✓ Loaded {len(df)} rows")
        
        # Explore data
        explore_data(df, storage.run_id)
        
        # Export if requested
        if args.export:
            df.write_csv(args.export)
            print(f"\n✓ Exported to {args.export}")
        
        # Return pandas if requested
        if args.pandas:
            df_pandas = df.to_pandas()
            print(f"\n✓ Converted to pandas DataFrame")
            print(f"  Use: df = df_pandas  # in your script")
            return df_pandas
        
        # Return polars DataFrame for interactive use
        print(f"\n" + "=" * 80)
        print("TIP: Use this script programmatically:")
        print("=" * 80)
        print("""
from pathlib import Path
from neural_polygraph import ExperimentStorage

storage = ExperimentStorage(Path("experiments/01_spectroscopy"))
df = storage.read_metrics()

# Now use df (Polars DataFrame)
facts = df.filter(pl.col("condition") == "fact")
hallucinations = df.filter(pl.col("condition") == "hallucination")
        """)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nAvailable runs:")
        try:
            runs = storage.list_runs()
            if runs:
                for run_id in runs:
                    print(f"  - {run_id}")
            else:
                print("  No runs found. Run the experiment first:")
                print(f"    python experiments/{args.experiment}.py")
        except:
            pass
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

