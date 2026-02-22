"""
Storage Layer for Neural Polygraph Experiments

Adapted from the experiments repo storage system to provide:
- Multiple runs per experiment with immutable timestamped directories
- Parquet for metrics (efficient columnar storage)
- Msgpack for metadata (compact binary format)
- Complete audit trail of all experiment runs
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import polars as pl


class ExperimentStorage:
    """
    Manages storage for hallucination detection experiments.
    
    Each experiment can have multiple runs, with each run stored in its own
    timestamped directory. This enables reproducibility, comparison, and
    complete history tracking.
    
    Structure:
        experiments/{experiment_name}/
        ├── protocol.py              # Reusable experiment script
        ├── field_notes.md           # Shared notes
        └── runs/                    # All experiment runs
            ├── 20251227_120000/     # Run 1
            │   ├── metrics.parquet
            │   └── manifest.json
            ├── 20251227_130000/     # Run 2
            │   ├── metrics.parquet
            │   └── manifest.json
            └── ...
    """
    
    def __init__(self, experiment_path: Path, run_id: Optional[str] = None) -> None:
        """
        Initialize experiment storage.
        
        Args:
            experiment_path: Path to experiment directory
            run_id: Optional run ID (timestamp). If None, creates a new run.
        """
        self.experiment_path = Path(experiment_path)
        self.runs_path = self.experiment_path / "runs"
        
        # Create runs directory if it doesn't exist
        self.runs_path.mkdir(parents=True, exist_ok=True)
        
        # Determine current run directory
        if run_id is None:
            # Create new run with timestamp
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.run_path = self.runs_path / run_id
        
        # Create run directory
        self.run_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Experiment storage initialized")
        print(f"  Experiment: {self.experiment_path.name}")
        print(f"  Run ID: {self.run_id}")
        print(f"  Run path: {self.run_path}")
    
    def write_manifest(self, metadata: Dict[str, Any], overwrite: bool = False) -> Path:
        """
        Save experiment metadata as JSON manifest.
        
        Args:
            metadata: Dictionary with experiment configuration and metadata
            overwrite: If True, overwrite existing file. If False, raise error.
            
        Returns:
            Path to the saved manifest file
        """
        manifest_path = self.run_path / "manifest.json"
        
        # Never overwrite within a run (each run is immutable)
        if manifest_path.exists() and not overwrite:
            raise RuntimeError(
                f"Manifest already exists at {manifest_path}. "
                f"Each run is immutable. Create a new run instead."
            )
        
        # Add run metadata
        metadata = metadata.copy()
        metadata["run_id"] = self.run_id
        metadata["run_timestamp"] = datetime.now().isoformat()
        
        with open(manifest_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Manifest saved: {manifest_path.name}")
        return manifest_path
    
    def write_metrics(self, data: Dict[str, List], overwrite: bool = False) -> Path:
        """
        Save metrics as Parquet file using Polars.
        
        Args:
            data: Dictionary with lists of values (will be converted to DataFrame)
            overwrite: If True, overwrite existing file. If False, raise error.
            
        Returns:
            Path to the saved metrics file
        """
        metrics_path = self.run_path / "metrics.parquet"
        
        # Never overwrite within a run (each run is immutable)
        if metrics_path.exists() and not overwrite:
            raise RuntimeError(
                f"Metrics already exist at {metrics_path}. "
                f"Each run is immutable. Create a new run instead."
            )
        
        # Convert dict to Polars DataFrame
        df = pl.DataFrame(data)
        
        # Add run_id column to metrics
        df = df.with_columns(pl.lit(self.run_id).alias("run_id"))
        
        # Save as Parquet
        df.write_parquet(metrics_path)
        
        print(f"✓ Metrics saved: {metrics_path.name} ({len(df)} rows)")
        return metrics_path
    
    def write_results_json(self, results: Dict[str, Any], filename: str = "results.json") -> Path:
        """
        Save additional results as JSON (for complex nested data).
        
        Args:
            results: Dictionary with results
            filename: Name of the results file
            
        Returns:
            Path to the saved results file
        """
        results_path = self.run_path / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {filename}")
        return results_path
    
    def read_metrics(self, run_id: Optional[str] = None) -> pl.DataFrame:
        """
        Load metrics from a specific run.
        
        Args:
            run_id: Optional run ID. If None, reads from current run.
        
        Returns:
            Polars DataFrame with metrics
        """
        if run_id is None:
            metrics_path = self.run_path / "metrics.parquet"
        else:
            metrics_path = self.runs_path / run_id / "metrics.parquet"
        
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
        
        return pl.read_parquet(metrics_path)
    
    def read_manifest(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load manifest from a specific run.
        
        Args:
            run_id: Optional run ID. If None, reads from current run.
        
        Returns:
            Dictionary with manifest data
        """
        if run_id is None:
            manifest_path = self.run_path / "manifest.json"
        else:
            manifest_path = self.runs_path / run_id / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def list_runs(self) -> List[str]:
        """
        List all run IDs for this experiment.
        
        Returns:
            List of run IDs (timestamp strings), sorted newest first
        """
        if not self.runs_path.exists():
            return []
        
        runs = [
            d.name for d in self.runs_path.iterdir()
            if d.is_dir() and (d / "metrics.parquet").exists()
        ]
        return sorted(runs, reverse=True)
    
    def get_latest_run(self) -> Optional[str]:
        """
        Get the most recent run ID.
        
        Returns:
            Run ID of most recent run, or None if no runs exist
        """
        runs = self.list_runs()
        return runs[0] if runs else None
    
    def read_all_runs(self) -> pl.DataFrame:
        """
        Load metrics from all runs combined.
        
        Returns:
            Combined DataFrame with metrics from all runs (includes run_id column)
        """
        runs = self.list_runs()
        if not runs:
            raise FileNotFoundError(f"No runs found in {self.runs_path}")
        
        dfs = []
        for run_id in runs:
            run_path = self.runs_path / run_id / "metrics.parquet"
            if run_path.exists():
                df = pl.read_parquet(run_path)
                # Ensure run_id column exists
                if "run_id" not in df.columns:
                    df = df.with_columns(pl.lit(run_id).alias("run_id"))
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No metrics files found in any runs")
        
        return pl.concat(dfs)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about this experiment.
        
        Returns:
            Dictionary with experiment summary
        """
        runs = self.list_runs()
        
        summary = {
            "experiment_name": self.experiment_path.name,
            "experiment_path": str(self.experiment_path),
            "current_run_id": self.run_id,
            "total_runs": len(runs),
            "all_runs": runs,
            "latest_run": runs[0] if runs else None,
        }
        
        # Try to get row counts from each run
        if runs:
            run_sizes = {}
            for run_id in runs:
                try:
                    df = self.read_metrics(run_id)
                    run_sizes[run_id] = len(df)
                except:
                    run_sizes[run_id] = None
            summary["run_sizes"] = run_sizes
        
        return summary
    
    def __repr__(self) -> str:
        runs = self.list_runs()
        return (
            f"ExperimentStorage("
            f"experiment={self.experiment_path.name}, "
            f"run_id={self.run_id}, "
            f"total_runs={len(runs)})"
        )

