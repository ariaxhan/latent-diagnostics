#!/usr/bin/env python3
"""
Neural Polygraph CLI - Unified Interface for Experiment Management

●cli|unified:true|research_optimized:true|dual_mode:interactive_fast
→features|auto_discover:experiments|run:manage|access:data|export:batch|compare:analyze
→design|modular:clean|intuitive:menus|fast:commands|flexible:control
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich import box
    from rich.text import Text
except ImportError:
    print("Installing rich library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich import box
    from rich.text import Text

try:
    from hallucination_detector.storage import ExperimentStorage
except ImportError:
    from storage import ExperimentStorage

import polars as pl

console = Console()


class ExperimentRegistry:
    """Auto-discovers and manages all experiments."""
    
    def __init__(self, experiments_root: Path = Path("experiments")):
        self.experiments_root = experiments_root
        self.experiments = self._discover_experiments()
    
    def _discover_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Scan experiments directory and catalog all experiments."""
        experiments = {}
        
        if not self.experiments_root.exists():
            return experiments
        
        for item in self.experiments_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                has_runs = (item / "runs").exists()
                has_script = (self.experiments_root / f"{item.name}.py").exists()
                
                if has_runs or has_script:
                    exp_info = self._analyze_experiment(item)
                    experiments[item.name] = exp_info
        
        return experiments
    
    def _analyze_experiment(self, exp_path: Path) -> Dict[str, Any]:
        """Analyze an experiment directory and extract metadata."""
        info = {
            "name": exp_path.name,
            "path": exp_path,
            "has_runs": False,
            "run_count": 0,
            "latest_run": None,
            "total_samples": 0,
            "has_script": False,
            "script_path": None,
        }
        
        script_path = self.experiments_root / f"{exp_path.name}.py"
        if script_path.exists():
            info["has_script"] = True
            info["script_path"] = script_path
        
        runs_path = exp_path / "runs"
        if runs_path.exists():
            info["has_runs"] = True
            try:
                # List runs without creating new ones
                runs = sorted([
                    d.name for d in runs_path.iterdir()
                    if d.is_dir() and (d / "metrics.parquet").exists()
                ], reverse=True)
                
                info["run_count"] = len(runs)
                info["latest_run"] = runs[0] if runs else None
                
                if runs:
                    try:
                        # Use latest run to get sample count
                        storage = ExperimentStorage(exp_path, run_id=runs[0])
                        df = storage.read_all_runs()
                        info["total_samples"] = len(df)
                    except:
                        pass
            except:
                pass
        
        return info
    
    def get_experiment(self, name: str) -> Optional[Dict[str, Any]]:
        return self.experiments.get(name)
    
    def list_experiments(self) -> List[str]:
        return sorted(self.experiments.keys())
    
    def refresh(self):
        self.experiments = self._discover_experiments()


class NeuralPolygraphCLI:
    """Unified CLI for Neural Polygraph experiments."""
    
    def __init__(self):
        self.registry = ExperimentRegistry()
        self.current_experiment = None
        self.current_storage = None
    
    # ========== Display Methods ==========
    
    def display_header(self, subtitle: str = ""):
        """Display application header."""
        console.clear()
        title = "[bold cyan]Neural Polygraph CLI[/bold cyan]"
        if subtitle:
            title += f"\n[dim]{subtitle}[/dim]"
        header = Panel(title, box=box.DOUBLE, border_style="cyan")
        console.print(header)
        console.print()
    
    def display_experiments_table(self):
        """Display experiments overview table."""
        table = Table(
            title="📊 Experiments",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("#", style="dim", no_wrap=True, width=3)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Runs", justify="right", style="green", width=6)
        table.add_column("Samples", justify="right", style="yellow", width=8)
        table.add_column("Latest Run", style="dim")
        table.add_column("Status", justify="center", width=6)
        
        for i, name in enumerate(self.registry.list_experiments(), 1):
            exp = self.registry.get_experiment(name)
            
            run_count = str(exp["run_count"]) if exp["run_count"] > 0 else "-"
            sample_count = str(exp["total_samples"]) if exp["total_samples"] > 0 else "-"
            latest = exp["latest_run"][:16] if exp["latest_run"] else "-"
            
            status = "✓" if exp["has_runs"] else "○"
            status_style = "green" if exp["has_runs"] else "dim"
            
            table.add_row(
                str(i),
                name,
                run_count,
                sample_count,
                latest,
                f"[{status_style}]{status}[/{status_style}]"
            )
        
        if not self.registry.experiments:
            console.print("[yellow]No experiments found in experiments/ directory[/yellow]")
        else:
            console.print(table)
    
    # ========== Interactive Mode ==========
    
    def interactive_mode(self):
        """Launch interactive menu mode."""
        self.main_menu()
    
    def main_menu(self):
        """Main interactive menu."""
        self.display_header("Interactive Mode")
        self.display_experiments_table()
        
        console.print("\n[bold]Main Menu[/bold]")
        console.print("  [cyan]1[/cyan] → Select Experiment")
        console.print("  [cyan]2[/cyan] → Run Experiment")
        console.print("  [cyan]3[/cyan] → Batch Export All")
        console.print("  [cyan]4[/cyan] → Compare Experiments")
        console.print("  [cyan]r[/cyan] → Refresh")
        console.print("  [cyan]q[/cyan] → Quit")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]", choices=["1", "2", "3", "4", "r", "q"])
        
        if choice == "1":
            self.select_experiment_menu()
        elif choice == "2":
            self.run_experiment_menu()
        elif choice == "3":
            self.batch_export_interactive()
        elif choice == "4":
            self.compare_experiments_menu()
        elif choice == "r":
            self.registry.refresh()
            console.print("[green]✓ Refreshed[/green]")
            Prompt.ask("\nPress Enter")
            self.main_menu()
        elif choice == "q":
            console.print("\n[cyan]Goodbye![/cyan]")
            return
    
    def select_experiment_menu(self):
        """Select and work with a specific experiment."""
        self.display_header("Select Experiment")
        
        experiments = self.registry.list_experiments()
        if not experiments:
            console.print("[yellow]No experiments found[/yellow]")
            Prompt.ask("\nPress Enter")
            self.main_menu()
            return
        
        for i, name in enumerate(experiments, 1):
            exp = self.registry.get_experiment(name)
            status = "✓" if exp["has_runs"] else "○"
            console.print(f"  [cyan]{i}[/cyan] → {name} [{status}]")
        console.print(f"  [cyan]b[/cyan] → Back")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]")
        
        if choice.lower() == "b":
            self.main_menu()
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                exp_name = experiments[idx]
                self.current_experiment = self.registry.get_experiment(exp_name)
                
                if self.current_experiment["has_runs"]:
                    self.current_storage = ExperimentStorage(self.current_experiment["path"])
                
                self.experiment_menu()
            else:
                console.print("[red]Invalid[/red]")
                Prompt.ask("\nPress Enter")
                self.select_experiment_menu()
        except ValueError:
            console.print("[red]Invalid[/red]")
            Prompt.ask("\nPress Enter")
            self.select_experiment_menu()
    
    def experiment_menu(self):
        """Menu for a specific experiment."""
        if not self.current_experiment:
            self.main_menu()
            return
        
        self.display_header(f"Experiment: {self.current_experiment['name']}")
        
        exp = self.current_experiment
        
        panel = Panel(
            f"[bold]{exp['name']}[/bold]\n"
            f"Runs: {exp['run_count']} | Samples: {exp['total_samples']}\n"
            f"Latest: {exp['latest_run'] or 'N/A'}",
            title="📁 Details",
            border_style="cyan"
        )
        console.print(panel)
        
        console.print("\n[bold]Actions[/bold]")
        console.print("  [cyan]1[/cyan] → View Data")
        console.print("  [cyan]2[/cyan] → Export")
        console.print("  [cyan]3[/cyan] → Run")
        console.print("  [cyan]4[/cyan] → Compare Runs")
        console.print("  [cyan]5[/cyan] → Quick Stats")
        console.print("  [cyan]b[/cyan] → Back")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]", choices=["1", "2", "3", "4", "5", "b"])
        
        if choice == "1":
            self.view_data_menu()
        elif choice == "2":
            self.export_menu()
        elif choice == "3":
            self.run_single_experiment()
        elif choice == "4":
            self.compare_runs_interactive()
        elif choice == "5":
            self.quick_stats_interactive()
        elif choice == "b":
            self.current_experiment = None
            self.current_storage = None
            self.main_menu()
    
    def view_data_menu(self):
        """View data menu."""
        if not self.current_storage:
            console.print("[yellow]No data available[/yellow]")
            Prompt.ask("\nPress Enter")
            self.experiment_menu()
            return
        
        self.display_header(f"View Data: {self.current_experiment['name']}")
        
        console.print("  [cyan]1[/cyan] → Latest Run")
        console.print("  [cyan]2[/cyan] → Specific Run")
        console.print("  [cyan]3[/cyan] → All Runs")
        console.print("  [cyan]b[/cyan] → Back")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]", choices=["1", "2", "3", "b"])
        
        if choice == "1":
            self.show_data(None)
        elif choice == "2":
            self.select_run_to_view()
        elif choice == "3":
            self.show_data(None, all_runs=True)
        elif choice == "b":
            self.experiment_menu()
    
    def show_data(self, run_id: Optional[str], all_runs: bool = False):
        """Display data summary."""
        try:
            # Get the actual latest run if not specified
            if not run_id and not all_runs:
                runs_path = self.current_experiment["path"] / "runs"
                runs = sorted([
                    d.name for d in runs_path.iterdir()
                    if d.is_dir() and (d / "metrics.parquet").exists()
                ], reverse=True)
                if not runs:
                    console.print("[yellow]No runs found[/yellow]")
                    Prompt.ask("\nPress Enter")
                    self.view_data_menu()
                    return
                run_id = runs[0]
            
            # Create temporary storage for reading
            temp_storage = ExperimentStorage(self.current_experiment["path"], run_id=run_id or self.current_experiment["latest_run"])
            
            if all_runs:
                df = temp_storage.read_all_runs()
                title = f"All Runs: {self.current_experiment['name']}"
            elif run_id:
                df = temp_storage.read_metrics(run_id)
                title = f"Run: {run_id}"
            else:
                df = temp_storage.read_metrics()
                title = f"Latest: {run_id}"
            
            self.display_header(title)
            
            console.print(f"[bold]Rows:[/bold] {len(df)}")
            console.print(f"[bold]Columns:[/bold] {', '.join(df.columns)}\n")
            
            if "condition" in df.columns:
                console.print("[bold]By Condition:[/bold]")
                console.print(df.group_by("condition").agg(pl.count().alias("count")))
                console.print()
            
            if "domain" in df.columns:
                console.print("[bold]By Domain:[/bold]")
                console.print(df.group_by("domain").agg(pl.count().alias("count")))
                console.print()
            
            console.print("[bold]Sample (first 5):[/bold]")
            console.print(df.head(5))
            
            Prompt.ask("\nPress Enter")
            self.view_data_menu()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            Prompt.ask("\nPress Enter")
            self.view_data_menu()
    
    def select_run_to_view(self):
        """Select specific run to view."""
        runs = self.current_storage.list_runs()
        
        self.display_header("Select Run")
        
        for i, run_id in enumerate(runs, 1):
            console.print(f"  [cyan]{i}[/cyan] → {run_id}")
        console.print(f"  [cyan]b[/cyan] → Back")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]")
        
        if choice.lower() == "b":
            self.view_data_menu()
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                self.show_data(runs[idx])
            else:
                console.print("[red]Invalid[/red]")
                Prompt.ask("\nPress Enter")
                self.select_run_to_view()
        except ValueError:
            console.print("[red]Invalid[/red]")
            Prompt.ask("\nPress Enter")
            self.select_run_to_view()
    
    def export_menu(self):
        """Export menu."""
        if not self.current_storage:
            console.print("[yellow]No data[/yellow]")
            Prompt.ask("\nPress Enter")
            self.experiment_menu()
            return
        
        self.display_header(f"Export: {self.current_experiment['name']}")
        
        console.print("  [cyan]1[/cyan] → Latest (CSV)")
        console.print("  [cyan]2[/cyan] → Latest (Parquet)")
        console.print("  [cyan]3[/cyan] → All Runs (CSV)")
        console.print("  [cyan]4[/cyan] → All Runs (Parquet)")
        console.print("  [cyan]b[/cyan] → Back")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]", choices=["1", "2", "3", "4", "b"])
        
        if choice == "1":
            self.export_data(None, "csv", False)
        elif choice == "2":
            self.export_data(None, "parquet", False)
        elif choice == "3":
            self.export_data(None, "csv", True)
        elif choice == "4":
            self.export_data(None, "parquet", True)
        elif choice == "b":
            self.experiment_menu()
    
    def export_data(self, run_id: Optional[str], format: str, all_runs: bool):
        """Export data to file."""
        try:
            # Get the actual latest run if not specified
            if not run_id and not all_runs:
                runs_path = self.current_experiment["path"] / "runs"
                runs = sorted([
                    d.name for d in runs_path.iterdir()
                    if d.is_dir() and (d / "metrics.parquet").exists()
                ], reverse=True)
                if not runs:
                    console.print("[yellow]No runs found[/yellow]")
                    Prompt.ask("\nPress Enter")
                    self.export_menu()
                    return
                run_id = runs[0]
            
            if all_runs:
                # Create temporary storage to read all runs
                temp_storage = ExperimentStorage(self.current_experiment["path"], run_id=self.current_experiment["latest_run"])
                df = temp_storage.read_all_runs()
                default_name = f"{self.current_experiment['name']}_all_runs.{format}"
            elif run_id:
                temp_storage = ExperimentStorage(self.current_experiment["path"], run_id=run_id)
                df = temp_storage.read_metrics(run_id)
                default_name = f"{self.current_experiment['name']}_{run_id}.{format}"
            else:
                temp_storage = ExperimentStorage(self.current_experiment["path"], run_id=run_id)
                df = temp_storage.read_metrics()
                default_name = f"{self.current_experiment['name']}_latest.{format}"
            
            filename = Prompt.ask(f"Filename", default=default_name)
            if not filename.endswith(f".{format}"):
                filename += f".{format}"
            
            try:
                if format == "csv":
                    df.write_csv(filename)
                else:
                    df.write_parquet(filename)
                
                console.print(f"[green]✓ Exported {len(df)} rows to {filename}[/green]")
            except Exception as export_error:
                if "nested data" in str(export_error).lower() or "csv" in str(export_error).lower():
                    console.print(f"[yellow]⚠ CSV format doesn't support nested data structures[/yellow]")
                    console.print(f"[cyan]Tip: Use Parquet format instead (option 2 or 4)[/cyan]")
                    
                    # Offer to export as Parquet instead
                    if Confirm.ask("\nExport as Parquet instead?", default=True):
                        parquet_filename = filename.replace('.csv', '.parquet')
                        df.write_parquet(parquet_filename)
                        console.print(f"[green]✓ Exported {len(df)} rows to {parquet_filename}[/green]")
                else:
                    console.print(f"[red]Error: {export_error}[/red]")
            
            Prompt.ask("\nPress Enter")
            self.export_menu()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            Prompt.ask("\nPress Enter")
            self.export_menu()
    
    def run_experiment_menu(self):
        """Run experiment menu."""
        self.display_header("Run Experiment")
        
        experiments = self.registry.list_experiments()
        runnable = [n for n in experiments if self.registry.get_experiment(n)["has_script"]]
        
        if not runnable:
            console.print("[yellow]No runnable experiments[/yellow]")
            Prompt.ask("\nPress Enter")
            self.main_menu()
            return
        
        for i, name in enumerate(runnable, 1):
            console.print(f"  [cyan]{i}[/cyan] → {name}")
        console.print(f"  [cyan]b[/cyan] → Back")
        
        choice = Prompt.ask("\n[bold]Choose[/bold]")
        
        if choice.lower() == "b":
            self.main_menu()
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runnable):
                self.run_experiment_by_name(runnable[idx])
            else:
                console.print("[red]Invalid[/red]")
                Prompt.ask("\nPress Enter")
                self.run_experiment_menu()
        except ValueError:
            console.print("[red]Invalid[/red]")
            Prompt.ask("\nPress Enter")
            self.run_experiment_menu()
    
    def run_single_experiment(self):
        """Run current experiment."""
        if not self.current_experiment or not self.current_experiment["has_script"]:
            console.print("[yellow]No script available[/yellow]")
            Prompt.ask("\nPress Enter")
            self.experiment_menu()
            return
        
        self.run_experiment_by_name(self.current_experiment["name"])
    
    def run_experiment_by_name(self, exp_name: str):
        """Run experiment by name."""
        exp = self.registry.get_experiment(exp_name)
        
        console.print(f"\n[bold]Running: {exp_name}[/bold]")
        
        if not Confirm.ask("Proceed?", default=True):
            if self.current_experiment:
                self.experiment_menu()
            else:
                self.run_experiment_menu()
            return
        
        import subprocess
        try:
            console.print("\n[dim]" + "=" * 80 + "[/dim]")
            result = subprocess.run(
                [sys.executable, str(exp["script_path"])],
                cwd=Path.cwd(),
                check=False
            )
            console.print("[dim]" + "=" * 80 + "[/dim]\n")
            
            if result.returncode == 0:
                console.print("[green]✓ Completed[/green]")
            else:
                console.print(f"[yellow]⚠ Exit code {result.returncode}[/yellow]")
            
            self.registry.refresh()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        
        Prompt.ask("\nPress Enter")
        
        if self.current_experiment:
            self.experiment_menu()
        else:
            self.run_experiment_menu()
    
    def batch_export_interactive(self):
        """Batch export all experiments."""
        self.display_header("Batch Export")
        
        format_choice = Prompt.ask("Format", choices=["csv", "parquet"], default="csv")
        output_dir = Prompt.ask("Output directory", default="exports")
        
        self.batch_export(output_dir, format_choice)
        
        Prompt.ask("\nPress Enter")
        self.main_menu()
    
    def compare_experiments_menu(self):
        """Compare experiments."""
        console.print("[yellow]Feature coming soon[/yellow]")
        Prompt.ask("\nPress Enter")
        self.main_menu()
    
    def compare_runs_interactive(self):
        """Compare runs for current experiment."""
        if not self.current_storage:
            console.print("[yellow]No data[/yellow]")
            Prompt.ask("\nPress Enter")
            self.experiment_menu()
            return
        
        self.compare_runs(self.current_experiment["name"])
        Prompt.ask("\nPress Enter")
        self.experiment_menu()
    
    def quick_stats_interactive(self):
        """Show quick stats for current experiment."""
        if not self.current_experiment["has_runs"]:
            console.print("[yellow]No data[/yellow]")
            Prompt.ask("\nPress Enter")
            self.experiment_menu()
            return
        
        # Use latest run
        self.quick_stats(self.current_experiment["name"], self.current_experiment["latest_run"])
        Prompt.ask("\nPress Enter")
        self.experiment_menu()
    
    # ========== Fast Mode (Command-line) ==========
    
    def list_experiments_cmd(self):
        """List all experiments (fast mode)."""
        self.display_experiments_table()
    
    def show_experiment_cmd(self, exp_name: str, run_id: Optional[str] = None, all_runs: bool = False):
        """Show experiment data (fast mode)."""
        exp_path = Path("experiments") / exp_name
        
        if not exp_path.exists():
            console.print(f"[red]Experiment '{exp_name}' not found[/red]")
            return
        
        try:
            # Get latest run if not specified and not using all runs
            if not run_id and not all_runs:
                runs_path = exp_path / "runs"
                runs = sorted([
                    d.name for d in runs_path.iterdir()
                    if d.is_dir() and (d / "metrics.parquet").exists()
                ], reverse=True)
                if not runs:
                    console.print("[yellow]No runs found[/yellow]")
                    return
                run_id = runs[0]
            
            storage = ExperimentStorage(exp_path, run_id=run_id)
            
            if all_runs:
                df = storage.read_all_runs()
                title = f"{exp_name} - All Runs"
            else:
                df = storage.read_metrics()
                title = f"{exp_name} - Run {storage.run_id}"
            
            console.print(Panel(title, border_style="cyan"))
            console.print(f"\n[bold]Rows:[/bold] {len(df)}")
            console.print(f"[bold]Columns:[/bold] {', '.join(df.columns)}")
            
            if "condition" in df.columns:
                console.print(f"\n[bold]By Condition:[/bold]")
                console.print(df.group_by("condition").agg(pl.count().alias("count")))
            
            if "domain" in df.columns:
                console.print(f"\n[bold]By Domain:[/bold]")
                console.print(df.group_by("domain").agg(pl.count().alias("count")))
            
            console.print(f"\n[bold]Sample (first 5):[/bold]")
            console.print(df.head(5))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def export_experiment_cmd(
        self, 
        exp_name: str, 
        output: str, 
        run_id: Optional[str] = None, 
        all_runs: bool = False,
        format: str = "csv"
    ):
        """Export experiment (fast mode)."""
        exp_path = Path("experiments") / exp_name
        
        if not exp_path.exists():
            console.print(f"[red]Experiment '{exp_name}' not found[/red]")
            return
        
        try:
            # Get latest run if not specified and not using all runs
            if not run_id and not all_runs:
                runs_path = exp_path / "runs"
                runs = sorted([
                    d.name for d in runs_path.iterdir()
                    if d.is_dir() and (d / "metrics.parquet").exists()
                ], reverse=True)
                if not runs:
                    console.print("[yellow]No runs found[/yellow]")
                    return
                run_id = runs[0]
            
            storage = ExperimentStorage(exp_path, run_id=run_id)
            
            if all_runs:
                df = storage.read_all_runs()
            else:
                df = storage.read_metrics()
            
            try:
                if format == "csv":
                    df.write_csv(output)
                elif format == "parquet":
                    df.write_parquet(output)
                
                console.print(f"[green]✓ Exported {len(df)} rows to {output}[/green]")
            except Exception as export_error:
                if "nested data" in str(export_error).lower() or "csv" in str(export_error).lower():
                    console.print(f"[yellow]⚠ CSV format doesn't support nested data structures[/yellow]")
                    console.print(f"[cyan]Tip: This experiment has complex data. Use --format parquet instead:[/cyan]")
                    parquet_output = output.replace('.csv', '.parquet')
                    console.print(f"[dim]  python run_cli.py export {exp_name} {parquet_output} --format parquet[/dim]")
                else:
                    console.print(f"[red]Error: {export_error}[/red]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def batch_export(self, output_dir: str = "exports", format: str = "csv"):
        """Batch export all experiments."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        console.print(f"[bold]Batch Export → {output_dir}/[/bold]\n")
        
        exported = 0
        for exp_name in self.registry.list_experiments():
            exp = self.registry.get_experiment(exp_name)
            if not exp["has_runs"]:
                continue
            
            try:
                storage = ExperimentStorage(exp["path"])
                df = storage.read_all_runs()
                
                filename = output_path / f"{exp_name}_all_runs.{format}"
                if format == "csv":
                    df.write_csv(filename)
                else:
                    df.write_parquet(filename)
                
                console.print(f"[green]✓ {exp_name}: {len(df)} rows → {filename.name}[/green]")
                exported += 1
            except Exception as e:
                console.print(f"[red]✗ {exp_name}: {e}[/red]")
        
        console.print(f"\n[bold]Exported {exported} experiments[/bold]")
    
    def run_experiment_cmd(self, exp_name: str):
        """Run experiment (fast mode)."""
        script_path = Path("experiments") / f"{exp_name}.py"
        
        if not script_path.exists():
            console.print(f"[red]Script not found: {script_path}[/red]")
            return
        
        console.print(f"[bold]Running: {exp_name}[/bold]\n")
        console.print("[dim]" + "=" * 80 + "[/dim]")
        
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=Path.cwd(),
                check=False
            )
            
            console.print("[dim]" + "=" * 80 + "[/dim]\n")
            
            if result.returncode == 0:
                console.print("[green]✓ Completed[/green]")
            else:
                console.print(f"[yellow]⚠ Exit code {result.returncode}[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def compare_runs(self, exp_name: str):
        """Compare runs (fast mode)."""
        exp_path = Path("experiments") / exp_name
        
        if not exp_path.exists():
            console.print(f"[red]Experiment '{exp_name}' not found[/red]")
            return
        
        try:
            storage = ExperimentStorage(exp_path)
            runs = storage.list_runs()
            
            if not runs:
                console.print("[yellow]No runs found[/yellow]")
                return
            
            console.print(Panel(f"Run Comparison: {exp_name}", border_style="cyan"))
            
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("Run ID", style="cyan")
            table.add_column("Samples", justify="right", style="green")
            table.add_column("Mean L0", justify="right", style="yellow")
            table.add_column("Mean Gini", justify="right", style="yellow")
            
            for run_id in runs:
                try:
                    df = storage.read_metrics(run_id)
                    samples = len(df)
                    
                    mean_l0 = df["l0_norm"].mean() if "l0_norm" in df.columns else None
                    mean_gini = df["gini_coefficient"].mean() if "gini_coefficient" in df.columns else None
                    
                    table.add_row(
                        run_id,
                        str(samples),
                        f"{mean_l0:.2f}" if mean_l0 else "-",
                        f"{mean_gini:.4f}" if mean_gini else "-"
                    )
                except:
                    table.add_row(run_id, "Error", "-", "-")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def quick_stats(self, exp_name: str, run_id: Optional[str] = None):
        """Quick stats (fast mode)."""
        exp_path = Path("experiments") / exp_name
        
        if not exp_path.exists():
            console.print(f"[red]Experiment '{exp_name}' not found[/red]")
            return
        
        try:
            # Get latest run if not specified
            if not run_id:
                runs_path = exp_path / "runs"
                runs = sorted([
                    d.name for d in runs_path.iterdir()
                    if d.is_dir() and (d / "metrics.parquet").exists()
                ], reverse=True)
                if not runs:
                    console.print("[yellow]No runs found[/yellow]")
                    return
                run_id = runs[0]
            
            storage = ExperimentStorage(exp_path, run_id=run_id)
            df = storage.read_metrics()
            
            console.print(Panel(f"Quick Stats: {exp_name}", border_style="cyan"))
            
            console.print(f"\n[bold]Dataset:[/bold] {len(df)} rows, {len(df.columns)} columns")
            
            if "condition" in df.columns and "l0_norm" in df.columns:
                console.print(f"\n[bold]Fact vs Hallucination:[/bold]")
                comparison = df.group_by("condition").agg([
                    pl.count().alias("count"),
                    pl.mean("l0_norm").alias("mean_l0"),
                    pl.mean("gini_coefficient").alias("mean_gini"),
                ])
                console.print(comparison)
            
            if "domain" in df.columns:
                console.print(f"\n[bold]By Domain:[/bold]")
                domain_stats = df.group_by("domain").agg([
                    pl.count().alias("count"),
                    pl.mean("l0_norm").alias("mean_l0"),
                ])
                console.print(domain_stats)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Polygraph CLI - Unified experiment management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python run_cli.py
  
  # Fast mode - List experiments
  python run_cli.py list
  
  # Show data
  python run_cli.py show 01_spectroscopy
  python run_cli.py show 01_spectroscopy --all-runs
  
  # Export
  python run_cli.py export 01_spectroscopy results.csv
  python run_cli.py export 01_spectroscopy data.parquet --format parquet --all-runs
  
  # Batch export
  python run_cli.py batch-export
  python run_cli.py batch-export --output exports --format parquet
  
  # Run experiment
  python run_cli.py run 01_spectroscopy
  
  # Compare & stats
  python run_cli.py compare 01_spectroscopy
  python run_cli.py stats 01_spectroscopy
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["list", "show", "export", "batch-export", "run", "compare", "stats"],
        help="Command to execute (omit for interactive mode)"
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name")
    parser.add_argument("output", nargs="?", help="Output filename (for export)")
    parser.add_argument("--run", help="Specific run ID")
    parser.add_argument("--all-runs", action="store_true", help="Use all runs")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Export format")
    parser.add_argument("--output-dir", default="exports", help="Output directory (for batch-export)")
    
    args = parser.parse_args()
    
    cli = NeuralPolygraphCLI()
    
    try:
        if not args.command:
            # Interactive mode
            cli.interactive_mode()
        elif args.command == "list":
            cli.list_experiments_cmd()
        elif args.command == "show":
            if not args.experiment:
                console.print("[red]Error: experiment name required[/red]")
                return
            cli.show_experiment_cmd(args.experiment, run_id=args.run, all_runs=args.all_runs)
        elif args.command == "export":
            if not args.experiment or not args.output:
                console.print("[red]Error: experiment name and output file required[/red]")
                return
            cli.export_experiment_cmd(
                args.experiment, args.output, 
                run_id=args.run, all_runs=args.all_runs, format=args.format
            )
        elif args.command == "batch-export":
            cli.batch_export(output_dir=args.output_dir, format=args.format)
        elif args.command == "run":
            if not args.experiment:
                console.print("[red]Error: experiment name required[/red]")
                return
            cli.run_experiment_cmd(args.experiment)
        elif args.command == "compare":
            if not args.experiment:
                console.print("[red]Error: experiment name required[/red]")
                return
            cli.compare_runs(args.experiment)
        elif args.command == "stats":
            if not args.experiment:
                console.print("[red]Error: experiment name required[/red]")
                return
            cli.quick_stats(args.experiment, run_id=args.run)
    
    except KeyboardInterrupt:
        console.print("\n\n[cyan]Interrupted. Goodbye![/cyan]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

