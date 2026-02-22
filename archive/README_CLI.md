# Neural Polygraph CLI

**One unified tool for all experiment operations.**

## 🚀 Quick Start

```bash
# Interactive menu (recommended for exploration)
python run_cli.py

# Fast commands (recommended for automation)
python run_cli.py list
python run_cli.py show 01_spectroscopy
python run_cli.py export 01_spectroscopy results.csv
```

---

## 📖 Two Modes, One Tool

### Interactive Mode (Default)
Launch with no arguments for a beautiful menu-driven interface:

```bash
python run_cli.py
```

**Features:**
- ✨ Auto-discovers all experiments
- 📊 Visual tables and panels  
- 🎯 Navigate with numbers and letters
- 🔍 Browse runs and data interactively
- 📤 Guided export with prompts
- 🚀 Run experiments with confirmation
- 📈 Compare runs visually

**Perfect for:**
- First-time exploration
- When you're not sure what you need
- Visual data browsing
- Learning the system

### Fast Mode (Command-line)
Pass a command for instant execution:

```bash
python run_cli.py <command> [arguments]
```

**Features:**
- ⚡ Instant execution
- 🤖 Perfect for scripts
- 📦 Batch operations
- 🔧 No prompts or menus

**Perfect for:**
- Automation and scripts
- Quick one-off operations
- Integration with other tools
- When you know exactly what you want

---

## 📋 Commands

### List Experiments
```bash
python run_cli.py list
```
Shows all experiments with run counts, sample counts, and status.

### Show Data
```bash
# Show latest run
python run_cli.py show 01_spectroscopy

# Show specific run
python run_cli.py show 01_spectroscopy --run 20251228_210046

# Show all runs combined
python run_cli.py show 01_spectroscopy --all-runs
```
Displays row counts, breakdowns by condition/domain, and sample data.

### Export Data
```bash
# Export latest run to CSV
python run_cli.py export 01_spectroscopy results.csv

# Export to Parquet
python run_cli.py export 01_spectroscopy results.parquet --format parquet

# Export specific run
python run_cli.py export 01_spectroscopy run1.csv --run 20251228_210046

# Export all runs combined
python run_cli.py export 01_spectroscopy all_runs.csv --all-runs
```

### Batch Export
```bash
# Export all experiments to exports/ directory
python run_cli.py batch-export

# Custom output directory and format
python run_cli.py batch-export --output-dir my_exports --format parquet
```
Exports all experiments with data in one command.

### Run Experiment
```bash
python run_cli.py run 01_spectroscopy
```
Executes the experiment script and shows output.

### Compare Runs
```bash
python run_cli.py compare 01_spectroscopy
```
Shows comparison table with run IDs, sample counts, and key metrics.

### Quick Stats
```bash
# Stats for latest run
python run_cli.py stats 01_spectroscopy

# Stats for specific run
python run_cli.py stats 01_spectroscopy --run 20251228_210046
```
Shows dataset size, fact vs hallucination comparison, and domain breakdown.

---

## 🎯 Common Workflows

### Workflow 1: First Time User
```bash
# Start interactive mode
python run_cli.py

# Navigate menus:
# Main Menu → Select Experiment → 01_spectroscopy
# → View Data → Latest Run
# → Back → Export → Latest (CSV)
```

### Workflow 2: Quick Data Check
```bash
# See what's available
python run_cli.py list

# View latest results
python run_cli.py show 01_spectroscopy

# Export for analysis
python run_cli.py export 01_spectroscopy results.csv
```

### Workflow 3: Run and Analyze
```bash
# Run experiment
python run_cli.py run 01_spectroscopy

# Check stats
python run_cli.py stats 01_spectroscopy

# Export results
python run_cli.py export 01_spectroscopy results.csv
```

### Workflow 4: Batch Operations
```bash
# Export everything at once
python run_cli.py batch-export --format parquet

# Or use interactive mode for more control
python run_cli.py
# → Main Menu → Batch Export All
```

### Workflow 5: Compare Experiments
```bash
# Run experiment multiple times
python run_cli.py run 01_spectroscopy
python run_cli.py run 01_spectroscopy

# Compare results
python run_cli.py compare 01_spectroscopy
```

---

## 🔧 Advanced Usage

### Automation Script
Create `daily_export.sh`:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
python run_cli.py batch-export --output-dir "exports_${DATE}" --format parquet
echo "✓ Exported to exports_${DATE}"
```

### Shell Aliases
Add to your `.bashrc` or `.zshrc`:
```bash
alias np="python run_cli.py"
alias npl="python run_cli.py list"
alias nps="python run_cli.py show"
alias npe="python run_cli.py export"
```

Then use:
```bash
np              # Interactive mode
npl             # List experiments
nps 01_spectroscopy  # Show data
npe 01_spectroscopy results.csv  # Export
```

### Chaining Commands
```bash
# Run, compare, and export in sequence
python run_cli.py run 01_spectroscopy && \
python run_cli.py compare 01_spectroscopy && \
python run_cli.py export 01_spectroscopy latest.csv
```

### Watch Mode
Monitor experiments in real-time:
```bash
watch -n 5 'python run_cli.py list'
```

---

## 📚 Complete Command Reference

| Command | Arguments | Options | Description |
|---------|-----------|---------|-------------|
| *(none)* | - | - | Launch interactive mode |
| `list` | - | - | List all experiments |
| `show` | `<experiment>` | `--run`, `--all-runs` | Show experiment data |
| `export` | `<experiment> <output>` | `--run`, `--all-runs`, `--format` | Export data |
| `batch-export` | - | `--output-dir`, `--format` | Export all experiments |
| `run` | `<experiment>` | - | Run experiment script |
| `compare` | `<experiment>` | - | Compare runs |
| `stats` | `<experiment>` | `--run` | Quick statistics |

### Options

- `--run <run_id>` - Specify a particular run (e.g., `20251228_210046`)
- `--all-runs` - Use all runs combined instead of just latest
- `--format <fmt>` - Export format: `csv` or `parquet` (default: `csv`)
- `--output-dir <dir>` - Output directory for batch export (default: `exports`)

---

## 🐛 Troubleshooting

### "No experiments found"
- Make sure you're in the `neural-polygraph` directory
- Check that `experiments/` directory exists
- Run `python run_cli.py list` to see what's detected

### "Experiment not found"
- Check spelling: experiment names are case-sensitive
- Run `python run_cli.py list` to see available experiments
- Make sure the experiment directory exists in `experiments/`

### "No data available" / "Metrics file not found"
- Experiment hasn't been run yet
- Run it first: `python run_cli.py run 01_spectroscopy`
- Or use interactive mode to check what's available

### "CSV format does not support nested data"
- Some experiments have complex nested data structures
- Solution: Use Parquet format instead
- Interactive mode will offer to auto-convert
- Fast mode: Add `--format parquet` to your command

### "rich library not found"
- The tool will auto-install it
- Or manually: `pip install rich`

### Import errors
- Make sure you're in the `neural-polygraph` directory
- Check that `src/hallucination_detector/` exists
- Verify installation: `pip install -e .`

---

## 💡 Tips & Tricks

1. **Start Interactive**: First time? Use `python run_cli.py` to explore
2. **Learn Commands**: Once familiar, switch to fast mode for speed
3. **Tab Completion**: Use shell tab completion for experiment names
4. **Pipe Output**: Redirect output to files: `python run_cli.py show 01_spectroscopy > summary.txt`
5. **Batch Everything**: Use `batch-export` before analysis sessions
6. **Compare Often**: Run `compare` after multiple runs to track changes
7. **Stats First**: Use `stats` for quick overview before diving deeper

---

## 🎓 Examples

### Example 1: Exploring New Data
```bash
# Interactive exploration
python run_cli.py

# Navigate: Main Menu → Select Experiment → 01_spectroscopy
# → View Data → Latest Run
# Browse the data, then:
# → Back → Export → Latest (CSV)
```

### Example 2: Power User Workflow
```bash
# Quick overview
python run_cli.py list

# Check latest results
python run_cli.py stats 01_spectroscopy

# Export everything
python run_cli.py batch-export --format parquet
```

### Example 3: Research Session
```bash
# Run new experiment
python run_cli.py run 02_geometry

# Check results
python run_cli.py stats 02_geometry

# Compare with previous runs
python run_cli.py compare 02_geometry

# Export for analysis
python run_cli.py export 02_geometry geometry_results.csv --all-runs

# Open in your analysis tool (R, Python, Excel, etc.)
```

### Example 4: Automation
```bash
# Create a script: run_and_export.sh
#!/bin/bash
EXPERIMENT=$1
python run_cli.py run "$EXPERIMENT"
python run_cli.py export "$EXPERIMENT" "${EXPERIMENT}_$(date +%Y%m%d).csv"

# Use it:
./run_and_export.sh 01_spectroscopy
```

---

## 📊 Data Access & Formats

All data is stored in Parquet format for efficiency:
```
experiments/{experiment_name}/runs/{run_id}/metrics.parquet
```

### Export Formats

**CSV (Comma-Separated Values)**
- ✅ Human-readable
- ✅ Works with Excel, R, etc.
- ❌ Cannot handle nested data structures
- 📝 Use for simple experiments (01_spectroscopy, 02_geometry)

**Parquet (Columnar Binary)**
- ✅ Handles any data structure
- ✅ Faster and more efficient
- ✅ Preserves data types perfectly
- ✅ Smaller file sizes
- 📝 Use for complex experiments (03_ghost_features, etc.)
- 📝 Readable with Polars, Pandas, DuckDB, etc.

**Tip:** If CSV export fails, the CLI will suggest using Parquet automatically.

You can also access data programmatically:
```python
from pathlib import Path
from hallucination_detector.storage import ExperimentStorage

storage = ExperimentStorage(Path("experiments/01_spectroscopy"))
df = storage.read_metrics()  # Polars DataFrame

# Your analysis here
print(df.describe())
```

---

## 🚀 Getting Started Checklist

- [ ] Navigate to `neural-polygraph` directory
- [ ] Run `python run_cli.py` to launch interactive mode
- [ ] Explore the main menu
- [ ] Select an experiment and view its data
- [ ] Try exporting data to CSV
- [ ] Exit and try fast mode: `python run_cli.py list`
- [ ] Export data with: `python run_cli.py export 01_spectroscopy test.csv`
- [ ] Create a shell alias for convenience
- [ ] Start using it in your research workflow!

---

## 📖 Related Documentation

- `experiments/ACCESS_DATA.md` - Programmatic data access details
- `experiments/EXPORT_RESULTS.md` - Export options and formats
- `experiments/EXPERIMENTS_SUMMARY.md` - Experiment descriptions

---

**Happy researching! 🔬**

For questions or issues, check the troubleshooting section or examine the experiment files directly.

