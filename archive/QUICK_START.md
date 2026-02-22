# Neural Polygraph CLI - Quick Start

## 🚀 One Command to Rule Them All

```bash
python run_cli.py
```

That's it! Everything else is optional.

---

## ⚡ 30-Second Guide

### Interactive Mode (No typing needed)
```bash
python run_cli.py
```
Navigate with numbers, explore with menus.

### Fast Mode (For power users)
```bash
# List what you have
python run_cli.py list

# See the data
python run_cli.py show 01_spectroscopy

# Export it
python run_cli.py export 01_spectroscopy results.csv

# Export everything
python run_cli.py batch-export
```

---

## 📋 Command Cheat Sheet

| What you want | Command |
|---------------|---------|
| Explore interactively | `python run_cli.py` |
| List experiments | `python run_cli.py list` |
| View data | `python run_cli.py show <experiment>` |
| Export to CSV | `python run_cli.py export <experiment> file.csv` |
| Export everything | `python run_cli.py batch-export` |
| Run experiment | `python run_cli.py run <experiment>` |
| Compare runs | `python run_cli.py compare <experiment>` |
| Quick stats | `python run_cli.py stats <experiment>` |

---

## 🎯 Common Tasks

### "I just want to see my data"
```bash
python run_cli.py show 01_spectroscopy
```

### "I need to export for analysis"
```bash
python run_cli.py export 01_spectroscopy my_data.csv
```

### "Export everything at once"
```bash
python run_cli.py batch-export
# Creates exports/ directory with all data
```

### "I'm new, show me around"
```bash
python run_cli.py
# Use interactive menus to explore
```

---

## 💡 Pro Tips

1. **Start interactive**: First time? Just run `python run_cli.py`
2. **Go fast**: Once familiar, use commands for speed
3. **Batch export**: Before analysis, run `python run_cli.py batch-export`
4. **Create alias**: Add `alias np="python run_cli.py"` to your shell config

---

## 🐛 Troubleshooting

**"No experiments found"**
- You're not in the `neural-polygraph` directory
- Run `cd neural-polygraph` first

**"Experiment not found"**
- Check spelling with `python run_cli.py list`
- Experiment names are case-sensitive

**"No data available"**
- Run the experiment first: `python run_cli.py run 01_spectroscopy`

---

## 📚 Full Documentation

See `README_CLI.md` for complete documentation.

---

**That's it! You're ready to go. 🎉**

Start with `python run_cli.py` and explore from there.

