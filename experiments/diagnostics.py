"""
Diagnostic Suite for Attribution Metrics
=========================================

Exploratory analysis of attribution metrics behavior across conditions.
Surfaces confounds, fragility, and failure modes.

Run: python experiments/diagnostics.py
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, bootstrap
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

# Config
DATA_PATH = Path("data/results/pint_attribution_metrics.json")
OUTPUT_BASE = Path("experiments/diagnostics/runs")


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic run."""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    data_hash: str = ""
    n_bootstrap: int = 1000
    tokenizer_proxy: str = "char_whitespace"  # chars / 5 as token estimate

    @property
    def run_dir(self) -> Path:
        return OUTPUT_BASE / self.timestamp


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load and prepare data with derived fields."""
    with open(DATA_PATH) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["samples"])

    # Compute derived fields
    df["n_chars"] = df["text"].str.len()
    df["n_tokens_est"] = (df["n_chars"] / 5).astype(int)  # rough estimate
    df["n_words"] = df["text"].str.split().str.len()

    # Normalized metrics
    df["n_active_per_char"] = df["n_active"] / df["n_chars"]
    df["n_active_per_token"] = df["n_active"] / df["n_tokens_est"]
    df["n_edges_per_char"] = df["n_edges"] / df["n_chars"]
    df["n_edges_per_token"] = df["n_edges"] / df["n_tokens_est"]

    # Binary label
    df["is_injection"] = df["label"].astype(int)

    # Data hash for reproducibility
    data_hash = hashlib.md5(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]

    return df, {"metadata": raw["metadata"], "data_hash": data_hash}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def bootstrap_ci(data: np.ndarray, statistic=np.median, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for a statistic."""
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])


class DiagnosticSuite:
    """Run all diagnostic analyses."""

    def __init__(self, df: pd.DataFrame, config: DiagnosticConfig):
        self.df = df
        self.config = config
        self.results = {}
        self.findings = []

        # Setup output
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        (self.config.run_dir / "figures").mkdir(exist_ok=True)

        # Split groups
        self.inj = df[df["is_injection"] == 1]
        self.ben = df[df["is_injection"] == 0]

        # Core metrics
        self.raw_metrics = ["n_active", "n_edges", "mean_influence",
                          "top_100_concentration", "mean_activation", "logit_entropy"]
        self.norm_metrics = ["n_active_per_char", "n_active_per_token",
                            "n_edges_per_char", "n_edges_per_token"]

    def run_all(self):
        """Execute all diagnostic analyses."""
        print(f"Running diagnostics: {self.config.timestamp}")
        print(f"Samples: {len(self.df)} (inj={len(self.inj)}, ben={len(self.ben)})")
        print("-" * 60)

        self.A_length_controls()
        self.B_stratified_slices()
        self.C_metric_relationships()
        self.D_threshold_sensitivity()
        self.E_error_surface()
        self.F_method_comparison()
        self.G_resampling_uncertainty()

        self.save_outputs()
        return self.results

    def A_length_controls(self):
        """Analyze correlation of metrics with length, compare normalized variants."""
        print("\n[A] LENGTH CONTROLS")

        results = {"correlations_raw": {}, "correlations_normalized": {},
                   "length_stats": {}, "findings": []}

        # Length stats by group
        for name, group in [("injection", self.inj), ("benign", self.ben)]:
            results["length_stats"][name] = {
                "n_chars_median": float(group["n_chars"].median()),
                "n_chars_mean": float(group["n_chars"].mean()),
                "n_chars_std": float(group["n_chars"].std()),
                "n_words_median": float(group["n_words"].median()),
            }

        # Ratio
        inj_len = self.inj["n_chars"].median()
        ben_len = self.ben["n_chars"].median()
        length_ratio = inj_len / ben_len if ben_len > 0 else np.nan
        results["length_stats"]["injection_benign_ratio"] = float(length_ratio)

        if length_ratio > 1.3 or length_ratio < 0.7:
            results["findings"].append(f"LENGTH_CONFOUND_RISK: injection/benign median length ratio = {length_ratio:.2f}")

        # Correlations with length
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, metric in enumerate(self.raw_metrics):
            r_pearson, p_pearson = pearsonr(self.df["n_chars"], self.df[metric])
            r_spearman, p_spearman = spearmanr(self.df["n_chars"], self.df[metric])

            results["correlations_raw"][metric] = {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
            }

            if abs(r_pearson) > 0.7:
                results["findings"].append(f"STRONG_LENGTH_CORRELATION: {metric} r={r_pearson:.2f}")

            # Plot
            ax = axes[i]
            ax.scatter(self.df["n_chars"], self.df[metric],
                      c=self.df["is_injection"], cmap="coolwarm", alpha=0.6, s=30)
            ax.set_xlabel("n_chars")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric}\nr={r_pearson:.2f}, p={p_pearson:.2e}")

        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "A1_length_correlations.png", dpi=150)
        plt.close()

        # Normalized metrics: compare distributions before/after
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for i, (raw, norm) in enumerate([
            ("n_active", "n_active_per_char"),
            ("n_edges", "n_edges_per_char"),
        ]):
            # Before normalization
            ax = axes[i, 0]
            ax.hist(self.ben[raw], bins=20, alpha=0.6, label="benign", density=True)
            ax.hist(self.inj[raw], bins=20, alpha=0.6, label="injection", density=True)
            d = cohens_d(self.inj[raw].values, self.ben[raw].values)
            ax.set_title(f"{raw} (raw)\nCohen's d = {d:.2f}")
            ax.legend()

            # After normalization
            ax = axes[i, 1]
            ax.hist(self.ben[norm], bins=20, alpha=0.6, label="benign", density=True)
            ax.hist(self.inj[norm], bins=20, alpha=0.6, label="injection", density=True)
            d_norm = cohens_d(self.inj[norm].values, self.ben[norm].values)
            ax.set_title(f"{norm} (normalized)\nCohen's d = {d_norm:.2f}")
            ax.legend()

            # Record
            results["correlations_normalized"][norm] = {
                "cohens_d_before": float(d),
                "cohens_d_after": float(d_norm),
                "signal_preserved": abs(d_norm) > 0.2,
            }

            if abs(d) > 0.5 and abs(d_norm) < 0.2:
                results["findings"].append(f"NORMALIZATION_KILLS_SIGNAL: {raw} d={d:.2f} -> {norm} d={d_norm:.2f}")
            elif abs(d_norm) > abs(d):
                results["findings"].append(f"NORMALIZATION_INCREASES_SIGNAL: {raw} d={d:.2f} -> {norm} d={d_norm:.2f}")

        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "A2_normalization_effect.png", dpi=150)
        plt.close()

        self.results["A_length_controls"] = results
        self.findings.extend(results["findings"])
        print(f"  Findings: {len(results['findings'])}")
        for f in results["findings"]:
            print(f"    - {f}")

    def B_stratified_slices(self):
        """Analyze metrics by category/slice."""
        print("\n[B] STRATIFIED SLICES")

        results = {"by_category": {}, "by_length_bin": {}, "findings": []}

        # By category
        for cat in self.df["category"].unique():
            subset = self.df[self.df["category"] == cat]
            cat_stats = {"n": len(subset)}

            for metric in self.raw_metrics:
                cat_stats[metric] = {
                    "median": float(subset[metric].median()),
                    "mean": float(subset[metric].mean()),
                    "std": float(subset[metric].std()),
                }

            results["by_category"][cat] = cat_stats

            if len(subset) < 10:
                results["findings"].append(f"UNDERPOWERED_SLICE: category={cat} n={len(subset)}")

        # By length bins (quartiles)
        self.df["length_bin"] = pd.qcut(self.df["n_chars"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, metric in enumerate(self.raw_metrics):
            ax = axes[i]

            # Box plot by length bin, colored by injection status
            data_by_bin = []
            labels = []
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                subset = self.df[self.df["length_bin"] == q]
                data_by_bin.append(subset[metric].values)
                labels.append(f"{q}\n(n={len(subset)})")

            bp = ax.boxplot(data_by_bin, labels=labels, patch_artist=True)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by length quartile")

            # Record
            results["by_length_bin"][metric] = {}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                subset = self.df[self.df["length_bin"] == q]
                inj_rate = subset["is_injection"].mean()
                results["by_length_bin"][metric][q] = {
                    "median": float(subset[metric].median()),
                    "injection_rate": float(inj_rate),
                    "n": len(subset),
                }

        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "B1_length_quartile_distributions.png", dpi=150)
        plt.close()

        # Check injection rate by length quartile
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            subset = self.df[self.df["length_bin"] == q]
            inj_rate = subset["is_injection"].mean()
            if inj_rate > 0.3 or inj_rate < 0.05:
                results["findings"].append(f"UNBALANCED_QUARTILE: {q} injection_rate={inj_rate:.2%}")

        self.results["B_stratified_slices"] = results
        self.findings.extend(results["findings"])
        print(f"  Findings: {len(results['findings'])}")
        for f in results["findings"]:
            print(f"    - {f}")

    def C_metric_relationships(self):
        """Correlation matrix, partial correlations, redundancy analysis."""
        print("\n[C] METRIC RELATIONSHIPS")

        results = {"correlation_matrix": {}, "partial_correlations": {},
                   "redundant_pairs": [], "findings": []}

        # Correlation matrix
        corr_df = self.df[self.raw_metrics + ["n_chars", "is_injection"]].corr()
        results["correlation_matrix"] = corr_df.to_dict()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_df.columns)

        # Add correlation values
        for i in range(len(corr_df)):
            for j in range(len(corr_df)):
                ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.title("Metric Correlation Matrix")
        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "C1_correlation_matrix.png", dpi=150)
        plt.close()

        # Find redundant pairs (r > 0.9)
        for i, m1 in enumerate(self.raw_metrics):
            for m2 in self.raw_metrics[i+1:]:
                r = corr_df.loc[m1, m2]
                if abs(r) > 0.9:
                    results["redundant_pairs"].append({"m1": m1, "m2": m2, "r": float(r)})
                    results["findings"].append(f"REDUNDANT_METRICS: {m1} <-> {m2} r={r:.2f}")

        # Partial correlations (control for length)
        # Using simple residualization
        for metric in self.raw_metrics:
            # Residualize metric on n_chars
            slope, intercept = np.polyfit(self.df["n_chars"], self.df[metric], 1)
            residuals = self.df[metric] - (slope * self.df["n_chars"] + intercept)

            # Correlation of residuals with is_injection
            r_partial, p_partial = pearsonr(residuals, self.df["is_injection"])

            # Compare to raw
            r_raw, p_raw = pearsonr(self.df[metric], self.df["is_injection"])

            results["partial_correlations"][metric] = {
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_partial_length_controlled": float(r_partial),
                "p_partial": float(p_partial),
                "signal_change": "preserved" if abs(r_partial) > 0.1 else "lost",
            }

            if abs(r_raw) > 0.2 and abs(r_partial) < 0.1:
                results["findings"].append(f"LENGTH_CONFOUNDED: {metric} raw_r={r_raw:.2f} -> partial_r={r_partial:.2f}")

        self.results["C_metric_relationships"] = results
        self.findings.extend(results["findings"])
        print(f"  Findings: {len(results['findings'])}")
        for f in results["findings"]:
            print(f"    - {f}")

    def D_threshold_sensitivity(self):
        """Sweep thresholds, track decision stability."""
        print("\n[D] THRESHOLD SENSITIVITY")

        results = {"roc_curves": {}, "flip_analysis": {}, "findings": []}

        # For each metric, compute ROC
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        best_auc = 0
        best_metric = None

        for i, metric in enumerate(self.raw_metrics):
            ax = axes[i]

            # ROC curve (higher values = more injection-like for some metrics, inverse for others)
            # Try both directions
            y_true = self.df["is_injection"].values
            y_score = self.df[metric].values

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # Also try inverted
            fpr_inv, tpr_inv, _ = roc_curve(y_true, -y_score)
            roc_auc_inv = auc(fpr_inv, tpr_inv)

            if roc_auc_inv > roc_auc:
                fpr, tpr, roc_auc = fpr_inv, tpr_inv, roc_auc_inv
                direction = "lower_is_injection"
            else:
                direction = "higher_is_injection"

            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{metric}")
            ax.legend()

            results["roc_curves"][metric] = {
                "auc": float(roc_auc),
                "direction": direction,
            }

            if roc_auc > best_auc:
                best_auc = roc_auc
                best_metric = metric

            # Assess sensitivity: how many samples flip between 25th and 75th percentile threshold?
            p25, p75 = np.percentile(y_score, [25, 75])
            pred_p25 = (y_score > p25).astype(int) if direction == "higher_is_injection" else (y_score < p25).astype(int)
            pred_p75 = (y_score > p75).astype(int) if direction == "higher_is_injection" else (y_score < p75).astype(int)
            flips = np.sum(pred_p25 != pred_p75)
            flip_rate = flips / len(y_score)

            results["flip_analysis"][metric] = {
                "flips_between_q25_q75": int(flips),
                "flip_rate": float(flip_rate),
            }

            if flip_rate > 0.4:
                results["findings"].append(f"HIGH_THRESHOLD_SENSITIVITY: {metric} flip_rate={flip_rate:.1%}")

        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "D1_roc_curves.png", dpi=150)
        plt.close()

        results["best_single_metric"] = {"metric": best_metric, "auc": float(best_auc)}

        if best_auc < 0.7:
            results["findings"].append(f"WEAK_DISCRIMINATION: best AUC = {best_auc:.3f}")

        self.results["D_threshold_sensitivity"] = results
        self.findings.extend(results["findings"])
        print(f"  Best metric: {best_metric} (AUC={best_auc:.3f})")
        print(f"  Findings: {len(results['findings'])}")
        for f in results["findings"]:
            print(f"    - {f}")

    def E_error_surface(self):
        """Identify borderline cases and misclassifications."""
        print("\n[E] ERROR SURFACE")

        results = {"borderline_cases": [], "fp_cases": [], "fn_cases": [], "findings": []}

        # Use best metric from D for classification
        best_metric = self.results.get("D_threshold_sensitivity", {}).get("best_single_metric", {}).get("metric", "top_100_concentration")
        direction = self.results.get("D_threshold_sensitivity", {}).get("roc_curves", {}).get(best_metric, {}).get("direction", "lower_is_injection")

        # Find optimal threshold (Youden's J)
        y_true = self.df["is_injection"].values
        y_score = self.df[best_metric].values
        if direction == "lower_is_injection":
            y_score = -y_score

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]

        # Classify
        predictions = (y_score >= best_threshold).astype(int)

        # Confidence: distance from threshold normalized
        score_range = np.max(y_score) - np.min(y_score)
        confidence = np.abs(y_score - best_threshold) / score_range

        self.df["predicted"] = predictions
        self.df["confidence"] = confidence

        # Find errors
        fp_mask = (predictions == 1) & (y_true == 0)
        fn_mask = (predictions == 0) & (y_true == 1)

        fp_df = self.df[fp_mask].nsmallest(10, "confidence")
        fn_df = self.df[fn_mask].nsmallest(10, "confidence")

        for _, row in fp_df.iterrows():
            case = {
                "idx": int(row["idx"]),
                "text_preview": row["text"][:80] + "..." if len(row["text"]) > 80 else row["text"],
                "true_label": "benign",
                "predicted": "injection",
                "confidence": float(row["confidence"]),
                best_metric: float(row[best_metric]),
                "n_chars": int(row["n_chars"]),
                "notes": self._case_notes(row, "FP"),
            }
            results["fp_cases"].append(case)

        for _, row in fn_df.iterrows():
            case = {
                "idx": int(row["idx"]),
                "text_preview": row["text"][:80] + "..." if len(row["text"]) > 80 else row["text"],
                "true_label": "injection",
                "predicted": "benign",
                "confidence": float(row["confidence"]),
                best_metric: float(row[best_metric]),
                "n_chars": int(row["n_chars"]),
                "notes": self._case_notes(row, "FN"),
            }
            results["fn_cases"].append(case)

        # Borderline: lowest confidence regardless of correctness
        borderline_df = self.df.nsmallest(15, "confidence")
        for _, row in borderline_df.iterrows():
            case = {
                "idx": int(row["idx"]),
                "text_preview": row["text"][:60] + "..." if len(row["text"]) > 60 else row["text"],
                "true_label": "injection" if row["is_injection"] else "benign",
                "confidence": float(row["confidence"]),
            }
            results["borderline_cases"].append(case)

        # Summary stats
        n_fp = fp_mask.sum()
        n_fn = fn_mask.sum()
        accuracy = (predictions == y_true).mean()
        results["summary"] = {
            "classifier_metric": best_metric,
            "threshold": float(best_threshold) if direction == "higher_is_injection" else float(-best_threshold),
            "n_fp": int(n_fp),
            "n_fn": int(n_fn),
            "accuracy": float(accuracy),
            "direction": direction,
        }

        if n_fn > 5:
            results["findings"].append(f"MANY_FALSE_NEGATIVES: {n_fn} injections missed")
        if n_fp > 10:
            results["findings"].append(f"MANY_FALSE_POSITIVES: {n_fp} benign flagged")

        # Visualize error surface
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter all points
        colors = ["green" if p == t else "red" for p, t in zip(predictions, y_true)]
        markers = ["o" if t == 1 else "s" for t in y_true]

        for m, label in [("o", "injection"), ("s", "benign")]:
            mask = [markers[i] == m for i in range(len(markers))]
            ax.scatter(
                self.df.loc[mask, "n_chars"],
                self.df.loc[mask, best_metric],
                c=[colors[i] for i in range(len(colors)) if mask[i]],
                marker=m,
                alpha=0.6,
                s=50,
                label=f"true={label}",
            )

        ax.axhline(y=best_threshold if direction == "higher_is_injection" else -best_threshold,
                   color="black", linestyle="--", label="threshold")
        ax.set_xlabel("n_chars (length)")
        ax.set_ylabel(best_metric)
        ax.set_title(f"Error Surface: green=correct, red=error\n{n_fp} FP, {n_fn} FN, acc={accuracy:.1%}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "E1_error_surface.png", dpi=150)
        plt.close()

        self.results["E_error_surface"] = results
        self.findings.extend(results["findings"])
        print(f"  FP={n_fp}, FN={n_fn}, Accuracy={accuracy:.1%}")
        print(f"  Findings: {len(results['findings'])}")
        for f in results["findings"]:
            print(f"    - {f}")

    def _case_notes(self, row, error_type: str) -> str:
        """Generate case notes for an error."""
        notes = []

        if error_type == "FP":
            # Why might this benign be flagged?
            if row["n_chars"] > 150:
                notes.append("long_prompt")
            if "?" in row["text"]:
                notes.append("contains_question")
            if any(kw in row["text"].lower() for kw in ["forget", "ignore", "new task"]):
                notes.append("contains_injection_keywords_but_benign")
        else:
            # Why might this injection be missed?
            if row["n_chars"] < 80:
                notes.append("short_injection")
            if row["top_100_concentration"] > 0.005:
                notes.append("high_concentration_unusual")

        return "; ".join(notes) if notes else "no_obvious_pattern"

    def F_method_comparison(self):
        """Compare different metrics as classifiers."""
        print("\n[F] METHOD COMPARISON")

        results = {"metric_agreement": {}, "best_combinations": [], "findings": []}

        # Get predictions from each metric at optimal threshold
        y_true = self.df["is_injection"].values
        predictions = {}

        for metric in self.raw_metrics:
            y_score = self.df[metric].values

            # Find optimal direction and threshold
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_pos = auc(fpr, tpr)

            fpr_neg, tpr_neg, thresholds_neg = roc_curve(y_true, -y_score)
            auc_neg = auc(fpr_neg, tpr_neg)

            if auc_neg > auc_pos:
                y_score = -y_score
                fpr, tpr, thresholds = fpr_neg, tpr_neg, thresholds_neg

            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]

            predictions[metric] = (y_score >= best_threshold).astype(int)

        # Agreement matrix
        n_metrics = len(self.raw_metrics)
        agreement_matrix = np.zeros((n_metrics, n_metrics))

        for i, m1 in enumerate(self.raw_metrics):
            for j, m2 in enumerate(self.raw_metrics):
                agreement = (predictions[m1] == predictions[m2]).mean()
                agreement_matrix[i, j] = agreement

        results["metric_agreement"] = {
            f"{m1}_vs_{m2}": float(agreement_matrix[i, j])
            for i, m1 in enumerate(self.raw_metrics)
            for j, m2 in enumerate(self.raw_metrics)
            if i < j
        }

        # Find pairs with low agreement (different information)
        for i, m1 in enumerate(self.raw_metrics):
            for j, m2 in enumerate(self.raw_metrics[i+1:], i+1):
                agreement = agreement_matrix[i, j]
                if agreement < 0.7:
                    results["findings"].append(f"LOW_AGREEMENT: {m1} vs {m2} agree={agreement:.1%}")

        # Plot agreement matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(agreement_matrix, cmap="YlGn", vmin=0.5, vmax=1)

        ax.set_xticks(range(n_metrics))
        ax.set_yticks(range(n_metrics))
        ax.set_xticklabels(self.raw_metrics, rotation=45, ha="right")
        ax.set_yticklabels(self.raw_metrics)

        for i in range(n_metrics):
            for j in range(n_metrics):
                ax.text(j, i, f"{agreement_matrix[i, j]:.0%}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax, label="Agreement Rate")
        plt.title("Metric Prediction Agreement")
        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "F1_metric_agreement.png", dpi=150)
        plt.close()

        # Ensemble: vote across metrics
        vote_sum = np.sum([predictions[m] for m in self.raw_metrics], axis=0)
        majority_vote = (vote_sum >= len(self.raw_metrics) / 2).astype(int)
        ensemble_acc = (majority_vote == y_true).mean()

        results["ensemble"] = {
            "majority_vote_accuracy": float(ensemble_acc),
            "improvement_over_best": float(ensemble_acc - self.results["D_threshold_sensitivity"]["best_single_metric"]["auc"]),
        }

        self.results["F_method_comparison"] = results
        self.findings.extend(results["findings"])
        print(f"  Ensemble accuracy: {ensemble_acc:.1%}")
        print(f"  Findings: {len(results['findings'])}")

    def G_resampling_uncertainty(self):
        """Bootstrap confidence intervals for key statistics."""
        print("\n[G] RESAMPLING UNCERTAINTY")

        results = {"ci_effect_sizes": {}, "ci_medians": {}, "findings": []}

        # Bootstrap effect sizes
        np.random.seed(42)

        for metric in self.raw_metrics:
            inj_vals = self.inj[metric].values
            ben_vals = self.ben[metric].values

            # Point estimate
            d_point = cohens_d(inj_vals, ben_vals)

            # Bootstrap CI
            boot_ds = []
            for _ in range(self.config.n_bootstrap):
                inj_sample = np.random.choice(inj_vals, size=len(inj_vals), replace=True)
                ben_sample = np.random.choice(ben_vals, size=len(ben_vals), replace=True)
                boot_ds.append(cohens_d(inj_sample, ben_sample))

            ci_low, ci_high = np.percentile(boot_ds, [2.5, 97.5])

            results["ci_effect_sizes"][metric] = {
                "cohens_d": float(d_point),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "ci_width": float(ci_high - ci_low),
                "significant": not (ci_low < 0 < ci_high),
            }

            if ci_high - ci_low > 1.0:
                results["findings"].append(f"WIDE_CI: {metric} d={d_point:.2f} CI=[{ci_low:.2f}, {ci_high:.2f}]")

            if ci_low < 0 < ci_high:
                results["findings"].append(f"NON_SIGNIFICANT: {metric} CI crosses zero")

        # Bootstrap median differences
        for metric in self.raw_metrics:
            inj_vals = self.inj[metric].values
            ben_vals = self.ben[metric].values

            diff_point = np.median(inj_vals) - np.median(ben_vals)

            boot_diffs = []
            for _ in range(self.config.n_bootstrap):
                inj_sample = np.random.choice(inj_vals, size=len(inj_vals), replace=True)
                ben_sample = np.random.choice(ben_vals, size=len(ben_vals), replace=True)
                boot_diffs.append(np.median(inj_sample) - np.median(ben_sample))

            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

            results["ci_medians"][metric] = {
                "median_diff": float(diff_point),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
            }

        # Visualize effect sizes with CIs
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_sorted = sorted(
            self.raw_metrics,
            key=lambda m: abs(results["ci_effect_sizes"][m]["cohens_d"]),
            reverse=True
        )

        y_pos = range(len(metrics_sorted))
        ds = [results["ci_effect_sizes"][m]["cohens_d"] for m in metrics_sorted]
        ci_lows = [results["ci_effect_sizes"][m]["ci_95_low"] for m in metrics_sorted]
        ci_highs = [results["ci_effect_sizes"][m]["ci_95_high"] for m in metrics_sorted]

        ax.barh(y_pos, ds, xerr=[np.array(ds) - np.array(ci_lows), np.array(ci_highs) - np.array(ds)],
                capsize=5, alpha=0.7)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0.2, color="gray", linestyle="--", linewidth=0.5, label="small effect")
        ax.axvline(x=-0.2, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=0.5, label="medium effect")
        ax.axvline(x=-0.5, color="gray", linestyle=":", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_sorted)
        ax.set_xlabel("Cohen's d (injection - benign)")
        ax.set_title("Effect Sizes with 95% Bootstrap CI\n(n_inj=21, n_ben=115)")
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(self.config.run_dir / "figures" / "G1_effect_sizes_ci.png", dpi=150)
        plt.close()

        self.results["G_resampling_uncertainty"] = results
        self.findings.extend(results["findings"])
        print(f"  Findings: {len(results['findings'])}")
        for f in results["findings"][:5]:  # Limit output
            print(f"    - {f}")
        if len(results["findings"]) > 5:
            print(f"    ... and {len(results['findings']) - 5} more")

    def save_outputs(self):
        """Save all outputs."""
        print("\n" + "=" * 60)
        print("SAVING OUTPUTS")

        # Manifest
        manifest = {
            "timestamp": self.config.timestamp,
            "data_hash": self.config.data_hash,
            "n_samples": len(self.df),
            "n_injection": len(self.inj),
            "n_benign": len(self.ben),
            "tokenizer_proxy": self.config.tokenizer_proxy,
            "n_bootstrap": self.config.n_bootstrap,
            "all_findings": self.findings,
        }

        with open(self.config.run_dir / "diagnostics_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Full results
        with open(self.config.run_dir / "diagnostics_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Report
        self._write_report()

        # Next experiments
        self._write_next_experiments()

        print(f"Outputs saved to: {self.config.run_dir}")

    def _write_report(self):
        """Generate markdown diagnostic report."""
        report = []
        report.append("# Attribution Metrics Diagnostic Report")
        report.append(f"\n**Run:** {self.config.timestamp}")
        report.append(f"**Data hash:** {self.config.data_hash}")
        report.append(f"**Samples:** {len(self.df)} (injection={len(self.inj)}, benign={len(self.ben)})")
        report.append("")
        report.append("---")
        report.append("")

        # Summary of findings
        report.append("## Summary of Findings")
        report.append("")

        # Categorize findings
        confounds = [f for f in self.findings if "CONFOUND" in f or "LENGTH" in f or "CORRELATION" in f]
        sensitivity = [f for f in self.findings if "SENSITIVITY" in f or "FLIP" in f]
        power = [f for f in self.findings if "UNDERPOWERED" in f or "CI" in f or "SIGNIFICANT" in f]
        errors = [f for f in self.findings if "FALSE" in f or "ERROR" in f]

        if confounds:
            report.append("### Confounds Detected")
            for f in confounds:
                report.append(f"- {f}")
            report.append("")

        if sensitivity:
            report.append("### Threshold Sensitivity")
            for f in sensitivity:
                report.append(f"- {f}")
            report.append("")

        if power:
            report.append("### Statistical Power Concerns")
            for f in power:
                report.append(f"- {f}")
            report.append("")

        if errors:
            report.append("### Classification Errors")
            for f in errors:
                report.append(f"- {f}")
            report.append("")

        # Section details
        report.append("---")
        report.append("")
        report.append("## Detailed Results")
        report.append("")

        # A: Length Controls
        report.append("### A. Length Controls")
        report.append("")
        if "A_length_controls" in self.results:
            ls = self.results["A_length_controls"]["length_stats"]
            report.append(f"- Injection median length: {ls['injection']['n_chars_median']:.0f} chars")
            report.append(f"- Benign median length: {ls['benign']['n_chars_median']:.0f} chars")
            report.append(f"- Ratio: {ls['injection_benign_ratio']:.2f}x")
            report.append("")
            report.append("**Length correlations (raw metrics):**")
            report.append("")
            report.append("| Metric | Pearson r | Interpretation |")
            report.append("|--------|-----------|----------------|")
            for m, v in self.results["A_length_controls"]["correlations_raw"].items():
                r = v["pearson_r"]
                interp = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                report.append(f"| {m} | {r:.3f} | {interp} |")
            report.append("")

        # B: Stratified Slices
        report.append("### B. Stratified Slices")
        report.append("")
        if "B_stratified_slices" in self.results:
            report.append("**By category:**")
            report.append("")
            for cat, stats in self.results["B_stratified_slices"]["by_category"].items():
                report.append(f"- **{cat}**: n={stats['n']}")
            report.append("")

        # C: Metric Relationships
        report.append("### C. Metric Relationships")
        report.append("")
        if "C_metric_relationships" in self.results:
            if self.results["C_metric_relationships"]["redundant_pairs"]:
                report.append("**Redundant metric pairs (r > 0.9):**")
                for pair in self.results["C_metric_relationships"]["redundant_pairs"]:
                    report.append(f"- {pair['m1']} <-> {pair['m2']} (r={pair['r']:.2f})")
                report.append("")

            report.append("**Partial correlations (controlling for length):**")
            report.append("")
            report.append("| Metric | Raw r | Partial r | Signal |")
            report.append("|--------|-------|-----------|--------|")
            for m, v in self.results["C_metric_relationships"]["partial_correlations"].items():
                report.append(f"| {m} | {v['r_raw']:.3f} | {v['r_partial_length_controlled']:.3f} | {v['signal_change']} |")
            report.append("")

        # D: Threshold Sensitivity
        report.append("### D. Threshold Sensitivity")
        report.append("")
        if "D_threshold_sensitivity" in self.results:
            best = self.results["D_threshold_sensitivity"]["best_single_metric"]
            report.append(f"**Best single metric:** {best['metric']} (AUC={best['auc']:.3f})")
            report.append("")
            report.append("| Metric | AUC | Direction |")
            report.append("|--------|-----|-----------|")
            for m, v in self.results["D_threshold_sensitivity"]["roc_curves"].items():
                report.append(f"| {m} | {v['auc']:.3f} | {v['direction']} |")
            report.append("")

        # E: Error Surface
        report.append("### E. Error Surface")
        report.append("")
        if "E_error_surface" in self.results:
            summary = self.results["E_error_surface"]["summary"]
            report.append(f"**Classifier:** {summary['classifier_metric']}")
            report.append(f"**Accuracy:** {summary['accuracy']:.1%}")
            report.append(f"**False Positives:** {summary['n_fp']}")
            report.append(f"**False Negatives:** {summary['n_fn']}")
            report.append("")

            if self.results["E_error_surface"]["fn_cases"]:
                report.append("**Sample False Negatives (missed injections):**")
                report.append("")
                for case in self.results["E_error_surface"]["fn_cases"][:3]:
                    report.append(f"- idx={case['idx']}: \"{case['text_preview']}\"")
                    report.append(f"  - Notes: {case['notes']}")
                report.append("")

        # G: Uncertainty
        report.append("### G. Resampling Uncertainty")
        report.append("")
        if "G_resampling_uncertainty" in self.results:
            report.append("**Effect sizes with 95% CI:**")
            report.append("")
            report.append("| Metric | Cohen's d | 95% CI | Significant |")
            report.append("|--------|-----------|--------|-------------|")
            for m, v in self.results["G_resampling_uncertainty"]["ci_effect_sizes"].items():
                sig = "yes" if v["significant"] else "no"
                report.append(f"| {m} | {v['cohens_d']:.2f} | [{v['ci_95_low']:.2f}, {v['ci_95_high']:.2f}] | {sig} |")
            report.append("")

        # Figures
        report.append("---")
        report.append("")
        report.append("## Figures")
        report.append("")
        for fig_path in sorted((self.config.run_dir / "figures").glob("*.png")):
            report.append(f"- `{fig_path.name}`")

        with open(self.config.run_dir / "diagnostics_report.md", "w") as f:
            f.write("\n".join(report))

    def _write_next_experiments(self):
        """Generate next experiments recommendations."""
        experiments = []
        experiments.append("# Next Experiments")
        experiments.append("")
        experiments.append("Ranked recommendations based on diagnostic findings.")
        experiments.append("")

        # Based on findings, recommend experiments
        recs = []

        # 1. If length confound detected
        if any("LENGTH" in f or "CONFOUND" in f for f in self.findings):
            recs.append({
                "priority": 1,
                "title": "Length-Matched Comparison",
                "rationale": "Strong length correlation detected. Need to compare injection vs benign at matched lengths.",
                "method": "Subsample both classes to have similar length distributions, then re-run analysis.",
                "expected_outcome": "If signal disappears, length was the confound. If signal persists, there's a real effect.",
            })

        # 2. If small sample size
        if any("UNDERPOWERED" in f or "n=21" in str(f) for f in self.findings):
            recs.append({
                "priority": 1,
                "title": "Expand Injection Sample",
                "rationale": "Only 21 injection samples. Wide CIs make conclusions unreliable.",
                "method": "Run attribution on additional PINT samples (target n=100+ injections).",
                "expected_outcome": "Tighter confidence intervals, more reliable effect size estimates.",
            })

        # 3. If metrics are redundant
        if any("REDUNDANT" in f for f in self.findings):
            recs.append({
                "priority": 2,
                "title": "Dimensionality Reduction",
                "rationale": "Multiple metrics appear redundant (r > 0.9).",
                "method": "PCA or factor analysis to identify independent dimensions.",
                "expected_outcome": "Simpler feature set, clearer interpretation.",
            })

        # 4. Token-level analysis
        recs.append({
            "priority": 2,
            "title": "Token-Level Attribution",
            "rationale": "Current analysis is prompt-level. Token-level patterns may reveal more.",
            "method": "Compute per-token activation metrics, look for position effects (early vs late tokens).",
            "expected_outcome": "Understanding of where in the prompt the signal originates.",
        })

        # 5. Cross-model validation
        recs.append({
            "priority": 3,
            "title": "Cross-Model Validation",
            "rationale": "Current findings are specific to Gemma-2-2b.",
            "method": "Run same analysis on different models (Llama, Mistral) with their SAEs.",
            "expected_outcome": "If patterns hold, they're general. If not, model-specific.",
        })

        # 6. Synthetic injection generation
        recs.append({
            "priority": 3,
            "title": "Synthetic Injection Variants",
            "rationale": "Natural dataset may have confounds beyond length (style, vocabulary).",
            "method": "Generate synthetic injections by adding 'ignore previous' prefixes to benign prompts.",
            "expected_outcome": "Controlled comparison isolating injection-specific features.",
        })

        # Write recommendations
        for rec in sorted(recs, key=lambda x: x["priority"]):
            experiments.append(f"## P{rec['priority']}: {rec['title']}")
            experiments.append("")
            experiments.append(f"**Rationale:** {rec['rationale']}")
            experiments.append("")
            experiments.append(f"**Method:** {rec['method']}")
            experiments.append("")
            experiments.append(f"**Expected Outcome:** {rec['expected_outcome']}")
            experiments.append("")

        with open(self.config.run_dir / "next_experiments.md", "w") as f:
            f.write("\n".join(experiments))


def main():
    """Run diagnostic suite."""
    df, meta = load_data()

    config = DiagnosticConfig()
    config.data_hash = meta["data_hash"]

    suite = DiagnosticSuite(df, config)
    suite.run_all()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RUN COMPLETE")
    print(f"Output: {config.run_dir}")
    print(f"Total findings: {len(suite.findings)}")


if __name__ == "__main__":
    main()
