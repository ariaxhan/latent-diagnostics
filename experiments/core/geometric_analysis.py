"""
Geometric Analysis of Attribution Metric Space
===============================================

Applies inertia tensor methodology (adapted from AIDA-TNG galaxy morphology)
to analyze the geometric structure of domain clusters in attribution metric space.

Instead of SAE features as masses in d_model space, we treat:
- Samples as points in metric space (6D)
- Domain clusters as "galaxies" with shape properties
- Compute axis ratios, effective dimensionality, misalignment

This reveals whether domains have different *shapes* (not just different locations)
in attribution space - a novel axis for characterizing task types.

Usage:
  python experiments/core/geometric_analysis.py --analyze
  python experiments/core/geometric_analysis.py --analyze --results-path data/results/domain_attribution_metrics.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# GEOMETRIC METRICS (Adapted from src/neural_polygraph/geometry.py)
# ============================================================

@dataclass
class DomainGeometry:
    """Geometric properties of a domain cluster in metric space."""

    domain: str
    n_samples: int

    # Principal Axis Ratios (3D shape proxy)
    c_over_a: float      # Minor/Major (Sphericity)
    b_over_a: float      # Intermediate/Major (Elongation)
    shape_class: Literal["Spherical", "Oblate", "Prolate", "Triaxial"]

    # Global Topology
    eigenvalue_entropy: float  # Dispersion of variance across dimensions
    effective_dimensionality: float  # Participation ratio

    # Alignment
    misalignment_angle: float  # Angle between centroid and major axis (degrees)

    # Centroid (mean position in metric space)
    centroid: List[float]

    # Raw eigendata
    eigenvalues: List[float]  # Top 6 eigenvalues
    explained_variance_ratio: List[float]


def compute_domain_geometry(
    points: np.ndarray,
    domain: str,
    metric_names: List[str]
) -> DomainGeometry:
    """
    Compute geometric properties of a point cloud in metric space.

    Adapts the inertia tensor methodology from astrophysics:
    - Points are treated as unit masses at positions in metric space
    - Covariance matrix = inertia tensor (in equal-mass limit)
    - Eigendecomposition reveals principal axes and shape

    Args:
        points: (N_samples, N_metrics) array of metric values
        domain: Name of the domain
        metric_names: Names of metrics for reporting

    Returns:
        DomainGeometry dataclass with shape properties
    """
    n_samples, n_dims = points.shape

    if n_samples < 3:
        return _empty_geometry(domain, n_samples, n_dims)

    # 1. COMPUTE CENTROID (mean position)
    centroid = np.mean(points, axis=0)

    # 2. CENTER THE CLOUD
    centered = points - centroid

    # 3. COMPUTE COVARIANCE / INERTIA TENSOR
    # For equal masses, this is proportional to the inertia tensor
    cov_matrix = np.cov(centered.T)

    # Handle 1D case
    if n_dims == 1:
        cov_matrix = np.array([[cov_matrix]])

    # 4. EIGENDECOMPOSITION
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort descending (largest variance first)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Ensure non-negative
    eigvals = np.maximum(eigvals, 0)

    # 5. COMPUTE SHAPE METRICS
    # Use top 3 eigenvalues to define ellipsoid shape
    L1 = max(eigvals[0], 1e-12)
    L2 = eigvals[1] if len(eigvals) > 1 else 0
    L3 = eigvals[2] if len(eigvals) > 2 else 0

    # Axis lengths = sqrt of eigenvalues (variance -> std dev)
    a = np.sqrt(L1)
    b = np.sqrt(max(L2, 0))
    c = np.sqrt(max(L3, 0))

    c_over_a = c / a if a > 0 else 0
    b_over_a = b / a if a > 0 else 0

    shape_class = _classify_shape(c_over_a, b_over_a)

    # 6. EFFECTIVE DIMENSIONALITY (Participation Ratio)
    # PR = (sum(lambda))^2 / sum(lambda^2)
    # High PR = variance spread across many dimensions
    # Low PR = variance concentrated in few dimensions
    total_var = np.sum(eigvals)
    sum_sq = np.sum(eigvals ** 2)
    eff_dim = (total_var ** 2) / (sum_sq + 1e-12)

    # 7. EIGENVALUE ENTROPY
    entropy = _compute_entropy(eigvals)

    # 8. MISALIGNMENT
    # Angle between centroid vector and major axis
    major_axis = eigvecs[:, 0]
    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm > 1e-9:
        cos_sim = np.abs(np.dot(centroid, major_axis)) / (centroid_norm * np.linalg.norm(major_axis) + 1e-9)
        cos_sim = np.clip(cos_sim, 0, 1)
        misalignment = np.degrees(np.arccos(cos_sim))
    else:
        misalignment = 0.0

    # Explained variance ratio
    total = np.sum(eigvals)
    explained = eigvals / total if total > 0 else eigvals

    return DomainGeometry(
        domain=domain,
        n_samples=n_samples,
        c_over_a=float(c_over_a),
        b_over_a=float(b_over_a),
        shape_class=shape_class,
        eigenvalue_entropy=float(entropy),
        effective_dimensionality=float(eff_dim),
        misalignment_angle=float(misalignment),
        centroid=centroid.tolist(),
        eigenvalues=eigvals[:6].tolist(),
        explained_variance_ratio=explained[:6].tolist()
    )


def _classify_shape(c_a: float, b_a: float) -> str:
    """
    Classify shape based on axis ratios.

    From AIDA-TNG / planetary science:
    - Spherical: b/a > 0.9 and c/a > 0.9 (all axes similar)
    - Oblate: b/a > 0.8 and c/a < 0.6 (disk/pancake)
    - Prolate: b/a < 0.6 and c/a < 0.6 (cigar/elongated)
    - Triaxial: everything else (irregular)
    """
    if b_a > 0.9 and c_a > 0.9:
        return "Spherical"
    elif b_a > 0.8 and c_a < 0.6:
        return "Oblate"
    elif b_a < 0.6 and c_a < 0.6:
        return "Prolate"
    else:
        return "Triaxial"


def _compute_entropy(eigenvalues: np.ndarray) -> float:
    """Normalized Shannon entropy of eigenvalue spectrum."""
    total = np.sum(eigenvalues)
    if total == 0:
        return 0.0
    probs = eigenvalues / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _empty_geometry(domain: str, n: int, dims: int) -> DomainGeometry:
    return DomainGeometry(
        domain=domain, n_samples=n,
        c_over_a=0.0, b_over_a=0.0, shape_class="Spherical",
        eigenvalue_entropy=0.0, effective_dimensionality=0.0,
        misalignment_angle=0.0, centroid=[0.0] * dims,
        eigenvalues=[0.0] * min(dims, 6),
        explained_variance_ratio=[0.0] * min(dims, 6)
    )


# ============================================================
# CROSS-DOMAIN GEOMETRY
# ============================================================

def compute_inter_domain_angles(
    geometries: Dict[str, DomainGeometry],
    df: pd.DataFrame,
    metrics: List[str]
) -> Dict[str, any]:
    """
    Compute geometric relationships between domain clusters.

    - Centroid distances (how far apart are domains?)
    - Principal axis angles (are clusters oriented the same way?)
    - Overlap metrics (do clusters interpenetrate?)
    """
    domains = list(geometries.keys())
    n = len(domains)

    # Centroid distance matrix
    centroid_distances = np.zeros((n, n))
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            c1 = np.array(geometries[d1].centroid)
            c2 = np.array(geometries[d2].centroid)
            centroid_distances[i, j] = np.linalg.norm(c1 - c2)

    return {
        "domains": domains,
        "centroid_distance_matrix": centroid_distances.tolist(),
        "mean_centroid_distance": float(np.mean(centroid_distances[np.triu_indices(n, k=1)])),
    }


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

RESULTS_DIR = Path("experiments/_runs/geometric_analysis/runs")

# Robust metrics (length-independent based on prior analysis)
ROBUST_METRICS = [
    "mean_influence",
    "max_influence",
    "top_100_concentration",
    "mean_activation",
    "logit_entropy",
    "max_logit_prob"
]


def analyze_geometric_structure(results_path: Path = None, min_samples: int = 10):
    """
    Main analysis: compute geometric properties of domain clusters.

    Args:
        results_path: Path to attribution results JSON
        min_samples: Minimum samples per domain (filters out sparse labels)
    """
    results_path = results_path or Path("data/results/domain_attribution_metrics.json")

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["samples"])

    # Filter to domains with sufficient samples
    domain_counts = df["domain"].value_counts()
    valid_domains = domain_counts[domain_counts >= min_samples].index
    df = df[df["domain"].isin(valid_domains)]

    print(f"Filtered to {len(valid_domains)} domains with >= {min_samples} samples")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    print(f"Geometric Analysis of Attribution Metric Space")
    print(f"=" * 50)
    print(f"Samples: {len(df)}")
    print(f"Output: {run_dir}")

    # Available metrics
    available_metrics = [m for m in ROBUST_METRICS if m in df.columns]
    print(f"Metrics: {available_metrics}")

    # Standardize metrics for geometric analysis
    # (important: we want shape, not scale)
    X = df[available_metrics].values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_standardized = (X - X_mean) / X_std

    df_std = df.copy()
    for i, m in enumerate(available_metrics):
        df_std[f"{m}_std"] = X_standardized[:, i]

    std_metrics = [f"{m}_std" for m in available_metrics]

    # --------------------------------------------------------
    # 1. COMPUTE GEOMETRY PER DOMAIN
    # --------------------------------------------------------
    print("\n--- Domain Geometries ---")

    domains = df["domain"].unique()
    geometries = {}

    for domain in domains:
        mask = df["domain"] == domain
        points = X_standardized[mask]

        geom = compute_domain_geometry(points, domain, available_metrics)
        geometries[domain] = geom

        print(f"\n{domain} (n={geom.n_samples}):")
        print(f"  Shape: {geom.shape_class}")
        print(f"  Axis ratios: b/a={geom.b_over_a:.3f}, c/a={geom.c_over_a:.3f}")
        print(f"  Effective dim: {geom.effective_dimensionality:.2f} / {len(available_metrics)}")
        print(f"  Entropy: {geom.eigenvalue_entropy:.3f}")
        print(f"  Misalignment: {geom.misalignment_angle:.1f}°")

    # --------------------------------------------------------
    # 2. CROSS-DOMAIN RELATIONSHIPS
    # --------------------------------------------------------
    print("\n--- Cross-Domain Geometry ---")

    inter_domain = compute_inter_domain_angles(geometries, df_std, std_metrics)
    print(f"Mean centroid distance: {inter_domain['mean_centroid_distance']:.3f}")

    # --------------------------------------------------------
    # 3. VISUALIZATIONS
    # --------------------------------------------------------

    # 3a. Shape classification summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axis ratio scatter
    ax = axes[0]
    for domain, geom in geometries.items():
        ax.scatter(geom.b_over_a, geom.c_over_a, s=100, label=domain)
        ax.annotate(domain, (geom.b_over_a, geom.c_over_a), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')

    # Add shape region boundaries
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.6, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.6, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel("b/a (Intermediate/Major)")
    ax.set_ylabel("c/a (Minor/Major)")
    ax.set_title("Domain Shape Classification")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)

    # Add region labels
    ax.text(0.95, 0.95, "Spherical", fontsize=8, ha='center', alpha=0.7)
    ax.text(0.3, 0.3, "Prolate", fontsize=8, ha='center', alpha=0.7)
    ax.text(0.9, 0.3, "Oblate", fontsize=8, ha='center', alpha=0.7)

    # Effective dimensionality bar
    ax = axes[1]
    domain_names = list(geometries.keys())
    eff_dims = [geometries[d].effective_dimensionality for d in domain_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(domain_names)))

    bars = ax.barh(domain_names, eff_dims, color=colors)
    ax.axvline(len(available_metrics), color='red', linestyle='--', label='Max possible')
    ax.set_xlabel("Effective Dimensionality")
    ax.set_title("Variance Distribution Across Metrics")
    ax.set_xlim(0, len(available_metrics) + 0.5)

    # Eigenvalue spectrum comparison
    ax = axes[2]
    for domain, geom in geometries.items():
        evr = geom.explained_variance_ratio[:len(available_metrics)]
        ax.plot(range(1, len(evr) + 1), evr, 'o-', label=domain, alpha=0.7)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Eigenvalue Spectra by Domain")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "domain_geometry_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: domain_geometry_summary.png")

    # 3b. Centroid distance heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    dist_matrix = np.array(inter_domain["centroid_distance_matrix"])
    im = ax.imshow(dist_matrix, cmap="YlOrRd")

    ax.set_xticks(range(len(domains)))
    ax.set_yticks(range(len(domains)))
    ax.set_xticklabels(inter_domain["domains"], rotation=45, ha="right")
    ax.set_yticklabels(inter_domain["domains"])

    for i in range(len(domains)):
        for j in range(len(domains)):
            ax.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Centroid Distance (standardized)")
    plt.title("Domain Centroid Distances in Metric Space")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "centroid_distances.png", dpi=150)
    plt.close()
    print(f"Saved: centroid_distances.png")

    # 3c. 3D PCA projection with ellipsoids
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_standardized)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for domain in domains:
        mask = df["domain"] == domain
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                  label=domain, alpha=0.6, s=30)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title("Domain Clusters in 3D Metric Space")
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "domain_clusters_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: domain_clusters_3d.png")

    # --------------------------------------------------------
    # 4. STATISTICAL TESTS
    # --------------------------------------------------------
    print("\n--- Statistical Tests ---")

    # Test: Are shape parameters significantly different across domains?
    shape_stats = {
        "eff_dim_variance": float(np.var([g.effective_dimensionality for g in geometries.values()])),
        "c_over_a_variance": float(np.var([g.c_over_a for g in geometries.values()])),
        "b_over_a_variance": float(np.var([g.b_over_a for g in geometries.values()])),
    }

    print(f"Variance in effective dimensionality: {shape_stats['eff_dim_variance']:.4f}")
    print(f"Variance in c/a ratio: {shape_stats['c_over_a_variance']:.4f}")
    print(f"Variance in b/a ratio: {shape_stats['b_over_a_variance']:.4f}")

    # --------------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------------

    results = {
        "metadata": {
            "timestamp": timestamp,
            "n_samples": len(df),
            "n_domains": len(domains),
            "metrics_used": available_metrics,
            "standardized": True,
        },
        "domain_geometries": {d: asdict(g) for d, g in geometries.items()},
        "inter_domain": inter_domain,
        "shape_statistics": shape_stats,
        "pca": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
        }
    }

    with open(run_dir / "geometric_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    report = generate_geometric_report(geometries, inter_domain, shape_stats, available_metrics)
    with open(run_dir / "geometric_analysis_report.md", "w") as f:
        f.write(report)

    print(f"\nAnalysis complete: {run_dir}")

    return results


def generate_geometric_report(
    geometries: Dict[str, DomainGeometry],
    inter_domain: Dict,
    shape_stats: Dict,
    metrics: List[str]
) -> str:
    """Generate markdown report."""

    lines = [
        "# Geometric Analysis of Attribution Metric Space",
        "",
        "## Methodology",
        "",
        "Adapted from AIDA-TNG galaxy morphology analysis:",
        "- Each sample is a point in 6D metric space",
        "- Domain clusters are analyzed as point clouds",
        "- Covariance matrix eigendecomposition reveals shape",
        "",
        f"**Metrics used:** {', '.join(metrics)}",
        "",
        "## Domain Geometries",
        "",
        "| Domain | n | Shape | b/a | c/a | Eff. Dim | Entropy | Misalign |",
        "|--------|---|-------|-----|-----|----------|---------|----------|",
    ]

    for domain, geom in geometries.items():
        lines.append(
            f"| {domain} | {geom.n_samples} | {geom.shape_class} | "
            f"{geom.b_over_a:.3f} | {geom.c_over_a:.3f} | "
            f"{geom.effective_dimensionality:.2f} | {geom.eigenvalue_entropy:.3f} | "
            f"{geom.misalignment_angle:.1f}° |"
        )

    lines.extend([
        "",
        "## Shape Interpretation",
        "",
        "- **Spherical**: Uniform variance across metrics (isotropic)",
        "- **Oblate**: Disk-like, variance concentrated in 2 dimensions",
        "- **Prolate**: Cigar-like, variance along one dominant axis",
        "- **Triaxial**: Irregular, mixed variance distribution",
        "",
        "## Key Findings",
        "",
    ])

    # Find most/least spherical
    sorted_by_sphericity = sorted(geometries.items(),
                                   key=lambda x: x[1].c_over_a * x[1].b_over_a,
                                   reverse=True)

    most_spherical = sorted_by_sphericity[0][0]
    least_spherical = sorted_by_sphericity[-1][0]

    lines.append(f"- **Most spherical domain**: {most_spherical} (uniform variance)")
    lines.append(f"- **Most elongated domain**: {least_spherical} (concentrated variance)")

    # Effective dimensionality insights
    sorted_by_eff_dim = sorted(geometries.items(),
                               key=lambda x: x[1].effective_dimensionality,
                               reverse=True)

    lines.append(f"- **Highest effective dimensionality**: {sorted_by_eff_dim[0][0]} "
                f"({sorted_by_eff_dim[0][1].effective_dimensionality:.2f})")
    lines.append(f"- **Lowest effective dimensionality**: {sorted_by_eff_dim[-1][0]} "
                f"({sorted_by_eff_dim[-1][1].effective_dimensionality:.2f})")

    lines.extend([
        "",
        "## Cross-Domain Distances",
        "",
        f"Mean centroid distance: {inter_domain['mean_centroid_distance']:.3f} (standardized units)",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Geometric Analysis of Attribution Metric Space")
    parser.add_argument("--analyze", action="store_true", help="Run geometric analysis")
    parser.add_argument("--results-path", type=Path, default=None,
                       help="Path to attribution results JSON")
    parser.add_argument("--min-samples", type=int, default=10,
                       help="Minimum samples per domain")

    args = parser.parse_args()

    if args.analyze:
        analyze_geometric_structure(args.results_path, args.min_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
