"""
src/geometry.py
Geometric Analysis of SAE Feature Activations.
Implements the 'Inertia Tensor' methodology from AIDA-TNG applied to Neural Latent Space.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union, Literal
from dataclasses import dataclass

@dataclass
class GeometricMetrics:
    # Principal Axis Ratios (The 3D shape proxy)
    c_over_a: float  # Minor/Major (Sphericity)
    b_over_a: float  # Intermediate/Major (Elongation)
    shape_class: Literal["Spherical", "Oblate", "Prolate", "Triaxial"]
    
    # Global Topology
    eigenvalue_entropy: float  # How dispersed is the energy across dimensions?
    dimensionality: float      # Participation Ratio (Effective Dimensionality)
    
    # Alignment
    misalignment_angle: float  # Angle between primary axis and centroid vector (degrees)
    
    # Raw Data (for plotting)
    eigenvalues: np.ndarray    # Top k eigenvalues
    principal_axes: np.ndarray # Top 3 eigenvectors

def compute_inertia_tensor(
    feature_acts: torch.Tensor, 
    decoder_weights: torch.Tensor,
    top_k_components: int = 3
) -> GeometricMetrics:
    """
    Computes the geometric properties of a set of active SAE features.
    
    Args:
        feature_acts: (N_features,) Tensor of activation magnitudes (w_i). 
                      Should be pre-filtered (only active features > 0).
        decoder_weights: (N_features, d_model) Tensor of feature directions (r_i).
                         These are the rows of W_dec corresponding to active indices.
    
    Returns:
        GeometricMetrics dataclass.
    """
    
    # 1. PREPARE MASSES AND POSITIONS
    # w_i: "Mass" of the feature (activation strength)
    # r_i: "Position" of the feature (direction in residual stream)
    w = feature_acts.detach().cpu().numpy()
    r = decoder_weights.detach().cpu().numpy()
    
    # Normalize weights to sum to 1 (treat as probability distribution of mass)
    total_mass = np.sum(w)
    if total_mass == 0:
        return _empty_metrics(top_k_components)
        
    w_norm = w / total_mass
    
    # 2. COMPUTE CENTER OF MASS (Centroid)
    # R_cm = sum(w_i * r_i)
    centroid = np.dot(w_norm, r)
    
    # 3. CENTER THE CLOUD
    # r'_i = r_i - R_cm
    r_centered = r - centroid
    
    # 4. COMPUTE INERTIA / COVARIANCE TENSOR
    # In shape analysis, this is the weighted Covariance Matrix.
    # C = sum(w_i * r'_i * r'_i^T)
    # We use broadcasting for efficiency: (N, d) * (N, 1) -> (N, d)
    weighted_r = r_centered * np.sqrt(w_norm)[:, np.newaxis]
    inertia_tensor = np.dot(weighted_r.T, weighted_r)  # (d_model, d_model)
    
    # 5. EIGENDECOMPOSITION
    # We only need the top k eigenvalues to define the "3D Shape" of the thought.
    # Using eigh because covariance is symmetric/Hermitian.
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    
    # Sort descending (largest variance first)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    
    # 6. CALCULATE SHAPE METRICS (AIDA-TNG inspired)
    # We map the high-dim space to an equivalent ellipsoid in 3D defined by top 3 axes.
    # lambda_1 (major), lambda_2 (intermediate), lambda_3 (minor)
    # These represent VARIANCE along axes (lengths squared).
    
    L1, L2, L3 = eigvals[0], eigvals[1], eigvals[2]
    
    # Avoid div by zero
    L1 = max(L1, 1e-9)
    
    # Axis lengths are sqrt of eigenvalues
    a = np.sqrt(L1)
    b = np.sqrt(max(L2, 0))
    c = np.sqrt(max(L3, 0))
    
    c_over_a = c / a
    b_over_a = b / a
    
    # Classify Shape
    # Thresholds can be tuned, but standard physics definitions:
    # Spherical: b/a > 0.9 and c/a > 0.9
    # Oblate (Disk): b/a > 0.8 and c/a < 0.5
    # Prolate (Cigar): b/a < 0.5 and c/a < 0.5
    # Triaxial: Everything else
    shape_class = _classify_shape(c_over_a, b_over_a)
    
    # 7. MISALIGNMENT (Angle between Centroid and Major Axis)
    # Does the "average meaning" align with the "direction of maximum variance"?
    major_axis = eigvecs[:, 0]
    
    # Cosine similarity
    cos_sim = np.dot(centroid, major_axis) / (np.linalg.norm(centroid) * np.linalg.norm(major_axis) + 1e-9)
    angle_rad = np.arccos(np.clip(np.abs(cos_sim), 0, 1)) # abs because axis is bidirectional
    misalignment = np.degrees(angle_rad)
    
    # 8. EFFECTIVE DIMENSIONALITY (Participation Ratio)
    # PR = (sum(lambda)^2) / sum(lambda^2)
    pr_num = np.sum(eigvals)**2
    pr_den = np.sum(eigvals**2)
    eff_dim = pr_num / (pr_den + 1e-9)

    return GeometricMetrics(
        c_over_a=float(c_over_a),
        b_over_a=float(b_over_a),
        shape_class=shape_class,
        eigenvalue_entropy=_compute_entropy(eigvals),
        dimensionality=float(eff_dim),
        misalignment_angle=float(misalignment),
        eigenvalues=eigvals[:top_k_components],
        principal_axes=eigvecs[:, :top_k_components]
    )

def _classify_shape(c_a: float, b_a: float) -> str:
    # AIDA-TNG / Planetary Science thresholds
    if b_a > 0.9 and c_a > 0.9:
        return "Spherical"
    elif b_a > 0.8 and c_a < 0.6:
        return "Oblate" # Disk-like
    elif b_a < 0.6 and c_a < 0.6:
        return "Prolate" # Cigar-like
    else:
        return "Triaxial"

def _compute_entropy(eigenvalues: np.ndarray) -> float:
    # Normalized Shannon entropy of the eigenvalue spectrum
    # High entropy = Energy spread out (Spherical/Confused)
    # Low entropy = Energy concentrated (Directed/Sharp)
    total = np.sum(eigenvalues)
    if total == 0: return 0.0
    probs = eigenvalues / total
    # Filter zeros for log
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def _empty_metrics(k: int) -> GeometricMetrics:
    return GeometricMetrics(
        c_over_a=0.0, b_over_a=0.0, shape_class="Spherical",
        eigenvalue_entropy=0.0, dimensionality=0.0, misalignment_angle=0.0,
        eigenvalues=np.zeros(k), principal_axes=np.zeros((k, k))
    )