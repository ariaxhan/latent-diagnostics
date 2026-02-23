#!/usr/bin/env python3
"""
Variance decomposition via sequential regression.

Quantifies how much variance in attribution metrics is explained by:
- Domain (task type)
- Text length
- Residual (unexplained)

Expected: Length dominates n_active, but domain explains significant variance
in influence/concentration even after controlling for length.
"""

import json
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from pathlib import Path


def load_data(data_path):
    """Load domain attribution metrics and compute text length."""
    with open(data_path) as f:
        data = json.load(f)

    samples = data['samples']

    # Extract fields
    records = []
    for s in samples:
        records.append({
            'domain': s['domain'],
            'text': s['text'],
            'n_active': s['n_active'],
            'mean_influence': s['mean_influence'],
            'concentration': s['top_100_concentration']
        })

    df = pd.DataFrame(records)

    # Compute text length
    df['length'] = df['text'].str.len()

    return df


def sequential_variance_decomposition(y, domain_dummies, length):
    """
    Sequential R² decomposition: domain → length → residual

    Returns:
        domain_r2: Variance explained by domain alone
        length_r2: Additional variance explained by length
        residual: Unexplained variance
    """
    # Convert to numpy arrays to avoid pandas dtype issues
    y = np.asarray(y, dtype=float)
    X_domain = np.asarray(domain_dummies, dtype=float)
    X_length = np.asarray(length, dtype=float)

    # Model 1: domain only
    X1 = add_constant(X_domain)
    model1 = OLS(y, X1).fit()
    domain_r2 = model1.rsquared

    # Model 2: domain + length
    X2 = add_constant(np.column_stack([X_domain, X_length]))
    model2 = OLS(y, X2).fit()
    full_r2 = model2.rsquared

    # Length contribution = full R² - domain R²
    length_r2 = full_r2 - domain_r2

    # Residual
    residual = 1.0 - full_r2

    return domain_r2, length_r2, residual


def main():
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'results' / 'domain_attribution_metrics.json'
    df = load_data(data_path)

    print(f"Loaded {len(df)} samples across domains: {df['domain'].unique().tolist()}")
    print()

    # Create domain dummy variables
    domain_dummies = pd.get_dummies(df['domain'], prefix='domain', drop_first=True)

    # Metrics to analyze
    metrics = [
        ('n_active', 'n_active'),
        ('mean_influence', 'mean_influence'),
        ('concentration', 'concentration')
    ]

    # Results table
    results = []

    for metric_name, metric_col in metrics:
        y = df[metric_col]
        length = df[['length']]

        domain_r2, length_r2, residual = sequential_variance_decomposition(
            y, domain_dummies, length
        )

        results.append({
            'Metric': metric_name,
            'Domain R²': f"{domain_r2 * 100:.1f}%",
            'Length R²': f"{length_r2 * 100:.1f}%",
            'Residual': f"{residual * 100:.1f}%"
        })

    # Print table
    results_df = pd.DataFrame(results)

    print("=" * 70)
    print("VARIANCE DECOMPOSITION: metric ~ domain + length")
    print("=" * 70)
    print()
    print(results_df.to_string(index=False))
    print()
    print("Interpretation:")
    print("  Domain R²  = variance explained by task domain alone")
    print("  Length R²  = additional variance explained by text length")
    print("  Residual   = unexplained variance")
    print()


if __name__ == '__main__':
    main()
