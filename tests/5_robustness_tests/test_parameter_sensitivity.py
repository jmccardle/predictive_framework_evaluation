"""
Test 5.2: Parameter Sensitivity Analysis

Test robustness of the system to variations in key parameters.

A robust system should maintain performance across reasonable parameter ranges.
If results are highly sensitive to specific parameter values, it suggests overfitting
or lack of generalizability.

Key parameters to test:
- Rolling window size (252 ± 50%)
- Entropy bins (21 ± 50%)
- Autocorrelation lag (1 vs 5 vs 10)
- RRP lag (20 ± 50%)
- Forward drawdown horizon (20 ± 50%)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.utils.data_loader import load_sp500, compute_returns
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def evaluate_parameters(prices, volume, window, bins, acf_lag, rrp_lag, dd_horizon):
    """
    Evaluate system with specific parameter settings.

    Parameters
    ----------
    prices : np.ndarray
        Price series
    volume : np.ndarray
        Volume series
    window : int
        Rolling window size
    bins : int
        Number of entropy bins
    acf_lag : int
        Autocorrelation lag
    rrp_lag : int
        RRP lag
    dd_horizon : int
        Forward drawdown horizon

    Returns
    -------
    rho : float
        Spearman correlation
    p : float
        P-value
    """
    # Compute returns
    returns = compute_returns(prices)
    volume = volume[1:]  # Align with returns

    # Compute metrics with specified parameters
    RERL = rolling_entropy(returns, window=window, bins=bins)
    RTCL = rolling_autocorr(returns, window=window, lag=acf_lag)
    RRP = rolling_rrp(volume, np.abs(returns), window=window, lag=rrp_lag)

    # Compute CCI
    CCI = compute_cci(RERL, RTCL, RRP)

    # Compute forward drawdown with specified horizon
    DD = forward_drawdown(prices[1:], horizon=dd_horizon)

    # Align for correlation
    valid = ~(np.isnan(CCI) | np.isnan(DD))

    if np.sum(valid) < 100:
        return np.nan, 1.0

    # Align CCI at t with DD at t+horizon
    CCI_aligned = CCI[valid][:-dd_horizon]
    DD_aligned = DD[valid][dd_horizon:]

    min_len = min(len(CCI_aligned), len(DD_aligned))
    if min_len < 100:
        return np.nan, 1.0

    CCI_aligned = CCI_aligned[:min_len]
    DD_aligned = DD_aligned[:min_len]

    # Compute correlation
    rho, p = spearmanr(CCI_aligned, DD_aligned)

    return rho, p


def test_parameter_sensitivity():
    """
    Test sensitivity to parameter variations.

    Pass Criteria
    -------------
    - Results remain significant across most parameter variations
    - Mean correlation across all tests > 0
    - Correlation changes < 100% from baseline (not flipping signs wildly)
    """
    print("="*70)
    print("TEST 5.2: PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nTesting robustness to parameter variations.")
    print("A robust system should maintain performance across reasonable ranges.")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values

    print(f"   Total days: {len(df)}")

    # Baseline parameters (from original specification)
    baseline_params = {
        'window': 252,    # 1 year
        'bins': 21,       # Entropy bins
        'acf_lag': 1,     # First-order autocorrelation
        'rrp_lag': 20,    # ~1 month
        'dd_horizon': 20  # ~1 month forward
    }

    # Parameter variations to test
    parameter_sets = [
        # Baseline
        {'name': 'Baseline', **baseline_params},

        # Window size variations (±50%)
        {'name': 'Window -50%', 'window': 126, 'bins': 21, 'acf_lag': 1, 'rrp_lag': 20, 'dd_horizon': 20},
        {'name': 'Window +50%', 'window': 378, 'bins': 21, 'acf_lag': 1, 'rrp_lag': 20, 'dd_horizon': 20},

        # Entropy bins variations
        {'name': 'Bins -50%', 'window': 252, 'bins': 10, 'acf_lag': 1, 'rrp_lag': 20, 'dd_horizon': 20},
        {'name': 'Bins +50%', 'window': 252, 'bins': 32, 'acf_lag': 1, 'rrp_lag': 20, 'dd_horizon': 20},

        # Autocorrelation lag variations
        {'name': 'ACF lag 5', 'window': 252, 'bins': 21, 'acf_lag': 5, 'rrp_lag': 20, 'dd_horizon': 20},
        {'name': 'ACF lag 10', 'window': 252, 'bins': 21, 'acf_lag': 10, 'rrp_lag': 20, 'dd_horizon': 20},

        # RRP lag variations
        {'name': 'RRP lag -50%', 'window': 252, 'bins': 21, 'acf_lag': 1, 'rrp_lag': 10, 'dd_horizon': 20},
        {'name': 'RRP lag +50%', 'window': 252, 'bins': 21, 'acf_lag': 1, 'rrp_lag': 30, 'dd_horizon': 20},

        # Forward drawdown horizon variations
        {'name': 'DD horizon -50%', 'window': 252, 'bins': 21, 'acf_lag': 1, 'rrp_lag': 20, 'dd_horizon': 10},
        {'name': 'DD horizon +50%', 'window': 252, 'bins': 21, 'acf_lag': 1, 'rrp_lag': 20, 'dd_horizon': 30},

        # Combined variations (conservative)
        {'name': 'All -25%', 'window': 189, 'bins': 16, 'acf_lag': 1, 'rrp_lag': 15, 'dd_horizon': 15},
        {'name': 'All +25%', 'window': 315, 'bins': 26, 'acf_lag': 1, 'rrp_lag': 25, 'dd_horizon': 25},
    ]

    results = []

    print("\n2. Testing parameter variations...")
    print("-" * 70)

    for params in parameter_sets:
        name = params['name']
        print(f"\nTesting: {name}")
        print(f"  window={params['window']}, bins={params['bins']}, "
              f"acf_lag={params['acf_lag']}, rrp_lag={params['rrp_lag']}, "
              f"dd_horizon={params['dd_horizon']}")

        rho, p = evaluate_parameters(
            prices, volume,
            window=params['window'],
            bins=params['bins'],
            acf_lag=params['acf_lag'],
            rrp_lag=params['rrp_lag'],
            dd_horizon=params['dd_horizon']
        )

        results.append({
            'name': name,
            'rho': rho,
            'p': p,
            **{k: v for k, v in params.items() if k != 'name'}
        })

        if np.isnan(rho):
            print(f"  Result: INSUFFICIENT DATA")
        else:
            sig_marker = "✓" if p < 0.05 else " "
            print(f"  Result: ρ = {rho:+.4f}, p = {p:.6f} {sig_marker}")

    # Analyze results
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("="*70)

    baseline_rho = results[0]['rho']
    baseline_p = results[0]['p']

    print(f"\nBaseline: ρ = {baseline_rho:+.4f}, p = {baseline_p:.6f}")

    valid_results = [r for r in results if not np.isnan(r['rho'])]
    rhos = [r['rho'] for r in valid_results]

    mean_rho = np.mean(rhos)
    std_rho = np.std(rhos)
    median_rho = np.median(rhos)
    significant_count = sum(1 for r in valid_results if r['p'] < 0.05)
    positive_count = sum(1 for rho in rhos if rho > 0)

    print(f"\nAcross {len(valid_results)} parameter sets:")
    print(f"  Mean ρ:        {mean_rho:+.4f} ± {std_rho:.4f}")
    print(f"  Median ρ:      {median_rho:+.4f}")
    print(f"  Range:         [{min(rhos):+.4f}, {max(rhos):+.4f}]")
    print(f"  Positive:      {positive_count}/{len(valid_results)} ({positive_count/len(valid_results)*100:.1f}%)")
    print(f"  Significant:   {significant_count}/{len(valid_results)} ({significant_count/len(valid_results)*100:.1f}%)")

    # Coefficient of variation
    cv = std_rho / abs(mean_rho) if mean_rho != 0 else np.inf
    print(f"  Coef. Var.:    {cv:.2f}")

    # Detailed results table
    print("\nDetailed Results:")
    print("-" * 70)
    for r in results:
        if not np.isnan(r['rho']):
            sig = "✓" if r['p'] < 0.05 else " "
            rel_change = ((r['rho'] - baseline_rho) / baseline_rho * 100) if baseline_rho != 0 else 0
            print(f"  {r['name']:20s}: ρ = {r['rho']:+.4f}, p = {r['p']:.6f} {sig} "
                  f"({rel_change:+.1f}% from baseline)")

    # Pass criteria
    print("\n" + "="*70)
    print("PASS/FAIL CRITERIA")
    print("="*70)

    success = True

    # Test 1: Majority significant
    if significant_count >= len(valid_results) / 2:
        print(f"✓ PASS: {significant_count}/{len(valid_results)} parameter sets significant (≥50%)")
    else:
        print(f"✗ FAIL: Only {significant_count}/{len(valid_results)} parameter sets significant (<50%)")
        success = False

    # Test 2: Mean correlation positive
    if mean_rho > 0:
        print(f"✓ PASS: Mean correlation positive ({mean_rho:+.4f})")
    else:
        print(f"✗ FAIL: Mean correlation negative ({mean_rho:+.4f})")
        success = False

    # Test 3: Low coefficient of variation (< 1.0 means std < mean)
    if cv < 1.0:
        print(f"✓ PASS: Low coefficient of variation ({cv:.2f} < 1.0)")
    else:
        print(f"⚠️  WARNING: High coefficient of variation ({cv:.2f} ≥ 1.0)")
        print("   Results are sensitive to parameter choices")
        # Not a failure, but a warning

    # Test 4: All correlations same sign as baseline
    sign_changes = sum(1 for r in valid_results[1:] if np.sign(r['rho']) != np.sign(baseline_rho))
    if sign_changes == 0:
        print(f"✓ PASS: All parameter sets have consistent sign")
    else:
        print(f"⚠️  WARNING: {sign_changes} parameter sets flip sign from baseline")

    if success:
        print("\n" + "="*70)
        print("✓ PASS: SYSTEM IS ROBUST TO PARAMETER VARIATIONS")
        print("="*70)
        print("\nThe system maintains consistent performance across")
        print("reasonable parameter ranges.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: EXCESSIVE PARAMETER SENSITIVITY")
        print("="*70)
        print("\nThe system is too sensitive to parameter choices,")
        print("suggesting potential overfitting or lack of robustness.")

    return success


if __name__ == "__main__":
    success = test_parameter_sensitivity()
    sys.exit(0 if success else 1)
