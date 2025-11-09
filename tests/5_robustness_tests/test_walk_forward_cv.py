"""
Test 5.1: Walk-Forward Cross-Validation

Test the system on multiple out-of-sample periods using sliding window
cross-validation with strict temporal causality.

Based on TEST_PROGRAM_SPECIFICATION.md Section 5.1

Reference from the original paper:
"Walk-forward cross-validation across independent folds validates robustness."
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.utils.data_loader import load_sp500, compute_returns
from src.utils.normalization import compute_train_statistics, zscore_causal
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def evaluate_fold(train_prices, test_prices, train_volume, test_volume):
    """
    Evaluate a single fold with strict causal normalization.

    Parameters
    ----------
    train_prices : np.ndarray
        Training period prices
    test_prices : np.ndarray
        Test period prices
    train_volume : np.ndarray
        Training period volume
    test_volume : np.ndarray
        Test period volume

    Returns
    -------
    rho : float
        Spearman correlation between CCI and forward drawdown
    p : float
        P-value of correlation
    """
    # Compute returns
    train_returns = compute_returns(train_prices)
    test_returns = compute_returns(test_prices)

    # Align volumes with returns
    train_volume = train_volume[1:]
    test_volume = test_volume[1:]

    # Compute metrics on training data
    train_RERL = rolling_entropy(train_returns, window=252, bins=21)
    train_RTCL = rolling_autocorr(train_returns, window=252, lag=1)
    train_RRP = rolling_rrp(train_volume, np.abs(train_returns), window=252, lag=20)

    # Compute metrics on test data
    test_RERL = rolling_entropy(test_returns, window=252, bins=21)
    test_RTCL = rolling_autocorr(test_returns, window=252, lag=1)
    test_RRP = rolling_rrp(test_volume, np.abs(test_returns), window=252, lag=20)

    # Causal normalization: Use TRAIN statistics only
    # Compute train statistics for each metric
    train_stats_RERL = compute_train_statistics(train_RERL)
    train_stats_RTCL = compute_train_statistics(train_RTCL)
    train_stats_RRP = compute_train_statistics(train_RRP)

    # Normalize test metrics using train statistics
    test_RERL_norm = zscore_causal(test_RERL, train_stats_RERL)
    test_RTCL_norm = zscore_causal(test_RTCL, train_stats_RTCL)
    test_RRP_norm = zscore_causal(test_RRP, train_stats_RRP)

    # Compute CCI on normalized test metrics
    test_CCI = compute_cci(test_RERL_norm, test_RTCL_norm, test_RRP_norm)

    # Compute forward drawdown on test period
    test_DD = forward_drawdown(test_prices[1:], horizon=20)

    # Align for correlation (CCI at t, DD at t+20)
    valid = ~(np.isnan(test_CCI) | np.isnan(test_DD))

    if np.sum(valid) < 50:
        return np.nan, 1.0

    # Take first N-20 CCI values and last N-20 DD values
    test_CCI_aligned = test_CCI[valid][:-20]
    test_DD_aligned = test_DD[valid][20:]

    min_len = min(len(test_CCI_aligned), len(test_DD_aligned))
    if min_len < 50:
        return np.nan, 1.0

    test_CCI_aligned = test_CCI_aligned[:min_len]
    test_DD_aligned = test_DD_aligned[:min_len]

    # Compute correlation
    rho, p = spearmanr(test_CCI_aligned, test_DD_aligned)

    return rho, p


def test_walk_forward_cv(n_folds=5):
    """
    Walk-forward cross-validation with sliding window.

    Uses strict temporal causality:
    - Each fold has separate train/test periods
    - Test periods are strictly AFTER training periods
    - Normalization uses ONLY training statistics

    Pass Criteria
    -------------
    - ≥50% of folds show significant correlation (p < 0.05)
    - Mean correlation is positive
    """
    print("="*70)
    print("TEST 5.1: WALK-FORWARD CROSS-VALIDATION")
    print("="*70)
    print("\nTesting system robustness across multiple out-of-sample periods.")
    print("Sliding window with strict temporal causality.")

    # Load full dataset
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values
    dates = pd.to_datetime(df["Date"])

    print(f"   Total days: {len(df)}")
    print(f"   Date range: {dates.min()} to {dates.max()}")

    # Cross-validation parameters
    train_size = 2000  # ~8 years of trading days
    test_size = 200    # ~9 months
    step_size = 100    # Slide forward by 100 days each fold

    print(f"\n2. Cross-validation setup:")
    print(f"   Training size: {train_size} days (~8 years)")
    print(f"   Test size: {test_size} days (~9 months)")
    print(f"   Step size: {step_size} days")
    print(f"   Target folds: {n_folds}")

    fold_results = []

    print("\n3. Running cross-validation folds...")
    print("-" * 70)

    for fold in range(n_folds):
        start_idx = fold * step_size

        # Check if we have enough data for this fold
        if start_idx + train_size + test_size > len(prices):
            print(f"\nFold {fold+1}: Insufficient data, stopping at {fold} folds")
            break

        # Extract train and test periods
        train_prices = prices[start_idx:start_idx+train_size]
        test_prices = prices[start_idx+train_size:start_idx+train_size+test_size]

        train_volume = volume[start_idx:start_idx+train_size]
        test_volume = volume[start_idx+train_size:start_idx+train_size+test_size]

        train_dates = dates.iloc[start_idx:start_idx+train_size]
        test_dates = dates.iloc[start_idx+train_size:start_idx+train_size+test_size]

        print(f"\nFold {fold+1}/{n_folds}")
        print(f"  Train: [{start_idx}:{start_idx+train_size}] "
              f"{train_dates.min().date()} to {train_dates.max().date()}")
        print(f"  Test:  [{start_idx+train_size}:{start_idx+train_size+test_size}] "
              f"{test_dates.min().date()} to {test_dates.max().date()}")

        # Evaluate fold
        rho, p = evaluate_fold(train_prices, test_prices, train_volume, test_volume)

        fold_results.append({
            'fold': fold + 1,
            'rho': rho,
            'p': p,
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max()
        })

        if np.isnan(rho):
            print(f"  Result: INSUFFICIENT DATA")
        else:
            sig_marker = "✓" if p < 0.05 else " "
            print(f"  Result: ρ = {rho:+.4f}, p = {p:.6f} {sig_marker}")

    # Aggregate results
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)

    rhos = [r['rho'] for r in fold_results if not np.isnan(r['rho'])]
    ps = [r['p'] for r in fold_results if not np.isnan(r['p'])]

    if len(rhos) == 0:
        print("✗ FAIL: No valid folds")
        return False

    mean_rho = np.mean(rhos)
    std_rho = np.std(rhos)
    median_rho = np.median(rhos)
    significant_folds = sum(1 for p in ps if p < 0.05)
    positive_folds = sum(1 for r in rhos if r > 0)

    print(f"\nCompleted folds: {len(rhos)}")
    print(f"Mean ρ:          {mean_rho:+.4f} ± {std_rho:.4f}")
    print(f"Median ρ:        {median_rho:+.4f}")
    print(f"Range:           [{min(rhos):+.4f}, {max(rhos):+.4f}]")
    print(f"Positive folds:  {positive_folds}/{len(rhos)} ({positive_folds/len(rhos)*100:.1f}%)")
    print(f"Significant:     {significant_folds}/{len(rhos)} ({significant_folds/len(rhos)*100:.1f}%)")

    # Detailed fold results
    print("\nFold Details:")
    print("-" * 70)
    for r in fold_results:
        if not np.isnan(r['rho']):
            sig = "✓" if r['p'] < 0.05 else " "
            print(f"  Fold {r['fold']}: ρ = {r['rho']:+.4f}, p = {r['p']:.6f} {sig}")

    # Pass criteria
    print("\n" + "="*70)
    print("PASS/FAIL CRITERIA")
    print("="*70)

    success = True

    # Test 1: Majority of folds significant
    if significant_folds >= len(rhos) / 2:
        print(f"✓ PASS: {significant_folds}/{len(rhos)} folds significant (≥50%)")
    else:
        print(f"✗ FAIL: Only {significant_folds}/{len(rhos)} folds significant (<50%)")
        success = False

    # Test 2: Mean correlation positive
    if mean_rho > 0:
        print(f"✓ PASS: Mean correlation positive ({mean_rho:+.4f})")
    else:
        print(f"✗ FAIL: Mean correlation negative ({mean_rho:+.4f})")
        success = False

    # Test 3: At least 3 valid folds
    if len(rhos) >= 3:
        print(f"✓ PASS: Sufficient folds completed ({len(rhos)})")
    else:
        print(f"⚠️  WARNING: Only {len(rhos)} folds completed (expected {n_folds})")

    if success:
        print("\n" + "="*70)
        print("✓ PASS: SYSTEM IS ROBUST ACROSS MULTIPLE PERIODS")
        print("="*70)
        print("\nThe system shows consistent predictive power across")
        print("multiple out-of-sample test periods.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: INSUFFICIENT ROBUSTNESS")
        print("="*70)
        print("\nThe system does not show consistent performance across")
        print("multiple out-of-sample periods.")

    return success


if __name__ == "__main__":
    success = test_walk_forward_cv(n_folds=5)
    sys.exit(0 if success else 1)
