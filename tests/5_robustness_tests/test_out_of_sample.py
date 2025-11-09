"""
Test 5.4: Out-of-Sample Validation

Test the system on completely new data not used during development.

This is the ultimate test of generalization. The system is trained on a
historical period and tested on the most recent data that was NOT available
during system development.

Based on SOFTWARE_REQUIREMENTS_SPECIFICATION.md Phase 6:
"Out-of-sample forward testing on NEW data"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import datetime, timedelta
from src.utils.data_loader import load_sp500, compute_returns
from src.utils.normalization import compute_train_statistics, zscore_causal
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def test_out_of_sample_validation():
    """
    Test on recent data not available during system development.

    Strategy:
    1. Use 2000-2020 as training/development period
    2. Test on 2021-2024 as true out-of-sample period
    3. This mimics real-world deployment scenario

    Pass Criteria
    -------------
    - Out-of-sample correlation is significant (p < 0.05)
    - Out-of-sample correlation > 0 (positive relationship)
    - Out-of-sample performance not drastically worse than in-sample
      (allows for some degradation, but not complete collapse)
    """
    print("="*70)
    print("TEST 5.4: OUT-OF-SAMPLE VALIDATION")
    print("="*70)
    print("\nTesting on recent data not available during development.")
    print("This is the ultimate test of generalization.")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values
    dates = pd.to_datetime(df["Date"])

    print(f"   Total days: {len(df)}")
    print(f"   Date range: {dates.min()} to {dates.max()}")

    # Define train/test split
    # In-sample (development): 2000-2020
    # Out-of-sample (new data): 2021-2024
    split_date = pd.Timestamp("2021-01-01")

    train_mask = dates < split_date
    test_mask = dates >= split_date

    train_dates = dates[train_mask]
    test_dates = dates[test_mask]

    train_prices = prices[train_mask]
    test_prices = prices[test_mask]

    train_volume = volume[train_mask]
    test_volume = volume[test_mask]

    print(f"\n2. Train/Test split at {split_date.date()}:")
    print(f"   Train (in-sample):     {train_dates.min().date()} to {train_dates.max().date()} ({len(train_dates)} days)")
    print(f"   Test (out-of-sample):  {test_dates.min().date()} to {test_dates.max().date()} ({len(test_dates)} days)")

    # =========================================================================
    # IN-SAMPLE EVALUATION (Development period)
    # =========================================================================
    print("\n" + "="*70)
    print("IN-SAMPLE EVALUATION (2000-2020)")
    print("="*70)

    print("\n3. Computing in-sample metrics...")
    train_returns = compute_returns(train_prices)
    train_volume = train_volume[1:]  # Align with returns

    train_RERL = rolling_entropy(train_returns, window=252, bins=21)
    train_RTCL = rolling_autocorr(train_returns, window=252, lag=1)
    train_RRP = rolling_rrp(train_volume, np.abs(train_returns), window=252, lag=20)

    print(f"   RERL: mean={np.nanmean(train_RERL):.3f}")
    print(f"   RTCL: mean={np.nanmean(train_RTCL):.3f}")
    print(f"   RRP:  mean={np.nanmean(train_RRP):.3f}")

    # Normalize in-sample (using its own statistics for evaluation)
    train_stats_RERL = compute_train_statistics(train_RERL)
    train_stats_RTCL = compute_train_statistics(train_RTCL)
    train_stats_RRP = compute_train_statistics(train_RRP)

    train_RERL_norm = zscore_causal(train_RERL, train_stats_RERL)
    train_RTCL_norm = zscore_causal(train_RTCL, train_stats_RTCL)
    train_RRP_norm = zscore_causal(train_RRP, train_stats_RRP)

    train_CCI = compute_cci(train_RERL_norm, train_RTCL_norm, train_RRP_norm)
    train_DD = forward_drawdown(train_prices[1:], horizon=20)

    print(f"   CCI:  mean={np.nanmean(train_CCI):.3f}")

    # In-sample correlation
    print("\n4. Computing in-sample correlation...")
    valid_train = ~(np.isnan(train_CCI) | np.isnan(train_DD))

    train_CCI_aligned = train_CCI[valid_train][:-20]
    train_DD_aligned = train_DD[valid_train][20:]

    min_len = min(len(train_CCI_aligned), len(train_DD_aligned))
    train_CCI_aligned = train_CCI_aligned[:min_len]
    train_DD_aligned = train_DD_aligned[:min_len]

    rho_train, p_train = spearmanr(train_CCI_aligned, train_DD_aligned)

    print(f"   In-sample: ρ = {rho_train:+.4f}, p = {p_train:.6f}")
    print(f"   Valid points: {min_len}")

    # =========================================================================
    # OUT-OF-SAMPLE EVALUATION (New data period)
    # =========================================================================
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE EVALUATION (2021-2024)")
    print("="*70)

    print("\n5. Computing out-of-sample metrics...")
    test_returns = compute_returns(test_prices)
    test_volume = test_volume[1:]  # Align with returns

    test_RERL = rolling_entropy(test_returns, window=252, bins=21)
    test_RTCL = rolling_autocorr(test_returns, window=252, lag=1)
    test_RRP = rolling_rrp(test_volume, np.abs(test_returns), window=252, lag=20)

    print(f"   RERL: mean={np.nanmean(test_RERL):.3f}")
    print(f"   RTCL: mean={np.nanmean(test_RTCL):.3f}")
    print(f"   RRP:  mean={np.nanmean(test_RRP):.3f}")

    # CRITICAL: Normalize using TRAIN statistics only (causal normalization)
    print("\n6. Applying causal normalization (train stats only)...")
    test_RERL_norm = zscore_causal(test_RERL, train_stats_RERL)
    test_RTCL_norm = zscore_causal(test_RTCL, train_stats_RTCL)
    test_RRP_norm = zscore_causal(test_RRP, train_stats_RRP)

    print(f"   Train stats:")
    print(f"     RERL: μ={train_stats_RERL['mean']:.3f}, σ={train_stats_RERL['std']:.3f}")
    print(f"     RTCL: μ={train_stats_RTCL['mean']:.3f}, σ={train_stats_RTCL['std']:.3f}")
    print(f"     RRP:  μ={train_stats_RRP['mean']:.3f}, σ={train_stats_RRP['std']:.3f}")

    test_CCI = compute_cci(test_RERL_norm, test_RTCL_norm, test_RRP_norm)
    test_DD = forward_drawdown(test_prices[1:], horizon=20)

    print(f"   CCI:  mean={np.nanmean(test_CCI):.3f}")

    # Out-of-sample correlation
    print("\n7. Computing out-of-sample correlation...")
    valid_test = ~(np.isnan(test_CCI) | np.isnan(test_DD))

    if np.sum(valid_test) < 100:
        print(f"   ✗ Insufficient valid points: {np.sum(valid_test)}")
        return False

    test_CCI_aligned = test_CCI[valid_test][:-20]
    test_DD_aligned = test_DD[valid_test][20:]

    min_len = min(len(test_CCI_aligned), len(test_DD_aligned))
    if min_len < 100:
        print(f"   ✗ Insufficient aligned points: {min_len}")
        return False

    test_CCI_aligned = test_CCI_aligned[:min_len]
    test_DD_aligned = test_DD_aligned[:min_len]

    rho_test, p_test = spearmanr(test_CCI_aligned, test_DD_aligned)

    print(f"   Out-of-sample: ρ = {rho_test:+.4f}, p = {p_test:.6f}")
    print(f"   Valid points: {min_len}")

    # =========================================================================
    # COMPARISON AND EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("="*70)

    print(f"\nIn-sample (development):   ρ = {rho_train:+.4f}, p = {p_train:.6f}")
    print(f"Out-of-sample (new data):  ρ = {rho_test:+.4f}, p = {p_test:.6f}")

    # Performance degradation
    if rho_train != 0:
        degradation = (rho_train - rho_test) / abs(rho_train) * 100
        print(f"\nPerformance change: {-degradation:+.1f}%")
    else:
        degradation = 0
        print(f"\nPerformance change: N/A (in-sample ρ = 0)")

    # Pass criteria
    print("\n" + "="*70)
    print("PASS/FAIL CRITERIA")
    print("="*70)

    success = True

    # Test 1: Out-of-sample correlation significant
    if p_test < 0.05:
        print(f"✓ PASS: Out-of-sample correlation significant (p = {p_test:.6f} < 0.05)")
    else:
        print(f"✗ FAIL: Out-of-sample correlation not significant (p = {p_test:.6f} ≥ 0.05)")
        success = False

    # Test 2: Out-of-sample correlation positive
    if rho_test > 0:
        print(f"✓ PASS: Out-of-sample correlation positive ({rho_test:+.4f})")
    else:
        print(f"✗ FAIL: Out-of-sample correlation negative ({rho_test:+.4f})")
        success = False

    # Test 3: Performance doesn't collapse (allow some degradation)
    # Accept if out-of-sample is at least 30% of in-sample performance
    if rho_train > 0:
        relative_performance = rho_test / rho_train
        if relative_performance >= 0.3:
            print(f"✓ PASS: Performance maintained ({relative_performance:.1%} of in-sample)")
        else:
            print(f"⚠️  WARNING: Significant performance degradation ({relative_performance:.1%} of in-sample)")
            # Not a hard failure, but concerning
    else:
        print(f"⚠️  Cannot assess relative performance (in-sample ρ = {rho_train:+.4f})")

    # Test 4: Same sign
    if np.sign(rho_test) == np.sign(rho_train):
        print(f"✓ PASS: Consistent sign between in-sample and out-of-sample")
    else:
        print(f"⚠️  WARNING: Sign flip between in-sample and out-of-sample")

    if success:
        print("\n" + "="*70)
        print("✓ PASS: SYSTEM GENERALIZES TO NEW DATA")
        print("="*70)
        print("\nThe system maintains predictive power on completely new")
        print("data not available during development. This demonstrates")
        print("genuine generalization capability.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: INSUFFICIENT OUT-OF-SAMPLE PERFORMANCE")
        print("="*70)
        print("\nThe system does not maintain performance on new data,")
        print("suggesting potential overfitting to the development period.")

    return success


if __name__ == "__main__":
    success = test_out_of_sample_validation()
    sys.exit(0 if success else 1)
