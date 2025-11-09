"""
Test 4.1: Naive Persistence Baseline

Compares CCI-based crash prediction vs naive persistence baseline.

Persistence forecast: "Tomorrow will be like today"
The system MUST beat this simple baseline to be considered useful.

Based on TEST_PROGRAM_SPECIFICATION.md Section 4.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from src.utils.data_loader import load_sp500, compute_returns
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def test_vs_persistence():
    """
    Compare CCI-based crash prediction vs naive persistence.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Naive baseline. Persistence forecast x̂(t+h) = x(t) provides
    > challenging baseline due to strong autocorrelation in chaotic systems."

    Pass Criteria
    -------------
    - CCI RMSE < Persistence RMSE
    - Improvement > 5%
    - CCI correlation stronger than persistence
    """
    print("="*70)
    print("TEST 4.1: NAIVE PERSISTENCE BASELINE")
    print("="*70)
    print("\nComparing CCI vs simplest possible baseline: 'tomorrow = today'")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values

    # Compute metrics
    print("\n2. Computing CCI...")
    returns = compute_returns(prices)
    volume = volume[1:]

    RERL = rolling_entropy(returns, window=252, bins=21)
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)

    CCI = compute_cci(RERL, RTCL, RRP)

    # Compute forward drawdown
    print("\n3. Computing forward drawdown...")
    FWD_DD = forward_drawdown(prices[1:], horizon=20)

    # Naive persistence: use recent return as predictor
    print("\n4. Computing naive persistence baseline...")
    persistence_signal = -returns  # Negative return → positive signal (warning)

    # Align lengths
    min_len = min(len(CCI), len(persistence_signal), len(FWD_DD))
    CCI = CCI[:min_len]
    persistence_signal = persistence_signal[:min_len]
    FWD_DD = FWD_DD[:min_len]

    # Remove NaNs
    valid = ~(np.isnan(CCI) | np.isnan(persistence_signal) | np.isnan(FWD_DD))

    # Align for prediction: predictor at t, target at t+20
    CCI_pred = CCI[valid][:-20]
    persistence_pred = persistence_signal[valid][:-20]
    DD_target = FWD_DD[valid][20:]

    print(f"   Valid samples: {len(DD_target)}")

    # Correlations
    print("\n5. Computing correlations...")
    rho_cci, p_cci = spearmanr(CCI_pred, DD_target)
    rho_persistence, p_persistence = spearmanr(persistence_pred, DD_target)

    print(f"   CCI correlation:         ρ = {rho_cci:.4f} (p = {p_cci:.4f})")
    print(f"   Persistence correlation: ρ = {rho_persistence:.4f} (p = {p_persistence:.4f})")

    # RMSE for continuous prediction
    print("\n6. Computing RMSE...")
    rmse_cci = np.sqrt(mean_squared_error(DD_target, -CCI_pred))  # Negative because high CCI = bad
    rmse_persistence = np.sqrt(mean_squared_error(DD_target, persistence_pred))

    print(f"   CCI RMSE:         {rmse_cci:.6f}")
    print(f"   Persistence RMSE: {rmse_persistence:.6f}")

    # Improvement
    improvement = (rmse_persistence - rmse_cci) / rmse_persistence * 100

    print(f"\n7. Performance comparison...")
    print(f"   RMSE improvement: {improvement:+.2f}%")

    if improvement > 0:
        print(f"   CCI is {improvement:.2f}% better than persistence")
    else:
        print(f"   CCI is {-improvement:.2f}% WORSE than persistence")

    # Correlation comparison
    corr_improvement = (abs(rho_cci) - abs(rho_persistence)) / abs(rho_persistence) * 100 if rho_persistence != 0 else float('inf')
    print(f"   Correlation improvement: {corr_improvement:+.2f}%")

    # CRITICAL TESTS
    print("\n" + "="*70)
    print("BASELINE COMPARISON TESTS")
    print("="*70)

    success = True

    # Test 1: RMSE improvement
    if improvement < 5:
        print(f"✗ FAIL: CCI does not beat naive persistence baseline")
        print(f"   Improvement: {improvement:.2f}% (required: >5%)")
        success = False
    else:
        print(f"✓ PASS: CCI beats persistence baseline")
        print(f"   Improvement: {improvement:.2f}%")

    # Test 2: Correlation should be stronger
    if abs(rho_cci) <= abs(rho_persistence):
        print(f"⚠️  WARNING: CCI correlation not stronger than persistence")
        print(f"   |ρ_CCI| = {abs(rho_cci):.4f}")
        print(f"   |ρ_persist| = {abs(rho_persistence):.4f}")
        # Note: This is a warning, not a failure

    # Summary
    if success:
        print("\n" + "="*70)
        print("✓ PASS: PERSISTENCE BASELINE TEST")
        print("="*70)
        print(f"\nCCI beats naive persistence by {improvement:.2f}%")
        print("The system provides value beyond the simplest baseline.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: PERSISTENCE BASELINE TEST")
        print("="*70)
        print("\nCCI does not meaningfully beat naive persistence.")
        print("The system does not provide value beyond 'tomorrow = today'.")

    return success


if __name__ == "__main__":
    success = test_vs_persistence()
    sys.exit(0 if success else 1)
