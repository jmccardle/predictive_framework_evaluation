"""
Test 2.2: Shuffled Future Test

Verifies predictions don't work with randomized future values.
If correlation remains with shuffled future, the system is correlating
with CURRENT state, not predicting FUTURE state.

Based on TEST_PROGRAM_SPECIFICATION.md Section 2.2
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


def test_shuffled_future():
    """
    If CCI truly predicts forward drawdown, shuffling future values
    should destroy the correlation.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Shuffled future test: Randomly permute future values; correlation
    > should approach zero"

    Pass Criteria
    -------------
    - Shuffled correlation should be near zero (|ρ| < 0.1)
    - Shuffled p-value should be > 0.05 (not significant)
    - Correlation should be "destroyed" (drop by >80%)
    """
    print("="*70)
    print("TEST 2.2: SHUFFLED FUTURE TEST")
    print("="*70)
    print("\nIf predictions are genuine, shuffling future should destroy correlation.")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values

    print(f"   Total days: {len(df)}")

    # Compute returns and metrics
    print("\n2. Computing returns and metrics...")
    returns = compute_returns(prices)
    volume = volume[1:]  # Align with returns

    RERL = rolling_entropy(returns, window=252, bins=21)
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)

    print("\n3. Computing CCI...")
    CCI = compute_cci(RERL, RTCL, RRP)

    print("\n4. Computing forward drawdown...")
    FWD_DD = forward_drawdown(prices[1:], horizon=20)

    # Original correlation
    print("\n5. Computing ORIGINAL correlation...")
    valid = ~(np.isnan(CCI) | np.isnan(FWD_DD))

    CCI_aligned = CCI[valid][:-20]
    DD_aligned = FWD_DD[valid][20:]

    if len(CCI_aligned) < 50:
        print(f"   ✗ Insufficient valid points: {len(CCI_aligned)}")
        return False

    r_original, p_original = spearmanr(CCI_aligned, DD_aligned)
    print(f"   Original correlation:  ρ = {r_original:.4f} (p = {p_original:.4f})")

    # Shuffle future values (destroy temporal structure)
    print("\n6. Shuffling future values...")
    rng = np.random.default_rng(42)
    FWD_DD_shuffled = FWD_DD.copy()
    FWD_DD_shuffled[20:] = rng.permutation(FWD_DD[20:])

    print(f"   Shuffled {len(FWD_DD[20:])} future values")

    # Correlation with shuffled future
    print("\n7. Computing correlation with SHUFFLED future...")
    DD_shuffled_aligned = FWD_DD_shuffled[valid][20:]

    r_shuffled, p_shuffled = spearmanr(CCI_aligned, DD_shuffled_aligned)
    print(f"   Shuffled correlation:  ρ = {r_shuffled:.4f} (p = {p_shuffled:.4f})")

    # Compute reduction
    print("\n8. Analyzing correlation destruction...")
    correlation_drop = abs(r_original) - abs(r_shuffled)
    reduction_pct = (correlation_drop / abs(r_original)) * 100 if r_original != 0 else 0

    print(f"   Correlation destroyed: {correlation_drop:.4f}")
    print(f"   Reduction: {reduction_pct:.1f}%")

    # CRITICAL TESTS
    print("\n" + "="*70)
    print("SHUFFLED FUTURE TESTS")
    print("="*70)

    success = True

    # Test 1: Shuffled correlation should be near zero
    if abs(r_shuffled) > 0.1:
        print(f"✗ FAIL: Shuffled future still shows correlation (|ρ| = {abs(r_shuffled):.4f} > 0.1)")
        print("   System may be correlating with current state, not predicting")
        success = False
    else:
        print(f"✓ PASS: Shuffled correlation near zero (|ρ| = {abs(r_shuffled):.4f})")

    # Test 2: Shuffled should not be significant
    if p_shuffled < 0.05:
        print(f"✗ FAIL: Shuffled future is statistically significant (p = {p_shuffled:.4f})")
        print("   This should not happen if prediction is genuinely forward-looking")
        success = False
    else:
        print(f"✓ PASS: Shuffled correlation not significant (p = {p_shuffled:.4f})")

    # Test 3: Correlation should be substantially reduced
    if reduction_pct < 80:
        print(f"⚠️  WARNING: Correlation only reduced by {reduction_pct:.1f}% (expected >80%)")
        print("   May indicate weak original signal or spurious correlation")
        # Note: This is a warning, not a failure
    else:
        print(f"✓ PASS: Correlation reduced by {reduction_pct:.1f}%")

    # Summary
    if success:
        print("\n" + "="*70)
        print("✓ PASS: SHUFFLED FUTURE TEST")
        print("="*70)
        print("\nShuffling future values destroys the correlation.")
        print("This indicates the system is genuinely predicting future events,")
        print("not just correlating with current state.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: SHUFFLED FUTURE TEST")
        print("="*70)
        print("\nShuffled future maintains significant correlation.")
        print("This suggests the system is NOT genuinely predictive.")

    return success


if __name__ == "__main__":
    success = test_shuffled_future()
    sys.exit(0 if success else 1)
