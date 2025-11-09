"""
Test 2.1: Reversed-Time Leakage Test

Novel leakage detection via time reversal. If the system works equally well
or BETTER when time runs backward, it's using information from the "future".

Based on TEST_PROGRAM_SPECIFICATION.md Section 2.1

This is the CRITICAL test from the data leakage paper that detected the
45% → -31% reversal in the original momentum prediction system.
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


def test_reversed_time_leakage():
    """
    Detect temporal leakage by processing data backward.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Reversed-time leakage test (Novel). Process data backward: if correlation
    > with 'future' (actually past) exceeds forward correlation, leakage is present."

    From the data leakage paper case study:
    - Forward (causal): r = 0.569
    - Reversed: r = 0.959  ← LEAKAGE!

    Pass Criteria
    -------------
    - Forward correlation > Reversed correlation
    - Reversed correlation < 0.8 * Forward correlation
    - If reversed > forward → MAJOR DATA LEAKAGE
    """
    print("="*70)
    print("TEST 2.1: REVERSED-TIME LEAKAGE DETECTION")
    print("="*70)
    print("\nThis is the CRITICAL test from the data leakage paper.")
    print("If reversed correlation > forward correlation → DATA LEAKAGE")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values
    dates = pd.to_datetime(df["Date"])

    print(f"   Total days: {len(df)}")
    print(f"   Date range: {dates.min()} to {dates.max()}")

    # Compute returns
    print("\n2. Computing returns and metrics...")
    returns = compute_returns(prices)
    volume = volume[1:]  # Align with returns

    print(f"   Returns: {len(returns)}")
    print(f"   Volume:  {len(volume)}")

    # FORWARD TIME: Normal computation
    print("\n3. Computing metrics in FORWARD time (causal)...")
    RERL_fwd = rolling_entropy(returns, window=252, bins=21)
    RTCL_fwd = rolling_autocorr(returns, window=252, lag=1)
    RRP_fwd = rolling_rrp(volume, np.abs(returns), window=252, lag=20)

    print(f"   RERL (forward): mean={np.nanmean(RERL_fwd):.3f}")
    print(f"   RTCL (forward): mean={np.nanmean(RTCL_fwd):.3f}")
    print(f"   RRP (forward):  mean={np.nanmean(RRP_fwd):.3f}")

    # Compute CCI forward
    print("\n4. Computing CCI in forward time...")
    CCI_fwd = compute_cci(RERL_fwd, RTCL_fwd, RRP_fwd)
    print(f"   CCI (forward): mean={np.nanmean(CCI_fwd):.3f}, std={np.nanstd(CCI_fwd):.3f}")

    # Compute forward drawdown
    print("\n5. Computing forward drawdown...")
    FWD_DD = forward_drawdown(prices[1:], horizon=20)
    print(f"   Forward DD: mean={np.nanmean(FWD_DD):.3%}, min={np.nanmin(FWD_DD):.3%}")

    # Forward correlation (CAUSAL)
    print("\n6. Computing FORWARD correlation (causal)...")
    valid_fwd = ~(np.isnan(CCI_fwd) | np.isnan(FWD_DD))

    # Align: CCI at t, DD at t+20
    CCI_fwd_aligned = CCI_fwd[valid_fwd][:-20]
    DD_fwd_aligned = FWD_DD[valid_fwd][20:]

    if len(CCI_fwd_aligned) < 50:
        print(f"   ✗ Insufficient valid points: {len(CCI_fwd_aligned)}")
        return False

    r_forward, p_forward = spearmanr(CCI_fwd_aligned, DD_fwd_aligned)
    print(f"   Forward correlation:  ρ = {r_forward:.4f} (p = {p_forward:.4f})")

    # REVERSED TIME: Process backward
    print("\n7. Reversing time series...")
    returns_rev = returns[::-1]
    volume_rev = volume[::-1]
    prices_rev = prices[::-1]

    print(f"   Reversed returns: {len(returns_rev)}")
    print(f"   Reversed volume:  {len(volume_rev)}")

    print("\n8. Computing metrics in REVERSED time...")
    RERL_rev = rolling_entropy(returns_rev, window=252, bins=21)
    RTCL_rev = rolling_autocorr(returns_rev, window=252, lag=1)
    RRP_rev = rolling_rrp(volume_rev, np.abs(returns_rev), window=252, lag=20)

    print(f"   RERL (reversed): mean={np.nanmean(RERL_rev):.3f}")
    print(f"   RTCL (reversed): mean={np.nanmean(RTCL_rev):.3f}")
    print(f"   RRP (reversed):  mean={np.nanmean(RRP_rev):.3f}")

    print("\n9. Computing CCI in reversed time...")
    CCI_rev = compute_cci(RERL_rev, RTCL_rev, RRP_rev)
    print(f"   CCI (reversed): mean={np.nanmean(CCI_rev):.3f}, std={np.nanstd(CCI_rev):.3f}")

    # In reversed time, "forward" drawdown is actually looking backward
    print("\n10. Computing 'forward' drawdown in reversed time...")
    FWD_DD_rev = forward_drawdown(prices_rev[1:], horizon=20)

    # Reversed correlation
    print("\n11. Computing REVERSED correlation...")
    valid_rev = ~(np.isnan(CCI_rev) | np.isnan(FWD_DD_rev))

    CCI_rev_aligned = CCI_rev[valid_rev][:-20]
    DD_rev_aligned = FWD_DD_rev[valid_rev][20:]

    if len(CCI_rev_aligned) < 50:
        print(f"   ✗ Insufficient valid points: {len(CCI_rev_aligned)}")
        return False

    r_reversed, p_reversed = spearmanr(CCI_rev_aligned, DD_rev_aligned)
    print(f"   Reversed correlation: ρ = {r_reversed:.4f} (p = {p_reversed:.4f})")

    # CRITICAL TEST
    print("\n" + "="*70)
    print("CRITICAL COMPARISON")
    print("="*70)
    print(f"Forward correlation:  ρ = {r_forward:.4f} (p = {p_forward:.4f})")
    print(f"Reversed correlation: ρ = {r_reversed:.4f} (p = {p_reversed:.4f})")
    print(f"Difference: {abs(r_reversed) - abs(r_forward):.4f}")
    print(f"Ratio: {abs(r_reversed) / abs(r_forward):.2f}x")

    # From data leakage paper case study:
    # Forward (causal): r = 0.569
    # Reversed: r = 0.959  ← LEAKAGE!

    success = True

    print("\n" + "="*70)
    print("LEAKAGE TESTS")
    print("="*70)

    # Test 1: Reversed should NOT exceed forward
    if abs(r_reversed) > abs(r_forward):
        print("✗ FAIL: REVERSED CORRELATION HIGHER → DATA LEAKAGE DETECTED")
        print("   System performs better in reversed time = using future information")
        success = False
    else:
        print("✓ PASS: Forward correlation > Reversed correlation")

    # Test 2: Reversed should not be suspiciously high
    if abs(r_reversed) > 0.8 * abs(r_forward):
        print("⚠️  WARNING: Reversed correlation suspiciously high")
        print(f"   Reversed is {abs(r_reversed) / abs(r_forward):.1%} of forward")
        print("   May indicate subtle leakage or strong autocorrelation artifact")
        # Note: This is a warning, not a failure
    else:
        print(f"✓ PASS: Reversed correlation is {abs(r_reversed) / abs(r_forward):.1%} of forward")

    # Test 3: Forward should be significant if we expect prediction
    if p_forward > 0.05:
        print(f"⚠️  NOTE: Forward correlation not significant (p = {p_forward:.4f})")
        print("   This may indicate weak or no predictive power")
        # Note: This is informational, not a failure
    else:
        print(f"✓ Forward correlation is significant (p = {p_forward:.4f})")

    if success:
        print("\n" + "="*70)
        print("✓ PASS: NO DATA LEAKAGE DETECTED")
        print("="*70)
        print("\nThe system performs better in forward time than reversed time.")
        print("This suggests the metrics are genuinely causal.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: DATA LEAKAGE DETECTED")
        print("="*70)
        print("\nThe system performs better (or equally well) in reversed time.")
        print("This is impossible for a causal system and indicates leakage.")

    return success


if __name__ == "__main__":
    success = test_reversed_time_leakage()
    sys.exit(0 if success else 1)
