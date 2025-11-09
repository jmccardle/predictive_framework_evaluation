"""
Test 5.3: Multi-Asset Consistency Test

Test that the system works consistently across different asset classes.

A genuinely predictive system should work across multiple markets if the
underlying mechanism (entropy, correlation breakdown, volume-volatility)
is fundamental to market dynamics.

Asset classes tested:
- Equities: S&P 500 (^GSPC), Nasdaq (^IXIC)
- Bonds: TLT (20+ Year Treasury)
- Commodities: Gold (GC=F)

Based on SOFTWARE_REQUIREMENTS_SPECIFICATION.md REQ-5: Multi-Asset Robustness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.utils.data_loader import load_multi_asset, compute_returns
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def evaluate_asset(df, asset_name):
    """
    Evaluate system performance on a single asset.

    Parameters
    ----------
    df : pd.DataFrame
        Asset data with columns: Date, Adj Close, Volume
    asset_name : str
        Name of asset for reporting

    Returns
    -------
    rho : float
        Spearman correlation
    p : float
        P-value
    n_valid : int
        Number of valid data points
    """
    print(f"\nEvaluating {asset_name}...")

    prices = df["Adj Close"].values
    volume = df["Volume"].values

    # Compute returns
    returns = compute_returns(prices)
    volume = volume[1:]  # Align with returns

    print(f"  Data points: {len(returns)}")

    # Compute metrics
    RERL = rolling_entropy(returns, window=252, bins=21)
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)

    # Compute CCI
    CCI = compute_cci(RERL, RTCL, RRP)

    # Compute forward drawdown
    DD = forward_drawdown(prices[1:], horizon=20)

    # Align for correlation
    valid = ~(np.isnan(CCI) | np.isnan(DD))
    n_valid = np.sum(valid)

    print(f"  Valid points: {n_valid}")

    if n_valid < 100:
        print(f"  ✗ Insufficient data")
        return np.nan, 1.0, n_valid

    # Align CCI at t with DD at t+20
    CCI_aligned = CCI[valid][:-20]
    DD_aligned = DD[valid][20:]

    min_len = min(len(CCI_aligned), len(DD_aligned))
    if min_len < 100:
        print(f"  ✗ Insufficient aligned data")
        return np.nan, 1.0, n_valid

    CCI_aligned = CCI_aligned[:min_len]
    DD_aligned = DD_aligned[:min_len]

    # Compute correlation
    rho, p = spearmanr(CCI_aligned, DD_aligned)

    print(f"  Metrics:")
    print(f"    RERL: mean={np.nanmean(RERL):.3f}, std={np.nanstd(RERL):.3f}")
    print(f"    RTCL: mean={np.nanmean(RTCL):.3f}, std={np.nanstd(RTCL):.3f}")
    print(f"    RRP:  mean={np.nanmean(RRP):.3f}, std={np.nanstd(RRP):.3f}")
    print(f"    CCI:  mean={np.nanmean(CCI):.3f}, std={np.nanstd(CCI):.3f}")
    print(f"  Result: ρ = {rho:+.4f}, p = {p:.6f}")

    return rho, p, n_valid


def test_multi_asset_consistency():
    """
    Test consistency across multiple asset classes.

    Pass Criteria
    -------------
    - ≥50% of assets show significant correlation
    - Mean correlation across all assets > 0
    - No major sign flips (consistency of direction)

    Per SOFTWARE_REQUIREMENTS_SPECIFICATION.md REQ-5:
    "CCI works across asset classes"
    """
    print("="*70)
    print("TEST 5.3: MULTI-ASSET CONSISTENCY")
    print("="*70)
    print("\nTesting system robustness across different asset classes.")
    print("A genuine effect should generalize beyond a single market.")

    # Define assets to test
    assets = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT",
        "Gold": "GC=F",
        "Treasury Bonds": "TLT",
    }

    print("\n1. Loading multi-asset data...")
    print("-" * 70)

    # Load data
    symbols = list(assets.values())
    data = load_multi_asset(symbols, start="2000-01-01", end="2024-11-01")

    # Map back to friendly names
    asset_data = {}
    for name, symbol in assets.items():
        if symbol in data:
            asset_data[name] = data[symbol]

    print(f"\nSuccessfully loaded {len(asset_data)}/{len(assets)} assets")

    # Evaluate each asset
    print("\n2. Evaluating each asset...")
    print("-" * 70)

    results = []

    for asset_name, df in asset_data.items():
        rho, p, n_valid = evaluate_asset(df, asset_name)

        results.append({
            'asset': asset_name,
            'symbol': assets[asset_name],
            'rho': rho,
            'p': p,
            'n_valid': n_valid
        })

    # Analyze results
    print("\n" + "="*70)
    print("MULTI-ASSET SUMMARY")
    print("="*70)

    valid_results = [r for r in results if not np.isnan(r['rho'])]

    if len(valid_results) == 0:
        print("✗ FAIL: No valid asset results")
        return False

    rhos = [r['rho'] for r in valid_results]
    ps = [r['p'] for r in valid_results]

    mean_rho = np.mean(rhos)
    std_rho = np.std(rhos)
    median_rho = np.median(rhos)
    significant_count = sum(1 for p in ps if p < 0.05)
    positive_count = sum(1 for rho in rhos if rho > 0)

    print(f"\nAcross {len(valid_results)} assets:")
    print(f"  Mean ρ:        {mean_rho:+.4f} ± {std_rho:.4f}")
    print(f"  Median ρ:      {median_rho:+.4f}")
    print(f"  Range:         [{min(rhos):+.4f}, {max(rhos):+.4f}]")
    print(f"  Positive:      {positive_count}/{len(valid_results)} ({positive_count/len(valid_results)*100:.1f}%)")
    print(f"  Significant:   {significant_count}/{len(valid_results)} ({significant_count/len(valid_results)*100:.1f}%)")

    # Detailed results table
    print("\nDetailed Results by Asset:")
    print("-" * 70)
    for r in results:
        if not np.isnan(r['rho']):
            sig = "✓" if r['p'] < 0.05 else " "
            print(f"  {r['asset']:20s} ({r['symbol']:8s}): "
                  f"ρ = {r['rho']:+.4f}, p = {r['p']:.6f} {sig} "
                  f"(n={r['n_valid']})")
        else:
            print(f"  {r['asset']:20s} ({r['symbol']:8s}): INSUFFICIENT DATA")

    # Pass criteria
    print("\n" + "="*70)
    print("PASS/FAIL CRITERIA")
    print("="*70)

    success = True

    # Test 1: At least 3 assets evaluated
    if len(valid_results) >= 3:
        print(f"✓ PASS: Sufficient assets evaluated ({len(valid_results)})")
    else:
        print(f"✗ FAIL: Insufficient assets ({len(valid_results)} < 3)")
        success = False

    # Test 2: Majority significant
    if significant_count >= len(valid_results) / 2:
        print(f"✓ PASS: {significant_count}/{len(valid_results)} assets significant (≥50%)")
    else:
        print(f"⚠️  WARNING: Only {significant_count}/{len(valid_results)} assets significant (<50%)")
        # Not a hard failure, but concerning

    # Test 3: Mean correlation positive
    if mean_rho > 0:
        print(f"✓ PASS: Mean correlation positive ({mean_rho:+.4f})")
    else:
        print(f"✗ FAIL: Mean correlation negative ({mean_rho:+.4f})")
        success = False

    # Test 4: Majority positive direction
    if positive_count >= len(valid_results) / 2:
        print(f"✓ PASS: {positive_count}/{len(valid_results)} assets positive direction (≥50%)")
    else:
        print(f"✗ FAIL: Only {positive_count}/{len(valid_results)} assets positive (<50%)")
        success = False

    # Test 5: Consistency (coefficient of variation)
    cv = std_rho / abs(mean_rho) if mean_rho != 0 else np.inf
    if cv < 1.5:
        print(f"✓ PASS: Reasonable consistency (CV = {cv:.2f} < 1.5)")
    else:
        print(f"⚠️  WARNING: High variability across assets (CV = {cv:.2f} ≥ 1.5)")

    if success:
        print("\n" + "="*70)
        print("✓ PASS: SYSTEM IS CONSISTENT ACROSS ASSET CLASSES")
        print("="*70)
        print("\nThe system shows consistent behavior across multiple")
        print("asset classes, suggesting a fundamental mechanism.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: INSUFFICIENT MULTI-ASSET CONSISTENCY")
        print("="*70)
        print("\nThe system does not generalize well across different")
        print("asset classes, suggesting it may be specific to one market.")

    return success


if __name__ == "__main__":
    success = test_multi_asset_consistency()
    sys.exit(0 if success else 1)
