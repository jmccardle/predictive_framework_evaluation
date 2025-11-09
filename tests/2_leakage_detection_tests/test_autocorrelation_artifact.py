"""
Test 2.4: Autocorrelation Artifact Test

Checks if CCI is just measuring autocorrelation rather than predicting
future changes. Strong autocorrelation can create illusion of prediction.

Based on TEST_PROGRAM_SPECIFICATION.md Section 2.4

From the data leakage paper:
"At h=5 steps, Duffing oscillator exhibits r=0.83 autocorrelation.
Simple persistence already captures 83% of predictable variance."
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from src.utils.data_loader import load_sp500, compute_returns
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def test_autocorrelation_artifact():
    """
    Check if CCI is just measuring autocorrelation rather than
    predicting future changes.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Autocorrelation artifacts. Chaotic systems exhibit strong short-term
    > autocorrelation. Methods can appear predictive by exploiting this
    > structure without genuine forecasting."

    Pass Criteria
    -------------
    - CCI should exceed autocorrelation by >20%
    - CCI correlation > 1.2 * autocorrelation
    - Otherwise system is just capturing autocorrelation
    """
    print("="*70)
    print("TEST 2.4: AUTOCORRELATION ARTIFACT TEST")
    print("="*70)
    print("\nChecking if CCI provides genuine prediction beyond autocorrelation...")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values

    returns = compute_returns(prices)
    volume = volume[1:]

    # Measure autocorrelation at various lags
    print("\n2. Measuring autocorrelation at various lags...")
    lags = [1, 5, 10, 20, 40, 60]
    autocorrs = []

    for lag in lags:
        if lag < len(returns):
            r, p = pearsonr(returns[:-lag], returns[lag:])
            autocorrs.append(r)
            print(f"   Lag {lag:2d}: r = {r:+.4f} (p = {p:.4f})")
        else:
            autocorrs.append(np.nan)

    # Compute CCI
    print("\n3. Computing CCI...")
    RERL = rolling_entropy(returns, window=252, bins=21)
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)
    CCI = compute_cci(RERL, RTCL, RRP)

    # Compute forward drawdown
    print("\n4. Computing forward drawdown...")
    FWD_DD = forward_drawdown(prices[1:], horizon=20)

    # CCI-Drawdown correlation
    print("\n5. Computing CCI-DrawDown correlation...")
    valid = ~(np.isnan(CCI) | np.isnan(FWD_DD))
    CCI_aligned = CCI[valid][:-20]
    DD_aligned = FWD_DD[valid][20:]

    if len(CCI_aligned) < 50:
        print(f"   ✗ Insufficient valid points: {len(CCI_aligned)}")
        return False

    rho_cci, p_cci = spearmanr(CCI_aligned, DD_aligned)
    print(f"   CCI-DrawDown correlation: ρ = {rho_cci:.4f} (p = {p_cci:.4f})")

    # Compare to autocorrelation at lag=20
    print("\n6. Comparing to simple autocorrelation...")
    lag_20_idx = lags.index(20)
    autocorr_20 = autocorrs[lag_20_idx]

    print(f"   Autocorrelation at lag 20: {autocorr_20:+.4f}")
    print(f"   CCI-DrawDown correlation: {rho_cci:+.4f}")
    print(f"   Absolute values:")
    print(f"     |Autocorr|: {abs(autocorr_20):.4f}")
    print(f"     |CCI-DD|:   {abs(rho_cci):.4f}")

    improvement = abs(rho_cci) - abs(autocorr_20)
    improvement_pct = (improvement / abs(autocorr_20)) * 100 if autocorr_20 != 0 else float('inf')

    print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # Plot autocorrelation vs lag
    print("\n7. Creating visualization...")
    plt.figure(figsize=(12, 5))

    # Autocorrelation plot
    plt.subplot(1, 2, 1)
    plt.plot(lags, autocorrs, 'o-', linewidth=2, markersize=8, color='blue',
             label='Autocorrelation')
    plt.axhline(abs(rho_cci), color='red', linestyle='--', linewidth=2,
                label=f'|CCI-DD| = {abs(rho_cci):.4f}')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('Lag (days)', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.title('Autocorrelation vs Lag', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Improvement bar chart
    plt.subplot(1, 2, 2)
    categories = ['Autocorrelation\n(lag 20)', 'CCI-DrawDown']
    values = [abs(autocorr_20), abs(rho_cci)]
    colors = ['skyblue', 'coral']
    bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('|Correlation|', fontsize=12)
    plt.title(f'CCI vs Autocorrelation (Improvement: {improvement_pct:+.1f}%)', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, max(values) * 1.2)

    plt.tight_layout()
    output_file = 'results/autocorrelation_artifact_test.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved plot to: {output_file}")
    plt.close()

    # CRITICAL TESTS
    print("\n" + "="*70)
    print("AUTOCORRELATION ARTIFACT TESTS")
    print("="*70)

    success = True

    # Test: CCI should exceed autocorrelation by >20%
    if abs(rho_cci) <= abs(autocorr_20) * 1.2:
        print(f"✗ FAIL: CCI provides minimal improvement over autocorrelation")
        print(f"   CCI: {abs(rho_cci):.4f}")
        print(f"   Autocorr: {abs(autocorr_20):.4f}")
        print(f"   Improvement: {improvement_pct:+.1f}% (expected >20%)")
        print("   System may be exploiting temporal structure without genuine prediction")
        success = False
    else:
        print(f"✓ PASS: CCI exceeds simple autocorrelation by {improvement_pct:+.1f}%")
        print(f"   CCI: {abs(rho_cci):.4f}")
        print(f"   Autocorr: {abs(autocorr_20):.4f}")
        print("   This suggests genuine predictive signal beyond autocorrelation")

    # Additional info: Check if autocorrelation is very strong
    if abs(autocorr_20) > 0.3:
        print(f"\n⚠️  NOTE: Strong autocorrelation detected ({abs(autocorr_20):.4f})")
        print("   High autocorrelation makes it easier to appear predictive")
        print("   Baseline comparison tests become even more important")

    # Summary
    if success:
        print("\n" + "="*70)
        print("✓ PASS: AUTOCORRELATION ARTIFACT TEST")
        print("="*70)
        print("\nCCI provides meaningful improvement beyond simple autocorrelation.")
        print("The system appears to capture genuine predictive signal,")
        print("not just temporal momentum.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: AUTOCORRELATION ARTIFACT TEST")
        print("="*70)
        print("\nCCI does not meaningfully exceed autocorrelation.")
        print("The system may be capturing temporal momentum rather than")
        print("genuinely predicting future crisis events.")

    return success


if __name__ == "__main__":
    success = test_autocorrelation_artifact()
    sys.exit(0 if success else 1)
