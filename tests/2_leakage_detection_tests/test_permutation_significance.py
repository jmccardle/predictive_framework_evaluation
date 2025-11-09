"""
Test 2.3: Permutation Significance Test

Verifies that CCI-drawdown correlation exceeds chance levels through
rigorous permutation testing with 10,000 random shuffles.

Based on TEST_PROGRAM_SPECIFICATION.md Section 2.3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from src.utils.data_loader import load_sp500, compute_returns
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def test_permutation_significance(n_permutations=10000):
    """
    Test if CCI-drawdown correlation exceeds chance levels.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Permutation testing. 10,000 random permutations establish null
    > distribution for correlation differences, computing exact p-values."

    Pass Criteria
    -------------
    - Permutation p-value < 0.01 (significant at 1% level)
    - Observed correlation in tail of null distribution
    - Less than 1% of permutations exceed observed correlation
    """
    print("="*70)
    print("TEST 2.3: PERMUTATION SIGNIFICANCE TEST")
    print("="*70)
    print(f"\nRunning {n_permutations:,} permutations to establish null distribution...")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values

    # Compute metrics
    print("\n2. Computing metrics...")
    returns = compute_returns(prices)
    volume = volume[1:]

    RERL = rolling_entropy(returns, window=252, bins=21)
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)

    CCI = compute_cci(RERL, RTCL, RRP)
    FWD_DD = forward_drawdown(prices[1:], horizon=20)

    # Compute observed correlation
    print("\n3. Computing observed correlation...")
    valid = ~(np.isnan(CCI) | np.isnan(FWD_DD))
    CCI_valid = CCI[valid][:-20]
    DD_valid = FWD_DD[valid][20:]

    if len(CCI_valid) < 50:
        print(f"   ✗ Insufficient valid points: {len(CCI_valid)}")
        return False

    observed_rho, _ = spearmanr(CCI_valid, DD_valid)
    print(f"   Observed ρ: {observed_rho:.4f}")
    print(f"   Sample size: {len(CCI_valid)}")

    # Generate null distribution by permuting CCI labels
    print(f"\n4. Generating null distribution ({n_permutations:,} permutations)...")
    print("   This may take 1-2 minutes...")

    rng = np.random.default_rng(42)
    null_distribution = []

    # Progress indicator
    progress_points = [int(n_permutations * p) for p in [0.25, 0.5, 0.75, 1.0]]

    for i in range(n_permutations):
        # Permute CCI values (break temporal structure)
        CCI_permuted = rng.permutation(CCI_valid)
        rho_null, _ = spearmanr(CCI_permuted, DD_valid)
        null_distribution.append(rho_null)

        # Progress
        if (i + 1) in progress_points:
            pct = ((i + 1) / n_permutations) * 100
            print(f"   Progress: {pct:.0f}% ({i+1:,}/{n_permutations:,})")

    null_distribution = np.array(null_distribution)

    # Compute p-value (two-tailed)
    print("\n5. Computing permutation p-value...")
    p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_rho)) + 1) / (n_permutations + 1)

    print(f"   Observed ρ: {observed_rho:.4f}")
    print(f"   Null mean: {null_distribution.mean():.4f}")
    print(f"   Null std: {null_distribution.std():.4f}")
    print(f"   Null range: [{null_distribution.min():.4f}, {null_distribution.max():.4f}]")
    print(f"   Permutation p-value: {p_value:.6f}")

    # Z-score of observed correlation
    z_score = (observed_rho - null_distribution.mean()) / null_distribution.std()
    print(f"   Z-score: {z_score:.2f}")

    # Plot null distribution
    print("\n6. Creating visualization...")
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(null_distribution, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(observed_rho, color='red', linestyle='--', linewidth=2,
                label=f'Observed ρ = {observed_rho:.4f}')
    plt.axvline(-abs(observed_rho), color='red', linestyle='--', linewidth=2, alpha=0.5)
    plt.xlabel('Spearman ρ', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Permutation Test: p = {p_value:.6f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Q-Q plot
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(null_distribution, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Null Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'results/permutation_test_result.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved plot to: {output_file}")
    plt.close()

    # CRITICAL TESTS
    print("\n" + "="*70)
    print("PERMUTATION TESTS")
    print("="*70)

    success = True

    # Test 1: p-value should be < 0.01
    if p_value > 0.01:
        print(f"✗ FAIL: p = {p_value:.4f} > 0.01 (not significant)")
        print("   Correlation does not exceed chance levels")
        success = False
    else:
        print(f"✓ PASS: p = {p_value:.6f} < 0.01 (significant)")

    # Test 2: Z-score should be substantial
    if abs(z_score) < 2.58:  # 99% confidence
        print(f"⚠️  WARNING: Z-score = {z_score:.2f} is modest (< 2.58)")
        print("   Effect may be weak even if statistically significant")
        # Note: This is a warning, not a failure
    else:
        print(f"✓ Strong effect: Z-score = {z_score:.2f} (> 2.58)")

    # Test 3: Observed should be in tail
    percentile = np.sum(null_distribution <= observed_rho) / len(null_distribution) * 100
    print(f"\nObserved correlation at {percentile:.1f} percentile of null distribution")

    if percentile < 95 and percentile > 5:
        print(f"⚠️  WARNING: Observed not in tail (5-95 percentile)")
        # Note: This is informational
    else:
        print(f"✓ Observed in tail of distribution")

    # Summary
    if success:
        print("\n" + "="*70)
        print("✓ PASS: PERMUTATION SIGNIFICANCE TEST")
        print("="*70)
        print(f"\nThe observed correlation (ρ = {observed_rho:.4f}) is statistically")
        print(f"significant (p = {p_value:.6f}) and exceeds chance levels.")
        print(f"\nOut of {n_permutations:,} random permutations, only {int(p_value * n_permutations)}")
        print(f"showed equal or stronger correlation by chance.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: PERMUTATION SIGNIFICANCE TEST")
        print("="*70)
        print(f"\nThe observed correlation (ρ = {observed_rho:.4f}) does not")
        print(f"significantly exceed chance levels (p = {p_value:.4f}).")

    return success


if __name__ == "__main__":
    success = test_permutation_significance(n_permutations=10000)
    sys.exit(0 if success else 1)
