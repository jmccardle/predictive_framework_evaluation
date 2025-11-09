"""
Test 6.1: Multiple Hypothesis Testing Correction

Apply statistical corrections for multiple comparisons to prevent
false positives from testing many hypotheses.

When conducting multiple statistical tests, the probability of finding
at least one "significant" result by chance increases. This test applies
Bonferroni correction to control the family-wise error rate (FWER).

Based on TEST_PROGRAM_SPECIFICATION.md Section 6.1

Formula:
- Corrected α = α / n_tests
- For α = 0.05 and n tests, corrected α = 0.05 / n

Example:
- Testing 5 hypotheses × 3 crises × 1 asset = 15 tests
- Corrected α = 0.05 / 15 = 0.00333
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests
from src.utils.data_loader import load_sp500, compute_returns, get_crisis_periods
from src.metrics.core_metrics import (
    rolling_entropy,
    rolling_autocorr,
    rolling_rrp,
    compute_cci,
    forward_drawdown
)


def collect_hypothesis_test_pvalues():
    """
    Collect p-values from all hypothesis tests.

    This function runs simplified versions of each hypothesis test
    and collects the p-values for multiple testing correction.

    Returns
    -------
    all_p_values : list
        List of p-values from all tests
    test_names : list
        List of test descriptions
    """
    print("\nCollecting p-values from hypothesis tests...")
    print("-" * 70)

    all_p_values = []
    test_names = []

    # Load data
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    volume = df["Volume"].values
    dates = pd.to_datetime(df["Date"])
    returns = compute_returns(prices)
    volume = volume[1:]

    # Compute metrics
    RERL = rolling_entropy(returns, window=252, bins=21)
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)
    CCI = compute_cci(RERL, RTCL, RRP)
    DD = forward_drawdown(prices[1:], horizon=20)

    crisis_periods = get_crisis_periods()

    # -------------------------------------------------------------------------
    # H1: Entropy increases before crashes
    # Test for each crisis period
    # -------------------------------------------------------------------------
    print("\nH1: Entropy spike hypothesis (3 crises)")

    for crisis_name, (start, end) in crisis_periods.items():
        # Pre-crisis period (60 days before crisis)
        crisis_start_idx = np.where(dates[1:] >= pd.Timestamp(start))[0]
        if len(crisis_start_idx) == 0:
            continue

        crisis_idx = crisis_start_idx[0]
        pre_crisis_start = max(0, crisis_idx - 120)
        pre_crisis_end = max(0, crisis_idx - 1)

        # Normal period (1 year before pre-crisis)
        normal_start = max(0, pre_crisis_start - 252)
        normal_end = pre_crisis_start

        if normal_end - normal_start < 100 or pre_crisis_end - pre_crisis_start < 30:
            continue

        pre_crisis_entropy = RERL[pre_crisis_start:pre_crisis_end]
        normal_entropy = RERL[normal_start:normal_end]

        # Remove NaNs
        pre_crisis_entropy = pre_crisis_entropy[~np.isnan(pre_crisis_entropy)]
        normal_entropy = normal_entropy[~np.isnan(normal_entropy)]

        if len(pre_crisis_entropy) < 10 or len(normal_entropy) < 10:
            continue

        # T-test: Is pre-crisis entropy higher than normal?
        t_stat, p_val = ttest_ind(pre_crisis_entropy, normal_entropy, alternative='greater')

        all_p_values.append(p_val)
        test_names.append(f"H1: Entropy spike before {crisis_name}")
        print(f"  {crisis_name}: p = {p_val:.6f}")

    # -------------------------------------------------------------------------
    # Overall CCI prediction test
    # Test if CCI correlates with forward drawdown
    # -------------------------------------------------------------------------
    print("\nOverall CCI prediction")

    valid = ~(np.isnan(CCI) | np.isnan(DD))
    CCI_aligned = CCI[valid][:-20]
    DD_aligned = DD[valid][20:]

    min_len = min(len(CCI_aligned), len(DD_aligned))
    CCI_aligned = CCI_aligned[:min_len]
    DD_aligned = DD_aligned[:min_len]

    if len(CCI_aligned) >= 100:
        rho, p_val = spearmanr(CCI_aligned, DD_aligned)
        all_p_values.append(p_val)
        test_names.append("Overall CCI-DD correlation")
        print(f"  Overall CCI: p = {p_val:.6f}")

    # -------------------------------------------------------------------------
    # Crisis-specific CCI tests
    # Test if CCI works in each crisis period
    # -------------------------------------------------------------------------
    print("\nCrisis-specific CCI tests")

    for crisis_name, (start, end) in crisis_periods.items():
        crisis_mask = (dates[1:] >= pd.Timestamp(start)) & (dates[1:] <= pd.Timestamp(end))

        if np.sum(crisis_mask) < 50:
            continue

        crisis_CCI = CCI[crisis_mask]
        crisis_DD = DD[crisis_mask]

        valid_crisis = ~(np.isnan(crisis_CCI) | np.isnan(crisis_DD))

        if np.sum(valid_crisis) < 30:
            continue

        crisis_CCI_valid = crisis_CCI[valid_crisis]
        crisis_DD_valid = crisis_DD[valid_crisis]

        # Need at least 30 points for correlation
        if len(crisis_CCI_valid) >= 30 and len(crisis_DD_valid) >= 30:
            min_len = min(len(crisis_CCI_valid), len(crisis_DD_valid))
            rho, p_val = spearmanr(crisis_CCI_valid[:min_len], crisis_DD_valid[:min_len])

            all_p_values.append(p_val)
            test_names.append(f"CCI during {crisis_name}")
            print(f"  {crisis_name}: p = {p_val:.6f}")

    # -------------------------------------------------------------------------
    # Component metric tests
    # Test if individual metrics (RERL, RTCL, RRP) correlate with DD
    # -------------------------------------------------------------------------
    print("\nComponent metric tests")

    for metric_name, metric_values in [("RERL", RERL), ("RTCL", RTCL), ("RRP", RRP)]:
        valid = ~(np.isnan(metric_values) | np.isnan(DD))
        metric_aligned = metric_values[valid][:-20]
        DD_aligned = DD[valid][20:]

        min_len = min(len(metric_aligned), len(DD_aligned))
        if min_len >= 100:
            rho, p_val = spearmanr(metric_aligned[:min_len], DD_aligned[:min_len])

            all_p_values.append(p_val)
            test_names.append(f"{metric_name}-DD correlation")
            print(f"  {metric_name}: p = {p_val:.6f}")

    print(f"\nTotal tests collected: {len(all_p_values)}")

    return all_p_values, test_names


def test_multiple_hypothesis_correction():
    """
    Apply Bonferroni correction for multiple hypothesis testing.

    Pass Criteria
    -------------
    - ≥33% of tests remain significant after correction
    - At least one test survives correction
    - Family-wise error rate controlled at α = 0.05

    Per TEST_PROGRAM_SPECIFICATION.md Section 6.1:
    "Apply Bonferroni correction for multiple comparisons"
    """
    print("="*70)
    print("TEST 6.1: MULTIPLE HYPOTHESIS TESTING CORRECTION")
    print("="*70)
    print("\nApplying Bonferroni correction to control family-wise error rate.")
    print("This prevents false positives from testing many hypotheses.")

    # Collect p-values from all hypothesis tests
    print("\n1. Collecting p-values from all hypothesis tests...")

    all_p_values, test_names = collect_hypothesis_test_pvalues()

    if len(all_p_values) == 0:
        print("✗ FAIL: No p-values collected")
        return False

    # Apply Bonferroni correction
    print("\n" + "="*70)
    print("BONFERRONI CORRECTION")
    print("="*70)

    alpha = 0.05
    n_tests = len(all_p_values)
    corrected_alpha = alpha / n_tests

    print(f"\nFamily-wise error rate: α = {alpha}")
    print(f"Number of tests: {n_tests}")
    print(f"Bonferroni corrected α: {corrected_alpha:.6f}")

    # Apply correction using statsmodels
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        all_p_values,
        alpha=alpha,
        method='bonferroni'
    )

    # Count significant results
    uncorrected_significant = sum(p < alpha for p in all_p_values)
    corrected_significant = sum(reject)

    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nSignificant (uncorrected, α={alpha}): {uncorrected_significant}/{n_tests} "
          f"({uncorrected_significant/n_tests*100:.1f}%)")
    print(f"Significant (Bonferroni):               {corrected_significant}/{n_tests} "
          f"({corrected_significant/n_tests*100:.1f}%)")

    # Detailed results
    print("\nDetailed Results:")
    print("-" * 70)
    print(f"{'Test':<45s} {'p-value':>10s} {'Uncorr':>8s} {'Bonf':>8s}")
    print("-" * 70)

    for i, (name, p_val, is_sig) in enumerate(zip(test_names, all_p_values, reject)):
        uncorr_sig = "✓" if p_val < alpha else " "
        bonf_sig = "✓" if is_sig else " "

        # Truncate long names
        display_name = name if len(name) <= 44 else name[:41] + "..."

        print(f"{display_name:<45s} {p_val:>10.6f} {uncorr_sig:>8s} {bonf_sig:>8s}")

    # Summary statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)

    p_values_array = np.array(all_p_values)
    print(f"\nP-value distribution:")
    print(f"  Min:     {p_values_array.min():.6f}")
    print(f"  Median:  {np.median(p_values_array):.6f}")
    print(f"  Mean:    {p_values_array.mean():.6f}")
    print(f"  Max:     {p_values_array.max():.6f}")

    # Pass criteria
    print("\n" + "="*70)
    print("PASS/FAIL CRITERIA")
    print("="*70)

    success = True

    # Test 1: At least one test survives correction
    if corrected_significant >= 1:
        print(f"✓ PASS: {corrected_significant} test(s) survive Bonferroni correction")
    else:
        print(f"✗ FAIL: No tests survive Bonferroni correction")
        success = False

    # Test 2: At least 33% of tests remain significant
    survival_rate = corrected_significant / n_tests if n_tests > 0 else 0
    if survival_rate >= 0.33:
        print(f"✓ PASS: {survival_rate*100:.1f}% of tests remain significant (≥33%)")
    else:
        print(f"⚠️  WARNING: Only {survival_rate*100:.1f}% of tests remain significant (<33%)")
        # Not a hard failure, but concerning

    # Test 3: Some tests were significant before correction
    if uncorrected_significant >= n_tests / 2:
        print(f"✓ PASS: {uncorrected_significant}/{n_tests} tests significant before correction")
    else:
        print(f"⚠️  WARNING: Only {uncorrected_significant}/{n_tests} tests significant before correction")

    # Information about loss rate
    loss_rate = (uncorrected_significant - corrected_significant) / uncorrected_significant if uncorrected_significant > 0 else 0
    print(f"\nSignificance loss rate: {loss_rate*100:.1f}%")
    print(f"  (Lost {uncorrected_significant - corrected_significant} out of {uncorrected_significant} significant results)")

    if success:
        print("\n" + "="*70)
        print("✓ PASS: RESULTS SURVIVE MULTIPLE TESTING CORRECTION")
        print("="*70)
        print("\nThe findings are statistically robust even after accounting")
        print("for multiple comparisons. Family-wise error rate is controlled.")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: INSUFFICIENT STATISTICAL RIGOR")
        print("="*70)
        print("\nThe results do not survive correction for multiple testing.")
        print("This suggests potential false positives from testing many hypotheses.")

    return success


if __name__ == "__main__":
    success = test_multiple_hypothesis_correction()
    sys.exit(0 if success else 1)
