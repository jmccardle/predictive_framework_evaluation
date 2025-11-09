"""
Test 3.1: H1 Entropy Spike Hypothesis

Tests if market entropy (RERL) statistically increases before crashes.

Hypothesis H1: Market entropy increases 20-60 days before crashes.

Based on TEST_PROGRAM_SPECIFICATION.md Section 3.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from src.utils.data_loader import load_sp500, compute_returns, get_crisis_periods
from src.metrics.core_metrics import rolling_entropy


def test_h1_entropy_spike():
    """
    Test if RERL statistically increases before crashes.

    Reference from SOFTWARE_REQUIREMENTS_SPECIFICATION.md:
    > H1: Entropy Spike Hypothesis
    > Market entropy increases 20-60 days before crashes.

    Pass Criteria
    -------------
    - ≥2 of 3 crises show significant entropy increase (p < 0.05)
    - Effect size (Cohen's d) > 0.3 (small to medium)
    - Pre-crisis entropy > normal period entropy
    """
    print("="*70)
    print("TEST 3.1: H1 - ENTROPY SPIKE HYPOTHESIS")
    print("="*70)
    print("\nHypothesis: Market entropy increases 20-60 days before crashes")

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01", end="2024-11-01")
    prices = df["Adj Close"].values
    dates = pd.to_datetime(df["Date"])
    returns = compute_returns(prices)

    # Compute RERL
    print("\n2. Computing rolling entropy (RERL)...")
    RERL = rolling_entropy(returns, window=252, bins=21)
    RERL_series = pd.Series(RERL, index=dates[1:])

    print(f"   Mean entropy: {np.nanmean(RERL):.3f} bits")
    print(f"   Std entropy:  {np.nanstd(RERL):.3f} bits")

    # Define crisis periods
    CRISES = get_crisis_periods()

    print(f"\n3. Testing {len(CRISES)} crisis periods...")
    results = []
    fig, axes = plt.subplots(len(CRISES), 1, figsize=(14, 4*len(CRISES)))
    if len(CRISES) == 1:
        axes = [axes]

    for idx, (crisis_name, (start, end)) in enumerate(CRISES.items()):
        print(f"\n{'='*60}")
        print(f"CRISIS: {crisis_name}")
        print(f"{'='*60}")

        # Pre-crisis period (60 days before)
        pre_start = pd.to_datetime(start) - pd.Timedelta(days=60)
        pre_end = pd.to_datetime(start)

        pre_crisis_mask = (RERL_series.index >= pre_start) & (RERL_series.index < pre_end)
        pre_crisis_entropy = RERL_series[pre_crisis_mask].dropna()

        # Normal period (2 years before pre-crisis, excluding 60 days before crisis)
        normal_start = pre_start - pd.Timedelta(days=730)
        normal_end = pre_start - pd.Timedelta(days=60)

        normal_mask = (RERL_series.index >= normal_start) & (RERL_series.index < normal_end)
        normal_entropy = RERL_series[normal_mask].dropna()

        print(f"Normal period:     {normal_start.date()} to {normal_end.date()}")
        print(f"Pre-crisis period: {pre_start.date()} to {pre_end.date()}")
        print(f"Crisis period:     {start} to {end}")

        if len(pre_crisis_entropy) > 10 and len(normal_entropy) > 10:
            # T-test
            t_stat, p_val = ttest_ind(pre_crisis_entropy, normal_entropy)

            # Statistics
            normal_mean = normal_entropy.mean()
            normal_std = normal_entropy.std()
            pre_mean = pre_crisis_entropy.mean()
            pre_std = pre_crisis_entropy.std()
            mean_diff = pre_mean - normal_mean

            # Cohen's d (effect size)
            pooled_std = np.sqrt(((len(normal_entropy)-1) * normal_std**2 +
                                  (len(pre_crisis_entropy)-1) * pre_std**2) /
                                 (len(normal_entropy) + len(pre_crisis_entropy) - 2))
            cohens_d = mean_diff / pooled_std

            print(f"\nStatistics:")
            print(f"  Normal entropy:     {normal_mean:.4f} ± {normal_std:.4f} (n={len(normal_entropy)})")
            print(f"  Pre-crisis entropy: {pre_mean:.4f} ± {pre_std:.4f} (n={len(pre_crisis_entropy)})")
            print(f"  Difference:         {mean_diff:+.4f}")
            print(f"  Effect size (d):    {cohens_d:.4f}")
            print(f"  t-statistic:        {t_stat:.4f}")
            print(f"  p-value:            {p_val:.6f}")

            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect_desc = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_desc = "small"
            elif abs(cohens_d) < 0.8:
                effect_desc = "medium"
            else:
                effect_desc = "large"

            print(f"  Effect:             {effect_desc}")

            # Test result
            passed = (mean_diff > 0) and (p_val < 0.05) and (cohens_d > 0.3)

            if passed:
                print(f"  Result:             ✓ PASS")
            else:
                print(f"  Result:             ✗ FAIL")
                if mean_diff <= 0:
                    print(f"    Reason: Entropy decreased (not increased)")
                if p_val >= 0.05:
                    print(f"    Reason: Not statistically significant (p={p_val:.4f})")
                if cohens_d <= 0.3:
                    print(f"    Reason: Effect size too small (d={cohens_d:.4f})")

            results.append({
                'crisis': crisis_name,
                'mean_diff': mean_diff,
                'p_value': p_val,
                'effect_size': cohens_d,
                'passed': passed
            })

            # Visualization
            ax = axes[idx]

            # Get full time series for context
            plot_start = normal_start
            plot_end = pd.to_datetime(end) + pd.Timedelta(days=30)
            plot_mask = (RERL_series.index >= plot_start) & (RERL_series.index <= plot_end)
            plot_data = RERL_series[plot_mask]

            ax.plot(plot_data.index, plot_data.values, linewidth=1.5, color='blue', alpha=0.7)

            # Highlight regions
            ax.axvspan(normal_start, normal_end, alpha=0.2, color='green', label='Normal period')
            ax.axvspan(pre_start, pre_end, alpha=0.3, color='orange', label='Pre-crisis (test)')
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), alpha=0.3, color='red', label='Crisis')

            # Horizontal lines for means
            ax.axhline(normal_mean, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax.axhline(pre_mean, color='orange', linestyle='--', linewidth=2, alpha=0.7)

            ax.set_ylabel('Entropy (bits)', fontsize=11)
            ax.set_title(f"{crisis_name}: {'✓ PASS' if passed else '✗ FAIL'} "
                        f"(Δ={mean_diff:+.3f}, d={cohens_d:.2f}, p={p_val:.4f})",
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        else:
            print(f"  ✗ Insufficient data")
            results.append({
                'crisis': crisis_name,
                'mean_diff': np.nan,
                'p_value': np.nan,
                'effect_size': np.nan,
                'passed': False
            })

    # Save figure
    plt.tight_layout()
    output_file = 'results/h1_entropy_spike_test.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\nSaved visualization to: {output_file}")
    plt.close()

    # Overall result
    passed_count = sum(r['passed'] for r in results)
    total_count = len(results)

    print("\n" + "="*70)
    print("H1 HYPOTHESIS TEST RESULTS")
    print("="*70)
    print(f"Passed: {passed_count}/{total_count} crises show significant entropy increase")

    for r in results:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['crisis']:15s}  Δ={r['mean_diff']:+.4f}  d={r['effect_size']:.3f}  p={r['p_value']:.4f}")

    success = passed_count >= 2

    if success:
        print("\n" + "="*70)
        print("✓ PASS: H1 ENTROPY SPIKE HYPOTHESIS SUPPORTED")
        print("="*70)
        print(f"\n{passed_count} out of {total_count} crises show significant entropy increase")
        print("before the crisis period (criterion: ≥2 required).")
    else:
        print("\n" + "="*70)
        print("✗ FAIL: H1 ENTROPY SPIKE HYPOTHESIS NOT SUPPORTED")
        print("="*70)
        print(f"\nOnly {passed_count} out of {total_count} crises show significant entropy increase")
        print("(criterion: ≥2 required).")

    return success


if __name__ == "__main__":
    success = test_h1_entropy_spike()
    sys.exit(0 if success else 1)
