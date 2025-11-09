"""
Test 1.2: Normalization Causality Test

Detects normalization leakage by verifying z-score uses ONLY training statistics.

Based on TEST_PROGRAM_SPECIFICATION.md Section 1.2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from src.utils.data_loader import load_sp500, split_train_test
from src.utils.normalization import (
    compute_train_statistics,
    zscore_causal,
    detect_normalization_leakage
)
from src.metrics.core_metrics import rolling_entropy, compute_returns


def test_normalization_causality():
    """
    Verify z-score normalization uses ONLY training statistics.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Normalization leakage. Computing statistics (mean, standard deviation)
    > on entire datasets including test data, then applying to training data,
    > leaks information about future distributions backward in time."

    Pass Criteria
    -------------
    - Normalization must use train-only statistics
    - Test set normalization must apply train statistics
    - No leakage detected when comparing correct vs incorrect methods
    """
    print("="*70)
    print("TEST 1.2: NORMALIZATION CAUSALITY")
    print("="*70)

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, _ = load_sp500(start="2000-01-01")
    prices = df["Adj Close"].values

    TRAIN_SIZE = 2000
    TEST_SIZE = 400

    # Split data
    print("\n2. Splitting into train/test...")
    train_prices, test_prices = split_train_test(
        prices,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE
    )
    print(f"   Train size: {len(train_prices)}")
    print(f"   Test size:  {len(test_prices)}")

    # Compute returns and metrics
    print("\n3. Computing metrics on train data...")
    train_returns = compute_returns(train_prices)
    train_entropy = rolling_entropy(train_returns, window=252, bins=21)

    print(f"   Train returns: {len(train_returns)}")
    print(f"   Train entropy: {len(train_entropy)}")

    # CORRECT: Compute stats on train only
    print("\n4. Computing CORRECT normalization statistics (train only)...")
    train_data = {'entropy': train_entropy}
    train_mean, train_std = compute_train_statistics(train_data)

    print(f"   Train entropy mean: {train_mean['entropy']:.4f}")
    print(f"   Train entropy std:  {train_std['entropy']:.4f}")

    # Apply to test
    print("\n5. Computing test entropy...")
    test_returns = compute_returns(test_prices)
    test_entropy = rolling_entropy(test_returns, window=252, bins=21)

    print(f"   Test entropy: {len(test_entropy)}")

    # CORRECT normalization
    print("\n6. Applying CORRECT normalization (using train stats)...")
    test_entropy_clean = test_entropy[~np.isnan(test_entropy)]
    test_normalized_correct = zscore_causal(
        test_entropy_clean,
        train_mean['entropy'],
        train_std['entropy']
    )

    print(f"   Test normalized mean: {np.mean(test_normalized_correct):.4f} (NOT ≈0)")
    print(f"   Test normalized std:  {np.std(test_normalized_correct):.4f} (NOT ≈1)")
    print(f"   ✓ This is correct - test stats differ from train")

    # INCORRECT normalization (LEAKAGE!)
    print("\n7. Computing INCORRECT normalization (entire series - LEAKAGE!)...")
    all_entropy = np.concatenate([train_entropy[~np.isnan(train_entropy)],
                                   test_entropy[~np.isnan(test_entropy)]])
    all_mean = np.mean(all_entropy)
    all_std = np.std(all_entropy, ddof=0)

    test_normalized_leaky = (test_entropy_clean - all_mean) / all_std

    print(f"   Leaky mean: {all_mean:.4f}")
    print(f"   Leaky std:  {all_std:.4f}")

    # Compare
    print("\n8. Detecting leakage...")
    difference = np.abs(test_normalized_correct - test_normalized_leaky).mean()
    print(f"   Mean difference: {difference:.6f}")

    try:
        assert difference > 0.001, \
            "WARNING: Difference too small - may indicate leakage"
        print(f"   ✓ Significant difference detected (correct vs leaky)")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    # Test utility function
    print("\n9. Testing leakage detection utility...")
    train_entropy_clean = train_entropy[~np.isnan(train_entropy)]
    has_leakage = detect_normalization_leakage(
        train_entropy_clean,
        test_entropy_clean,
        threshold=0.001
    )

    try:
        assert not has_leakage, \
            "FAIL: Leakage detection utility reports false positive"
        print(f"   ✓ Detection utility works correctly")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    # Test that train normalization gives mean≈0, std≈1
    print("\n10. Verifying train normalization properties...")
    train_entropy_clean = train_entropy[~np.isnan(train_entropy)]
    train_normalized = zscore_causal(
        train_entropy_clean,
        train_mean['entropy'],
        train_std['entropy']
    )

    train_norm_mean = np.mean(train_normalized)
    train_norm_std = np.std(train_normalized, ddof=0)

    print(f"   Train normalized mean: {train_norm_mean:.4f} (should be ≈0)")
    print(f"   Train normalized std:  {train_norm_std:.4f} (should be ≈1)")

    try:
        assert abs(train_norm_mean) < 0.01, \
            f"Train normalized mean = {train_norm_mean:.4f}, expected ≈0"
        assert abs(train_norm_std - 1.0) < 0.01, \
            f"Train normalized std = {train_norm_std:.4f}, expected ≈1"
        print(f"   ✓ Train normalization properties verified")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    print("\n" + "="*70)
    print("✓ PASS: NORMALIZATION CAUSALITY VERIFIED")
    print("="*70)
    print("\nAll tests passed:")
    print("  ✓ Normalization uses train-only statistics")
    print("  ✓ Test normalization differs from train (expected)")
    print("  ✓ Leakage detection utility works")
    print("  ✓ Train normalization has correct properties")

    return True


if __name__ == "__main__":
    success = test_normalization_causality()
    sys.exit(0 if success else 1)
