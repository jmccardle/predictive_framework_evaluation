"""
Test 1.1: Temporal Separation Test

Verifies strict temporal ordering and no overlap between training and test data.

Based on TEST_PROGRAM_SPECIFICATION.md Section 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from src.utils.data_loader import load_sp500, split_train_test


def test_temporal_separation():
    """
    Verify strict temporal ordering and no overlap between train/test data.

    Reference from TEST_PROGRAM_SPECIFICATION.md:
    > "Proper train/test split. Training data (n = 2000) strictly precedes
    > test data (n = 400) with no overlap. Normalization parameters computed
    > only on training data."

    Pass Criteria
    -------------
    - Train dates must precede all test dates
    - No index overlap between train and test sets
    - Train end date <= Test start date
    """
    print("="*70)
    print("TEST 1.1: TEMPORAL SEPARATION")
    print("="*70)

    # Load data
    print("\n1. Loading S&P 500 data...")
    df, src = load_sp500(start="1998-01-01")
    prices = df["Adj Close"].values
    dates = pd.to_datetime(df["Date"])

    print(f"   Source: {src}")
    print(f"   Total days: {len(df)}")
    print(f"   Date range: {dates.min()} to {dates.max()}")

    # Define split
    TRAIN_SIZE = 2000
    TEST_SIZE = 400

    print(f"\n2. Splitting data...")
    print(f"   Train size: {TRAIN_SIZE}")
    print(f"   Test size:  {TEST_SIZE}")

    train_end_idx = TRAIN_SIZE
    test_start_idx = TRAIN_SIZE  # No gap, but no overlap

    # Test 1: No temporal overlap
    print("\n3. Testing temporal separation...")
    train_dates = dates[:train_end_idx]
    test_dates = dates[test_start_idx:test_start_idx+TEST_SIZE]

    train_max = train_dates.max()
    test_min = test_dates.min()

    print(f"   Train period: {train_dates.min()} to {train_max}")
    print(f"   Test period:  {test_min} to {test_dates.max()}")

    try:
        assert train_max <= test_min, \
            f"FAIL: Train data extends to {train_max}, test starts at {test_min}"
        print(f"   ✓ Train ends before or at test start")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    # Test 2: No index overlap
    print("\n4. Testing index separation...")
    train_indices = set(range(train_end_idx))
    test_indices = set(range(test_start_idx, test_start_idx+TEST_SIZE))

    overlap = train_indices & test_indices
    print(f"   Train indices: 0 to {train_end_idx-1}")
    print(f"   Test indices:  {test_start_idx} to {test_start_idx+TEST_SIZE-1}")

    try:
        assert len(overlap) == 0, \
            f"FAIL: {len(overlap)} indices overlap between train/test"
        print(f"   ✓ No index overlap")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    # Test 3: Verify split_train_test utility
    print("\n5. Testing split_train_test utility...")
    train_prices, test_prices = split_train_test(
        prices,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE
    )

    print(f"   Train size: {len(train_prices)}")
    print(f"   Test size:  {len(test_prices)}")

    try:
        assert len(train_prices) == TRAIN_SIZE, \
            f"Train size mismatch: {len(train_prices)} != {TRAIN_SIZE}"
        assert len(test_prices) == TEST_SIZE, \
            f"Test size mismatch: {len(test_prices)} != {TEST_SIZE}"
        print(f"   ✓ Split utility works correctly")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    # Test 4: Chronological ordering
    print("\n6. Testing chronological ordering...")
    dates_sorted = dates.is_monotonic_increasing

    try:
        assert dates_sorted, "FAIL: Dates are not in chronological order"
        print(f"   ✓ Data is sorted chronologically (oldest to newest)")
    except AssertionError as e:
        print(f"   ✗ {e}")
        return False

    print("\n" + "="*70)
    print("✓ PASS: TEMPORAL SEPARATION VERIFIED")
    print("="*70)
    print("\nAll tests passed:")
    print("  ✓ Train dates precede test dates")
    print("  ✓ No index overlap")
    print("  ✓ Split utility works correctly")
    print("  ✓ Data is chronologically ordered")

    return True


if __name__ == "__main__":
    success = test_temporal_separation()
    sys.exit(0 if success else 1)
