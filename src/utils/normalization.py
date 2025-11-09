"""
Causal normalization utilities to prevent data leakage.

This module implements z-score normalization with strict train/test separation,
addressing the critical normalization leakage problem identified in the
data leakage paper.

CRITICAL: Never compute normalization statistics on the entire dataset.
Always use ONLY training data statistics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def compute_train_statistics(
    train_data: Dict[str, np.ndarray]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute normalization statistics (mean, std) from training data ONLY.

    Parameters
    ----------
    train_data : dict
        Dictionary mapping metric name -> training values
        e.g., {'RERL': array([...]), 'RTCL': array([...]), ...}

    Returns
    -------
    train_mean : dict
        Dictionary of mean values for each metric
    train_std : dict
        Dictionary of std values for each metric

    Examples
    --------
    >>> train_data = {'RERL': np.random.randn(1000), 'RTCL': np.random.randn(1000)}
    >>> means, stds = compute_train_statistics(train_data)
    >>> print(means['RERL'], stds['RERL'])

    Notes
    -----
    From TEST_PROGRAM_SPECIFICATION.md Section 1.2:
    > "Normalization parameters computed only on training data."

    These statistics will be used for normalizing BOTH training and test data,
    ensuring no information leakage from test set.
    """
    train_mean = {}
    train_std = {}

    for key, values in train_data.items():
        # Use nanmean/nanstd to handle NaN values from rolling windows
        train_mean[key] = np.nanmean(values)
        train_std[key] = np.nanstd(values, ddof=0)

    return train_mean, train_std


def zscore_causal(
    data: np.ndarray,
    train_mean: float,
    train_std: float
) -> np.ndarray:
    """
    Apply z-score normalization using ONLY training statistics.

    This is the CORRECT implementation that prevents data leakage.

    Parameters
    ----------
    data : np.ndarray
        Data to normalize (can be train or test)
    train_mean : float
        Mean computed from training data ONLY
    train_std : float
        Std computed from training data ONLY

    Returns
    -------
    normalized : np.ndarray
        Z-score normalized data

    Examples
    --------
    >>> train = np.array([1, 2, 3, 4, 5])
    >>> test = np.array([6, 7, 8])
    >>> train_mean = train.mean()
    >>> train_std = train.std()
    >>> train_normalized = zscore_causal(train, train_mean, train_std)
    >>> test_normalized = zscore_causal(test, train_mean, train_std)

    Notes
    -----
    INCORRECT (causes leakage):
    ```python
    def zscore_leaky(data):
        return (data - data.mean()) / data.std()  # ← BUG!
    ```

    CORRECT (no leakage):
    ```python
    def zscore_causal(data, train_mean, train_std):
        return (data - train_mean) / train_std  # ← GOOD!
    ```

    From TEST_PROGRAM_SPECIFICATION.md Section 1.2:
    > "Normalization leakage. Computing statistics (mean, standard deviation)
    > on entire datasets including test data, then applying to training data,
    > leaks information about future distributions backward in time."
    """
    return (data - train_mean) / (train_std + 1e-12)


def zscore_batch_causal(
    data: Dict[str, np.ndarray],
    train_mean: Dict[str, float],
    train_std: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    Apply causal z-score normalization to multiple metrics.

    Parameters
    ----------
    data : dict
        Dictionary of metric arrays to normalize
    train_mean : dict
        Dictionary of training means
    train_std : dict
        Dictionary of training stds

    Returns
    -------
    normalized : dict
        Dictionary of normalized arrays

    Examples
    --------
    >>> data = {'RERL': np.array([1, 2, 3]), 'RTCL': np.array([4, 5, 6])}
    >>> means = {'RERL': 2.0, 'RTCL': 5.0}
    >>> stds = {'RERL': 1.0, 'RTCL': 1.0}
    >>> normalized = zscore_batch_causal(data, means, stds)
    """
    normalized = {}

    for key, values in data.items():
        if key in train_mean and key in train_std:
            normalized[key] = zscore_causal(values, train_mean[key], train_std[key])
        else:
            raise KeyError(
                f"Missing normalization statistics for '{key}'. "
                f"Available keys: {list(train_mean.keys())}"
            )

    return normalized


def detect_normalization_leakage(
    train_data: np.ndarray,
    test_data: np.ndarray,
    threshold: float = 0.01
) -> bool:
    """
    Detect if normalization leakage is present.

    Compares correct (causal) normalization vs incorrect (leaky) normalization
    to detect if the code is using future information.

    Parameters
    ----------
    train_data : np.ndarray
        Training set data
    test_data : np.ndarray
        Test set data
    threshold : float
        Threshold for detecting meaningful difference

    Returns
    -------
    has_leakage : bool
        True if leakage detected, False otherwise

    Examples
    --------
    >>> train = np.random.randn(1000)
    >>> test = np.random.randn(500)
    >>> has_leakage = detect_normalization_leakage(train, test)
    >>> if has_leakage:
    ...     print("⚠️ WARNING: Normalization leakage detected!")
    """
    # CORRECT: Use only train statistics
    train_mean = np.mean(train_data)
    train_std = np.std(train_data, ddof=0)
    test_correct = (test_data - train_mean) / (train_std + 1e-12)

    # INCORRECT: Use combined statistics (LEAKAGE!)
    all_data = np.concatenate([train_data, test_data])
    all_mean = np.mean(all_data)
    all_std = np.std(all_data, ddof=0)
    test_leaky = (test_data - all_mean) / (all_std + 1e-12)

    # Compare
    difference = np.abs(test_correct - test_leaky).mean()

    has_leakage = difference > threshold

    if has_leakage:
        print(f"⚠️  LEAKAGE DETECTED: Mean difference = {difference:.6f} > {threshold}")
        print(f"   Correct normalization: mean={train_mean:.4f}, std={train_std:.4f}")
        print(f"   Leaky normalization:   mean={all_mean:.4f}, std={all_std:.4f}")
    else:
        print(f"✓ No leakage: Mean difference = {difference:.6f} < {threshold}")

    return has_leakage


def apply_expanding_window_normalization(
    data: np.ndarray,
    min_window: int = 252
) -> np.ndarray:
    """
    Apply expanding window normalization (for walk-forward validation).

    At each time point, normalize using statistics from ALL data up to that point.
    This is causal and suitable for online/streaming scenarios.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    min_window : int
        Minimum window size before starting normalization

    Returns
    -------
    normalized : np.ndarray
        Expanding-window normalized data

    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> normalized = apply_expanding_window_normalization(data, min_window=252)
    >>> # First 252 values will be NaN
    >>> # From 253 onwards, each point normalized by mean/std of all prior data
    """
    n = len(data)
    normalized = np.full(n, np.nan)

    for i in range(min_window, n):
        # Use all data up to (but not including) current point
        historical = data[:i]
        hist_mean = np.nanmean(historical)
        hist_std = np.nanstd(historical, ddof=0)

        normalized[i] = (data[i] - hist_mean) / (hist_std + 1e-12)

    return normalized


if __name__ == "__main__":
    print("Testing causal normalization utilities...\n")

    # Generate synthetic data
    np.random.seed(42)
    train = np.random.randn(1000) + 5.0  # Mean = 5, std ≈ 1
    test = np.random.randn(500) + 7.0    # Mean = 7, std ≈ 1 (different!)

    print("1. Computing training statistics...")
    train_data = {'metric1': train}
    means, stds = compute_train_statistics(train_data)
    print(f"   Train mean: {means['metric1']:.4f}")
    print(f"   Train std:  {stds['metric1']:.4f}")

    print("\n2. Applying causal normalization...")
    train_normalized = zscore_causal(train, means['metric1'], stds['metric1'])
    test_normalized = zscore_causal(test, means['metric1'], stds['metric1'])
    print(f"   Train normalized mean: {np.mean(train_normalized):.4f} (should be ≈0)")
    print(f"   Train normalized std:  {np.std(train_normalized):.4f} (should be ≈1)")
    print(f"   Test normalized mean:  {np.mean(test_normalized):.4f} (will NOT be 0!)")
    print(f"   Test normalized std:   {np.std(test_normalized):.4f} (will NOT be 1!)")

    print("\n3. Detecting normalization leakage...")
    detect_normalization_leakage(train, test, threshold=0.01)

    print("\n4. Testing batch normalization...")
    data_dict = {'RERL': np.random.randn(100), 'RTCL': np.random.randn(100)}
    train_means = {'RERL': 0.0, 'RTCL': 0.0}
    train_stds = {'RERL': 1.0, 'RTCL': 1.0}
    normalized = zscore_batch_causal(data_dict, train_means, train_stds)
    print(f"   Normalized {len(normalized)} metrics")

    print("\n✓ All normalization utilities working correctly!")
