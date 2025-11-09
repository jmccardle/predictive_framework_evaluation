"""
Core metrics for financial crisis prediction system.

This module implements the three primary metrics and composite index:
1. RERL - Rolling Entropy (market disorder)
2. RTCL - Rolling Temporal Correlation (autocorrelation patterns)
3. RRP - Returns-Reversal Precedence (volume-volatility relationship)
4. CCI - Composite Crisis Index

Based on TEST_PROGRAM_SPECIFICATION.md and SOFTWARE_REQUIREMENTS_SPECIFICATION.md
"""

import numpy as np
import pandas as pd
from typing import Optional


def rolling_entropy(
    returns: np.ndarray,
    window: int = 252,
    bins: int = 21
) -> np.ndarray:
    """
    Compute rolling Shannon entropy of return distribution (RERL).

    Measures the "disorder" or unpredictability in market returns.
    Hypothesis H1: Entropy increases 20-60 days before crashes.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns (typically log returns)
    window : int, default=252
        Rolling window size (252 trading days ≈ 1 year)
    bins : int, default=21
        Number of histogram bins for entropy calculation

    Returns
    -------
    entropy : np.ndarray
        Rolling entropy values (same length as returns, first `window` values are NaN)

    Notes
    -----
    Shannon entropy formula: H = -Σ(p_k * log₂(p_k))
    - Units: bits
    - Range: [0, log₂(bins)] = [0, 4.4] for 21 bins
    - Higher entropy = more unpredictable/chaotic returns

    Mathematical definition from SOFTWARE_REQUIREMENTS_SPECIFICATION.md:
        For window of returns r[i-252:i]:
        - Bin into 21-bin histogram
        - Compute probabilities: p_k = count_k / total
        - RERL(i) = -Σ(p_k * log₂(p_k))

    Examples
    --------
    >>> returns = np.random.randn(1000) * 0.01
    >>> entropy = rolling_entropy(returns, window=252, bins=21)
    >>> print(f"Mean entropy: {np.nanmean(entropy):.3f} bits")
    """
    n = len(returns)
    entropy = np.full(n, np.nan)

    for i in range(window, n):
        # Extract window of returns
        window_data = returns[i - window:i]

        # Create histogram
        counts, _ = np.histogram(window_data, bins=bins)

        # Compute probabilities (with small epsilon to avoid log(0))
        probs = (counts + 1e-12) / (counts.sum() + bins * 1e-12)

        # Shannon entropy in bits
        entropy[i] = -np.sum(probs * np.log2(probs + 1e-12))

    return entropy


def rolling_autocorr(
    returns: np.ndarray,
    window: int = 252,
    lag: int = 1
) -> np.ndarray:
    """
    Compute rolling autocorrelation (RTCL).

    Measures temporal correlation patterns in returns.
    Hypothesis H2: Autocorrelation patterns change before crises.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    window : int, default=252
        Rolling window size
    lag : int, default=1
        Autocorrelation lag (1 = correlation with previous day)

    Returns
    -------
    autocorr : np.ndarray
        Rolling autocorrelation values (Pearson ρ)

    Notes
    -----
    Mathematical definition from SOFTWARE_REQUIREMENTS_SPECIFICATION.md:
        RTCL(i) = corr(r[i-252:i-1], r[i-251:i])

    - Units: Pearson correlation coefficient
    - Range: [-1, 1]
    - Positive = trend persistence
    - Negative = mean reversion
    - Near zero = random walk

    Examples
    --------
    >>> returns = np.random.randn(1000) * 0.01
    >>> autocorr = rolling_autocorr(returns, window=252, lag=1)
    >>> print(f"Mean autocorrelation: {np.nanmean(autocorr):.3f}")
    """
    n = len(returns)
    autocorr = np.full(n, np.nan)

    for i in range(window + lag, n):
        # Extract window
        window_data = returns[i - window:i]

        # Split into lagged segments
        x = window_data[:-lag]  # r[t-lag]
        y = window_data[lag:]   # r[t]

        # Compute Pearson correlation
        if len(x) > 0 and len(y) > 0:
            corr_matrix = np.corrcoef(x, y)
            autocorr[i] = corr_matrix[0, 1]

    return autocorr


def rolling_rrp(
    volume: np.ndarray,
    returns_abs: np.ndarray,
    window: int = 252,
    lag: int = 20
) -> np.ndarray:
    """
    Compute Returns-Reversal Precedence (RRP).

    Measures whether volume surges precede volatility spikes.
    Hypothesis H3: Volume surges precede volatility with 20-day lag.

    Parameters
    ----------
    volume : np.ndarray
        Trading volume
    returns_abs : np.ndarray
        Absolute returns (proxy for volatility)
    window : int, default=252
        Rolling window size
    lag : int, default=20
        Lag between volume and volatility (20 days per specification)

    Returns
    -------
    rrp : np.ndarray
        RRP correlation values

    Notes
    -----
    Mathematical definition from SOFTWARE_REQUIREMENTS_SPECIFICATION.md:
        RRP(i) = corr(volume[i-272:i-20], |returns|[i-252:i])

    Logic: Does volume at (t-20) predict volatility at (t)?
    - Positive RRP = volume predicts volatility
    - Negative RRP = inverse relationship
    - Near zero = no predictive relationship

    Examples
    --------
    >>> volume = np.random.rand(1000) * 1e9
    >>> returns_abs = np.abs(np.random.randn(1000) * 0.01)
    >>> rrp = rolling_rrp(volume, returns_abs, window=252, lag=20)
    >>> print(f"Mean RRP: {np.nanmean(rrp):.3f}")
    """
    n = len(volume)
    rrp = np.full(n, np.nan)

    # Ensure both arrays are aligned
    min_len = min(len(volume), len(returns_abs))
    volume = volume[:min_len]
    returns_abs = returns_abs[:min_len]

    for i in range(window + lag, n):
        # Volume from (i-window-lag) to (i-lag)
        # This is "past" volume that might predict "current" volatility
        v_seg = volume[i - window - lag:i - lag]

        # Current absolute returns (volatility)
        a_seg = returns_abs[i - window:i]

        # Correlation
        if len(v_seg) > 0 and len(a_seg) > 0 and len(v_seg) == len(a_seg):
            corr_matrix = np.corrcoef(v_seg, a_seg)
            rrp[i] = corr_matrix[0, 1]

    return rrp


def forward_drawdown(
    prices: np.ndarray,
    horizon: int = 20
) -> np.ndarray:
    """
    Compute forward maximum drawdown.

    For each time point, compute the maximum percentage decline
    over the next `horizon` days.

    Parameters
    ----------
    prices : np.ndarray
        Price series
    horizon : int, default=20
        Forward-looking window (20 days per specification)

    Returns
    -------
    drawdowns : np.ndarray
        Forward drawdown percentages (negative values indicate declines)

    Notes
    -----
    Formula: DD(t) = (min(P[t:t+h]) - P[t]) / P[t]

    - Negative values indicate price declines
    - Used as target variable for CCI prediction
    - DD <= -0.10 (i.e., -10%) is typically considered a "crash"

    CRITICAL: This is a FORWARD-LOOKING metric and must NEVER be used
    as a feature. It can only be used as a target variable for validation.

    Examples
    --------
    >>> prices = np.array([100, 95, 90, 92, 88])
    >>> dd = forward_drawdown(prices, horizon=2)
    >>> # At t=0: min(95, 90) = 90, DD = (90-100)/100 = -0.10
    """
    n = len(prices)
    drawdowns = np.full(n, np.nan)

    for i in range(n - horizon):
        # Forward window
        future_prices = prices[i:i + horizon + 1]

        # Current price
        current_price = prices[i]

        # Minimum future price
        min_future = np.min(future_prices[1:])  # Exclude current price

        # Drawdown percentage
        drawdowns[i] = (min_future - current_price) / current_price

    return drawdowns


def compute_cci(
    RERL: np.ndarray,
    RTCL: np.ndarray,
    RRP: np.ndarray,
    Lambda: Optional[np.ndarray] = None,
    train_mean: Optional[dict] = None,
    train_std: Optional[dict] = None
) -> np.ndarray:
    """
    Compute Composite Crisis Index (CCI).

    Combines the three metrics (and optional Lambda) into a single crisis predictor.

    Parameters
    ----------
    RERL : np.ndarray
        Rolling entropy values
    RTCL : np.ndarray
        Rolling autocorrelation values
    RRP : np.ndarray
        Returns-Reversal Precedence values
    Lambda : np.ndarray, optional
        Adaptive memory parameter (if None, excluded from CCI)
    train_mean : dict, optional
        Pre-computed mean values from training set for causal normalization
    train_std : dict, optional
        Pre-computed std values from training set for causal normalization

    Returns
    -------
    CCI : np.ndarray
        Composite Crisis Index

    Notes
    -----
    Mathematical definition from SOFTWARE_REQUIREMENTS_SPECIFICATION.md:

        FCI = z(-RERL) + z(RTCL)
        ΔRERL, ΔRTCL, ΔRRP = first differences
        PTE = sqrt((ΔRERL² + ΔRTCL² + ΔRRP²) / 3)
        CCI = z(λ) + z(PTE) - z(FCI)

    Interpretation:
    - High CCI = High crisis risk
    - Low CCI = Low crisis risk

    CRITICAL: Z-score normalization must use ONLY training set statistics
    to avoid data leakage. If train_mean and train_std are provided,
    they will be used. Otherwise, normalization is computed on the full
    series (ONLY acceptable for training data).

    Examples
    --------
    >>> RERL = rolling_entropy(returns, window=252, bins=21)
    >>> RTCL = rolling_autocorr(returns, window=252, lag=1)
    >>> RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)
    >>> CCI = compute_cci(RERL, RTCL, RRP)
    """
    # Helper function for z-score normalization
    def zscore(x, mean=None, std=None):
        x = pd.Series(x)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std(ddof=0)
        return (x - mean) / (std + 1e-12)

    # Financial Condition Index (FCI)
    # Lower entropy and higher autocorr = stable conditions
    if train_mean is not None and train_std is not None:
        z_rerl = zscore(-RERL, train_mean.get('RERL', None), train_std.get('RERL', None))
        z_rtcl = zscore(RTCL, train_mean.get('RTCL', None), train_std.get('RTCL', None))
    else:
        z_rerl = zscore(-RERL)
        z_rtcl = zscore(RTCL)

    FCI = z_rerl + z_rtcl

    # Phase Transition Entropy (PTE)
    # Magnitude of changes across all three metrics
    dRERL = np.diff(RERL, prepend=np.nan)
    dRTCL = np.diff(RTCL, prepend=np.nan)
    dRRP = np.diff(RRP, prepend=np.nan)

    PTE = np.sqrt((dRERL**2 + dRTCL**2 + dRRP**2) / 3)

    # Composite Crisis Index
    if train_mean is not None and train_std is not None:
        z_pte = zscore(PTE, train_mean.get('PTE', None), train_std.get('PTE', None))
        z_fci = zscore(FCI, train_mean.get('FCI', None), train_std.get('FCI', None))
    else:
        z_pte = zscore(PTE)
        z_fci = zscore(FCI)

    if Lambda is not None:
        if train_mean is not None and train_std is not None:
            z_lambda = zscore(Lambda, train_mean.get('Lambda', None), train_std.get('Lambda', None))
        else:
            z_lambda = zscore(Lambda)
        CCI = z_lambda + z_pte - z_fci
    else:
        CCI = z_pte - z_fci

    return CCI.values


if __name__ == "__main__":
    # Example usage and validation
    print("Testing core metrics...")

    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n) * 0.01  # 1% daily volatility
    volume = np.random.rand(n) * 1e9

    # Compute metrics
    print("\n1. Computing RERL (rolling entropy)...")
    RERL = rolling_entropy(returns, window=252, bins=21)
    print(f"   Mean: {np.nanmean(RERL):.3f} bits")
    print(f"   Range: [{np.nanmin(RERL):.3f}, {np.nanmax(RERL):.3f}]")

    print("\n2. Computing RTCL (rolling autocorrelation)...")
    RTCL = rolling_autocorr(returns, window=252, lag=1)
    print(f"   Mean: {np.nanmean(RTCL):.3f}")
    print(f"   Range: [{np.nanmin(RTCL):.3f}, {np.nanmax(RTCL):.3f}]")

    print("\n3. Computing RRP (Returns-Reversal Precedence)...")
    RRP = rolling_rrp(volume, np.abs(returns), window=252, lag=20)
    print(f"   Mean: {np.nanmean(RRP):.3f}")
    print(f"   Range: [{np.nanmin(RRP):.3f}, {np.nanmax(RRP):.3f}]")

    print("\n4. Computing forward drawdown...")
    prices = 100 * np.exp(np.cumsum(returns))
    FWD_DD = forward_drawdown(prices, horizon=20)
    print(f"   Mean: {np.nanmean(FWD_DD):.3%}")
    print(f"   Min (worst drawdown): {np.nanmin(FWD_DD):.3%}")

    print("\n5. Computing CCI (Composite Crisis Index)...")
    CCI = compute_cci(RERL, RTCL, RRP)
    print(f"   Mean: {np.nanmean(CCI):.3f}")
    print(f"   Std: {np.nanstd(CCI):.3f}")
    print(f"   Range: [{np.nanmin(CCI):.3f}, {np.nanmax(CCI):.3f}]")

    print("\n✓ All metrics computed successfully!")
