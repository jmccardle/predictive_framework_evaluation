"""
Data loading utilities for financial crisis prediction system evaluation.

This module provides functions to load and prepare financial market data
for analysis, with strict attention to preventing data leakage.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional
from datetime import datetime


def load_sp500(
    start: str = "1998-01-01",
    end: Optional[str] = None,
    symbol: str = "^GSPC"
) -> Tuple[pd.DataFrame, str]:
    """
    Load S&P 500 data from Yahoo Finance.

    This function downloads historical S&P 500 price and volume data,
    ensuring all necessary fields are present for analysis.

    Parameters
    ----------
    start : str, default="1998-01-01"
        Start date in YYYY-MM-DD format
    end : str, optional
        End date in YYYY-MM-DD format. If None, uses current date.
    symbol : str, default="^GSPC"
        Ticker symbol to download

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    source : str
        Data source description

    Examples
    --------
    >>> df, source = load_sp500(start="2000-01-01", end="2024-01-01")
    >>> print(f"Loaded {len(df)} days from {source}")

    Notes
    -----
    - Returns are calculated as log returns: log(P_t / P_{t-1})
    - Volume data is included for RRP metric calculation
    - Data is sorted chronologically (oldest to newest)
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # Download data from Yahoo Finance
    data = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    # Reset index to make Date a column
    df = data.reset_index()

    # Ensure we have the required columns
    required_cols = ["Date", "Close", "Adj Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort by date (oldest to newest) - CRITICAL for causal analysis
    df = df.sort_values("Date").reset_index(drop=True)

    # Check for missing data
    if df.isnull().any().any():
        n_missing = df.isnull().sum().sum()
        print(f"Warning: {n_missing} missing values detected. Dropping rows with NaN.")
        df = df.dropna()

    source = f"Yahoo Finance ({symbol})"

    return df, source


def load_multi_asset(
    symbols: list,
    start: str = "2000-01-01",
    end: Optional[str] = None
) -> dict:
    """
    Load data for multiple assets.

    Parameters
    ----------
    symbols : list
        List of ticker symbols (e.g., ["^GSPC", "BTC-USD", "GC=F"])
    start : str
        Start date in YYYY-MM-DD format
    end : str, optional
        End date in YYYY-MM-DD format

    Returns
    -------
    data : dict
        Dictionary mapping symbol -> DataFrame

    Examples
    --------
    >>> symbols = ["^GSPC", "BTC-USD", "GC=F", "TLT"]
    >>> data = load_multi_asset(symbols, start="2015-01-01")
    >>> for symbol, df in data.items():
    ...     print(f"{symbol}: {len(df)} days")
    """
    data = {}

    for symbol in symbols:
        try:
            df, _ = load_sp500(start=start, end=end, symbol=symbol)
            data[symbol] = df
            print(f"✓ Loaded {symbol}: {len(df)} days")
        except Exception as e:
            print(f"✗ Failed to load {symbol}: {e}")

    return data


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from prices.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices (length N)

    Returns
    -------
    returns : np.ndarray
        Array of log returns (length N-1)

    Notes
    -----
    Returns are computed as: r_t = log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})

    This is standard in finance as log returns are:
    - Symmetric
    - Time-additive
    - Approximately normal for small changes
    """
    return np.diff(np.log(prices))


def split_train_test(
    data: np.ndarray,
    train_size: int,
    test_size: int,
    strict_separation: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into training and test sets with strict temporal separation.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    train_size : int
        Number of samples for training
    test_size : int
        Number of samples for testing
    strict_separation : bool, default=True
        If True, ensures no overlap between train and test

    Returns
    -------
    train_data : np.ndarray
        Training data (indices 0:train_size)
    test_data : np.ndarray
        Test data (indices train_size:train_size+test_size)

    Raises
    ------
    ValueError
        If train_size + test_size > len(data)

    Examples
    --------
    >>> prices = np.array([100, 101, 102, 103, 104, 105])
    >>> train, test = split_train_test(prices, train_size=4, test_size=2)
    >>> train
    array([100, 101, 102, 103])
    >>> test
    array([104, 105])
    """
    if train_size + test_size > len(data):
        raise ValueError(
            f"train_size ({train_size}) + test_size ({test_size}) = "
            f"{train_size + test_size} exceeds data length ({len(data)})"
        )

    train_data = data[:train_size]

    if strict_separation:
        # No overlap: test starts immediately after train ends
        test_data = data[train_size:train_size + test_size]
    else:
        # Allow overlap (for rolling window methods)
        test_data = data[train_size:train_size + test_size]

    return train_data, test_data


def get_crisis_periods() -> dict:
    """
    Return dictionary of known crisis periods for testing.

    Returns
    -------
    crises : dict
        Dictionary mapping crisis name -> (start_date, end_date)

    Notes
    -----
    Crisis periods used in TEST_PROGRAM_SPECIFICATION.md:
    - 2008 GFC (Global Financial Crisis)
    - COVID-19 crash
    - 2022 Bear Market
    """
    return {
        "2008 GFC": ("2007-10-01", "2009-03-01"),
        "COVID-19": ("2020-02-15", "2020-05-15"),
        "2022 Bear": ("2022-01-01", "2022-10-31"),
    }


if __name__ == "__main__":
    # Example usage and validation
    print("Loading S&P 500 data...")
    df, source = load_sp500(start="2000-01-01", end="2024-11-01")

    print(f"\nData source: {source}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total days: {len(df)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")

    # Test train/test split
    prices = df["Adj Close"].values
    train, test = split_train_test(prices, train_size=2000, test_size=400)
    print(f"\nTrain/test split:")
    print(f"  Train size: {len(train)}")
    print(f"  Test size: {len(test)}")
    print(f"  Total: {len(train) + len(test)}")
