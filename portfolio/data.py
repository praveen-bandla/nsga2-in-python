"""
Data infrastructure for portfolio optimization.

Downloads S&P 500 stock prices via yfinance, computes return matrices
and covariance matrices. Includes synthetic data generation for testing
without internet access.

"""

import numpy as np

# 50 S&P 500 tickers across major GICS sectors
DEFAULT_TICKERS = [
    # Technology (10)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'AVGO', 'CSCO', 'ORCL', 'ADBE',
    # Healthcare (7)
    'JNJ', 'UNH', 'PFE', 'ABT', 'MRK', 'TMO', 'ABBV',
    # Financials (6)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK',
    # Consumer (8)
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'MCD', 'NKE',
    # Energy (3)
    'XOM', 'CVX', 'COP',
    # Industrials (6)
    'CAT', 'HON', 'UPS', 'GE', 'BA', 'RTX',
    # Utilities (3)
    'NEE', 'DUK', 'SO',
    # Materials (3)
    'LIN', 'APD', 'SHW',
    # Real Estate (2)
    'AMT', 'PLD',
    # Communication (2)
    'DIS', 'CMCSA',
]


def download_stock_data(tickers=None, start_date='2014-01-01', end_date='2024-01-01'):
    """
    Download adjusted close prices for given tickers from Yahoo Finance.

    Args:
        tickers: List of ticker symbols. Defaults to DEFAULT_TICKERS.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        pandas DataFrame of adjusted close prices (dates x tickers).
    """
    import yfinance as yf

    if tickers is None:
        tickers = DEFAULT_TICKERS

    print(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    if hasattr(data.columns, 'levels'):
        prices = data['Close']
    else:
        prices = data

    # drop stocks with too much missing data (>5% NaN), can adjust threshold
    threshold = len(prices) * 0.05
    prices = prices.dropna(axis=1, thresh=len(prices) - threshold)
    prices = prices.ffill().bfill()

    print(f"{prices.shape[1]} stocks with {prices.shape[0]} trading days")
    return prices


def compute_returns(prices):
    """
    Compute daily simple returns from price data.

    Args:
        prices: DataFrame of stock prices (dates x tickers).

    Returns:
        DataFrame of daily returns (dates x tickers), first row dropped.
    """
    returns = prices.pct_change().dropna()
    return returns


def compute_statistics(returns):
    """
    Compute mean returns, covariance matrix, and standard deviations.

    All values are daily (not annualized). Annualize externally if needed:
        annual_return = daily_mean * 252
        annual_cov = daily_cov * 252
        annual_std = daily_std * sqrt(252)

    Args:
        returns: DataFrame of daily returns.

    Returns:
        Tuple of (mean_returns, cov_matrix, std_returns) as numpy arrays.
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    std_returns = returns.std().values
    return mean_returns, cov_matrix, std_returns


def generate_synthetic_data(num_stocks=50, num_days=2520, seed=42):
    """
    Generate synthetic stock return data for testing without internet.

    Creates realistic-ish correlated returns using a factor model:
        r_i = beta_i * f + epsilon_i

    Args:
        num_stocks: Number of stocks to simulate.
        num_days: Number of trading days (default ~10 years).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean_returns, cov_matrix, std_returns) as numpy arrays.
    """
    rng = np.random.RandomState(seed)

    # pos drift
    market_factor = rng.normal(0.0003, 0.012, num_days)

    betas = rng.uniform(0.5, 1.5, num_stocks)
    alphas = rng.normal(0.0001, 0.0002, num_stocks)
    idio_vols = rng.uniform(0.005, 0.025, num_stocks)

    # correlated returns
    returns = np.zeros((num_days, num_stocks))
    for i in range(num_stocks):
        epsilon = rng.normal(0, idio_vols[i], num_days)
        returns[:, i] = alphas[i] + betas[i] * market_factor + epsilon

    mean_returns = returns.mean(axis=0)
    cov_matrix = np.cov(returns, rowvar=False)
    std_returns = returns.std(axis=0)

    return mean_returns, cov_matrix, std_returns
