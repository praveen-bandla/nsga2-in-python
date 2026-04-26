"""
Data infrastructure for portfolio optimization.
Loads pre-computed data from the data pipeline (data/processed/).
Run the pipeline first: python data_pipeline/data_loader.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import configs


def load_from_pipeline():
    """
    Load returns, covariance, and std devs from the data pipeline output.
    Expects parquet files in data/processed/.
    """
    proc = Path(configs.PROC_DIR)
    returns_path = proc / configs.RETURNS_DAILY_FILENAME

    if not returns_path.exists():
        raise FileNotFoundError(
            f"Data not found at {returns_path}. "
            "Run the pipeline first: python data_pipeline/data_loader.py"
        )

    returns_df = pd.read_parquet(returns_path)
    returns_df = returns_df.dropna(axis=1, how='any')

    ticker_names = returns_df.columns.tolist()
    returns_array = returns_df.values.astype(np.float64)

    # Compute daily statistics directly from returns (consistent units).
    # The pipeline's covariance is annualized (daily * 252) which would
    # create a mismatch with daily mean_returns and std_returns.
    mean_returns = np.mean(returns_array, axis=0)
    cov_matrix   = np.cov(returns_array.T, ddof=1)
    std_returns  = np.std(returns_array, axis=0, ddof=1)

    print(f"Loaded {len(ticker_names)} stocks, {len(returns_df)} trading days")
    return mean_returns, cov_matrix, std_returns, ticker_names
