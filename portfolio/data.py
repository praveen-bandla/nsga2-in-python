import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import configs


def load_from_pipeline():
    proc = Path(configs.PROC_DIR)
    returns_path = proc / configs.RETURNS_DAILY_FILENAME

    if not returns_path.exists():
        raise FileNotFoundError(
            f"Data not found at {returns_path}. "
            "Run the pipeline first: python data_pipeline/data_loader.py"
        )

    returns_df = pd.read_parquet(returns_path)
    returns_df = returns_df.dropna(axis=1, how="any")

    mean_returns = returns_df.mean().values
    lw = LedoitWolf()
    lw.fit(returns_df.values)
    cov_matrix = lw.covariance_
    std_returns = np.sqrt(np.diag(cov_matrix))
    ticker_names = list(returns_df.columns)

    print(f"Loaded {len(ticker_names)} stocks, {len(returns_df)} trading days")
    return mean_returns, cov_matrix, std_returns, ticker_names
