from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

TICKER_PATH = ROOT_DIR / "data_pipeline" / "tickers.json"


# Data download configs
BATCH_SIZE = 100  # yfinance can typically handle ~100 tickers without timeouts
DELAY = 2.0
LOOKBACK_YEARS = 10
START_DATE = (datetime.today() - timedelta(days=365 * LOOKBACK_YEARS + 3)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")


with TICKER_PATH.open("r", encoding="utf-8") as f:
    _raw = json.load(f)

_TICKERS_JSON = _raw.get("tickers") or _raw.get("TICKERS")

if _TICKERS_JSON is None:
    raise KeyError(
        f"Expected 'tickers' or 'TICKERS' key in {TICKER_PATH}, found: {list(_raw.keys())}"
    )

TEST_TICKERS = _TICKERS_JSON["TEST"]
SP500_TICKERS = _TICKERS_JSON["SP500"]
DEFAULT_TICKERS = SP500_TICKERS


# Output filenames
RAW_BATCH_FILENAME_TEMPLATE = "batch_{batch_idx:02d}.parquet"
PRICES_ADJ_CLOSE_FILENAME = "prices_adj_close.parquet"
RETURNS_DAILY_FILENAME = "returns_daily.parquet"
RETURNS_MONTHLY_FILENAME = "returns_monthly.parquet"
COVARIANCE_DAILY_FILENAME = "covariance_daily.parquet"
COVARIANCE_MONTHLY_FILENAME = "covariance_monthly.parquet"
BENCHMARK_SPY_FILENAME = "benchmark_spy.parquet"


# Computation parameters
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
MIN_DAILY_OBSERVATIONS = 252
MIN_MONTHLY_OBSERVATIONS = 24

