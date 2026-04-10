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
DELAY = 3.0
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


# -----------------------------
# Backtesting / extractor (v1)
# -----------------------------
# These defaults are used by data_pipeline/data_extractor.py.
# They keep I/O + missing-value policy consistent across backtests.

# Processed inputs
EXTRACTOR_RETURNS_FILENAME = RETURNS_DAILY_FILENAME
EXTRACTOR_PRICES_FILENAME = PRICES_ADJ_CLOSE_FILENAME

# Missing-data handling
# - Returns: fill missing values with 0.0 (neutral return)
# - Prices: forward-fill then back-fill to get a valid price vector each day
EXTRACTOR_RETURNS_FILL_VALUE = 0.0
EXTRACTOR_PRICES_FILL_METHOD = "ffill_bfill"  # supported: "ffill_bfill"

# Ticker ordering
EXTRACTOR_SORT_TICKERS = True


# Computation parameters
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
MIN_DAILY_OBSERVATIONS = 252
MIN_MONTHLY_OBSERVATIONS = 24

