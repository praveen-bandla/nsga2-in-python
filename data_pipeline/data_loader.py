"""S&P 500 Data Pipeline.

Downloads price history, computes returns/covariance, saves locally.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import sys

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from configs import *

warnings.filterwarnings("ignore")


class SP500Pipeline:

    def __init__(self, tickers: list[str] | None = None):
        self.base = Path(__file__).parent

        self.raw = RAW_DIR
        self.proc = Path(PROC_DIR)
        self.end = END_DATE
        self.start = START_DATE
        self.batch_size = BATCH_SIZE
        self.delay = DELAY
        self.tickers = tickers if tickers is not None else list(DEFAULT_TICKERS)

        for d in (self.raw, self.proc):
            d.mkdir(parents=True, exist_ok=True)


    def get_tickers(self):
        print(f"{len(self.tickers)} tickers loaded")
        return self.tickers


    def download_prices(self, tickers):
        '''downloads adjusted close prices for the given tickers in batches (will download 492/503
        since some tickers are new or delisted within the 10 year window) '''
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        print(f"\ndownloading {len(tickers)} tickers in {len(batches)} batches")

        frames = []

        for i, batch in enumerate(tqdm(batches, unit="batch")):
            path = self.raw / RAW_BATCH_FILENAME_TEMPLATE.format(batch_idx=i)

            if path.exists():
                frames.append(pd.read_parquet(path))
                continue

            try:
                df = yf.download(batch, start=self.start, end=self.end, progress=False)
                df["Close"].to_parquet(path)
                frames.append(df["Close"])

            except Exception as e:
                tqdm.write(f"Batch {i:02d} failed: {e}")

            if i < len(batches) - 1:
                time.sleep(self.delay)

        prices = pd.concat(frames, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices.dropna(axis=1, how="all", inplace=True)
        print(f"price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")

        return prices


    def compute_matrices(self, prices):
        print("\ncomputing return & covariance matrices")
        prices.to_parquet(self.proc / PRICES_ADJ_CLOSE_FILENAME)

        # daily log returns: log(P_t / P_{t-1})
        daily = np.log(prices/prices.shift(1))
        daily = daily.dropna(how="all")

        # monthly log returns: resample to month-end, then log returns
        monthly_prices = prices.resample("ME").last()
        monthly = np.log(monthly_prices/monthly_prices.shift(1))
        monthly = monthly.dropna(how="all")

        daily.to_parquet(self.proc / RETURNS_DAILY_FILENAME)
        monthly.to_parquet(self.proc / RETURNS_MONTHLY_FILENAME)

        daily_filtered = daily.loc[:, daily.notna().sum() >= MIN_DAILY_OBSERVATIONS]
        cov_daily = daily_filtered.cov() * TRADING_DAYS_PER_YEAR
        cov_daily.to_parquet(self.proc / COVARIANCE_DAILY_FILENAME)

        monthly_filtered = monthly.loc[:, monthly.notna().sum() >= MIN_MONTHLY_OBSERVATIONS]
        cov_monthly = monthly_filtered.cov() * MONTHS_PER_YEAR
        cov_monthly.to_parquet(self.proc / COVARIANCE_MONTHLY_FILENAME)


    def download_benchmark(self):
        print("\ndownloading SPY benchmark")

        spy = yf.download("SPY", start=self.start, end=self.end, progress=False)
        spy = spy[["Close"]].rename(columns={"Close": "SPY"})
        spy.to_parquet(self.proc / BENCHMARK_SPY_FILENAME)


    def run(self):
        prices = self.download_prices(self.get_tickers())
        self.compute_matrices(prices)
        self.download_benchmark()
    
        print(f"\ndone {self.base.resolve()}")


if __name__ == "__main__":
    tickers = TEST_TICKERS if "--test" in sys.argv else None
    SP500Pipeline(tickers=tickers).run()