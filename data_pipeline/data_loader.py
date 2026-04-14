"""S&P 500 Data Pipeline.

Downloads price history, computes returns/covariance, saves locally.
"""

import time
import warnings
from pathlib import Path
from threading import Thread, Lock

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import sys

# Allow running as a script (python data_pipeline/data_loader.py) while still
# importing project-level modules like configs.py.
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import configs

warnings.filterwarnings("ignore")


class SP500Pipeline:

    def __init__(self, tickers: list[str] | None = None):
        self.base = Path(__file__).parent

        self.raw = Path(configs.RAW_DIR)
        self.proc = Path(configs.PROC_DIR)
        self.end = configs.END_DATE
        self.start = configs.START_DATE
        self.batch_size = configs.BATCH_SIZE
        self.delay = configs.DELAY
        self.tickers = tickers if tickers is not None else list(configs.DEFAULT_TICKERS)

        for d in (self.raw, self.proc):
            d.mkdir(parents=True, exist_ok=True)


    def get_tickers(self):
        print(f"{len(self.tickers)} tickers loaded")
        return self.tickers
    

    def download_prices(self, tickers):
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        print(f"\ndownloading {len(tickers)} tickers in {len(batches)} batches")

        frames: list[pd.DataFrame | None] = [None] * len(batches)
        lock = Lock()
        pbar = tqdm(total=len(batches), unit="batch")

        def fetch(i, batch):
            path = self.raw / configs.RAW_BATCH_FILENAME_TEMPLATE.format(batch_idx=i)
            try:
                if path.exists():
                    result = pd.read_parquet(path)
                else:

                    result = yf.download(batch, start=self.start, end=self.end, progress=False)
                    # Add validation check
                    if result is None or result.empty or "Close" not in result.columns:
                        tqdm.write(f"Batch {i:02d}: No valid data retrieved")
                        return
                    result["Close"].to_parquet(path)
                frames[i] = result
            except Exception as e:
                tqdm.write(f"Batch {i:02d} failed: {e}")
            finally:
                if i < len(batches) - 1:
                    time.sleep(self.delay)
                with lock:
                    pbar.update(1)

        threads = [Thread(target=fetch, args=(i, batch)) for i, batch in enumerate(batches)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pbar.close()

        prices = pd.concat([f for f in frames if f is not None], axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices.dropna(axis=1, how="all", inplace=True)
        print(f"price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
        return prices


    def compute_matrices(self, prices):
        print("\ncomputing return & covariance matrices")
        prices.to_parquet(self.proc / configs.PRICES_ADJ_CLOSE_FILENAME)

        # daily log returns: log(P_t / P_{t-1})
        daily = np.log(prices/prices.shift(1))
        daily = daily.dropna(how="all")

        # monthly log returns: resample to month-end, then log returns
        monthly_prices = prices.resample("ME").last()
        monthly = np.log(monthly_prices/monthly_prices.shift(1))
        monthly = monthly.dropna(how="all")

        daily.to_parquet(self.proc / configs.RETURNS_DAILY_FILENAME)
        monthly.to_parquet(self.proc / configs.RETURNS_MONTHLY_FILENAME)

        daily_filtered = daily.loc[:, daily.notna().sum() >= configs.MIN_DAILY_OBSERVATIONS]
        cov_daily = daily_filtered.cov() * configs.TRADING_DAYS_PER_YEAR
        cov_daily.to_parquet(self.proc / configs.COVARIANCE_DAILY_FILENAME)

        monthly_filtered = monthly.loc[:, monthly.notna().sum() >= configs.MIN_MONTHLY_OBSERVATIONS]
        cov_monthly = monthly_filtered.cov() * configs.MONTHS_PER_YEAR
        cov_monthly.to_parquet(self.proc / configs.COVARIANCE_MONTHLY_FILENAME)


    def download_benchmark(self):
        print("\ndownloading SPY benchmark")

        spy = yf.Ticker("SPY").history(start=self.start, end=self.end)["Close"]
    
        if spy is None or spy.empty:
            raise ValueError("Failed to download SPY benchmark data")
        
        spy.index = pd.DatetimeIndex(spy.index).tz_localize(None)  # strip timezone to match other frames
        spy = spy.rename("SPY").to_frame()
        spy.to_parquet(self.proc / configs.BENCHMARK_SPY_FILENAME)


    def run(self):
        prices = self.download_prices(self.get_tickers())
        self.compute_matrices(prices)
        self.download_benchmark()
    
        print(f"\ndone {self.base.resolve()}")


if __name__ == "__main__":
    tickers = configs.TEST_TICKERS if "--test" in sys.argv else None
    SP500Pipeline(tickers=tickers).run()