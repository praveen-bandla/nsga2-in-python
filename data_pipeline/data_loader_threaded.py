"""S&P 500 Data Pipeline.

Downloads price history, computes returns/covariance, saves locally.
"""

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
        self.batch_size = 5 # configs.BATCH_SIZE
        self.delay = configs.DELAY
        self.tickers = tickers if tickers is not None else list(configs.DEFAULT_TICKERS)

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

        # Pre-allocate results list - index = batch index, guarantees insertion order
        results: list[pd.DataFrame | pd.Series | None] = [None] * len(batches)
        lock = Lock()

        def fetch_batch(i: int, batch: list[str], pbar: tqdm) -> None:
            path = self.raw / configs.RAW_BATCH_FILENAME_TEMPLATE.format(batch_idx=i)

            try:
                if path.exists():
                    df = pd.read_parquet(path)
                else:
                    # yf.Ticker is stateless — fully thread-safe, no shared._DFS mutation
                    frames = {
                        ticker: yf.Ticker(ticker).history(
                            start=self.start, end=self.end, auto_adjust=True
                        )["Close"].rename(ticker)
                        for ticker in batch
                    }
                    df = pd.concat(frames.values(), axis=1)
                    df.columns = batch
                    df.to_parquet(path)

                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                results[i] = df

            except Exception as e:
                tqdm.write(f"Batch {i:02d} failed: {e}")

            finally:
                with lock:
                    pbar.update(1)

        with tqdm(total=len(batches), unit="batch") as pbar:
            threads = [
                Thread(target=fetch_batch, args=(i, batch, pbar), daemon=True)
                for i, batch in enumerate(batches)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()  # Block until ALL threads complete before proceeding

        # Filter None (failed batches), concat in original order
        frames = [r for r in results if r is not None]

        # for f in frames:
        #     print(f)

        prices = pd.concat(frames, axis=1)
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
        spy = yf.download("SPY", start=self.start, end=self.end, progress=False)
        spy = spy[["Close"]].rename(columns={"Close": "SPY"})
        spy.to_parquet(self.proc / configs.BENCHMARK_SPY_FILENAME)


    def run(self):
        prices = self.download_prices(self.get_tickers())
        self.compute_matrices(prices)
        self.download_benchmark()
        print(f"\ndone {self.base.resolve()}")


if __name__ == "__main__":
    tickers = configs.TEST_TICKERS if "--test" in sys.argv else None
    SP500Pipeline(tickers=tickers).run()