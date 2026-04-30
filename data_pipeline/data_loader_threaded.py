import sys
import warnings
from pathlib import Path
from threading import Lock, Thread

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from configs import *

warnings.filterwarnings("ignore")


class SP500Pipeline:
    def __init__(self, tickers: list[str] | None = None):
        self.base = Path(__file__).parent
        self.raw = RAW_DIR
        self.proc = PROC_DIR
        self.end = END_DATE
        self.start = START_DATE
        self.batch_size = 5
        self.delay = DELAY
        self.tickers = tickers if tickers is not None else list(DEFAULT_TICKERS)

        for directory in (self.raw, self.proc):
            directory.mkdir(parents=True, exist_ok=True)

    def get_tickers(self):
        print(f"{len(self.tickers)} tickers loaded")
        return self.tickers

    def download_prices(self, tickers):
        batches = [
            tickers[i : i + self.batch_size]
            for i in range(0, len(tickers), self.batch_size)
        ]
        print(f"\ndownloading {len(tickers)} tickers in {len(batches)} batches")

        results: list[pd.DataFrame | pd.Series | None] = [None] * len(batches)
        lock = Lock()

        def fetch_batch(i: int, batch: list[str], pbar: tqdm) -> None:
            path = self.raw / RAW_BATCH_FILENAME_TEMPLATE.format(batch_idx=i)
            try:
                if path.exists():
                    df = pd.read_parquet(path)
                else:
                    frames = {
                        ticker: yf.Ticker(ticker)
                        .history(start=self.start, end=self.end, auto_adjust=True)[
                            "Close"
                        ]
                        .rename(ticker)
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
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        frames = [result for result in results if result is not None]
        prices = pd.concat(frames, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices.dropna(axis=1, how="all", inplace=True)
        print(f"price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")
        return prices

    def compute_matrices(self, prices):
        print("\ncomputing return & covariance matrices")
        prices.to_parquet(self.proc / PRICES_ADJ_CLOSE_FILENAME)

        daily = np.log(prices / prices.shift(1)).dropna(how="all")
        monthly_prices = prices.resample("ME").last()
        monthly = np.log(monthly_prices / monthly_prices.shift(1)).dropna(how="all")

        daily.to_parquet(self.proc / RETURNS_DAILY_FILENAME)
        monthly.to_parquet(self.proc / RETURNS_MONTHLY_FILENAME)

        daily_filtered = daily.loc[:, daily.notna().sum() >= MIN_DAILY_OBSERVATIONS]
        cov_daily = daily_filtered.cov() * TRADING_DAYS_PER_YEAR
        cov_daily.to_parquet(self.proc / COVARIANCE_DAILY_FILENAME)

        monthly_filtered = monthly.loc[
            :, monthly.notna().sum() >= MIN_MONTHLY_OBSERVATIONS
        ]
        cov_monthly = monthly_filtered.cov() * TRADING_DAYS_PER_YEAR
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
