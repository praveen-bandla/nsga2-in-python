"""Backtesting data extractor.

Loads processed parquet outputs once (from configs.PROC_DIR) and serves
windowed tensor slices for daily backtesting.

Design goals (v1):
- Single disk read per backtest run (load full [start_date, final_end_date]).
- Models receive tensor-like arrays (np.ndarray) aligned by date + ticker.
- Supports a moving "pause" date that can be advanced day-by-day.

Expected processed inputs (created by data_pipeline/data_loader.py):
- returns_daily.parquet
- prices_adj_close.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sys


# Allow running as a script while importing project-level modules like configs.py.
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import configs


DateLike = str | date | datetime | np.datetime64 | pd.Timestamp


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


@dataclass(frozen=True)
class ExtractedWindow:
    """A single extracted window ready for model consumption."""

    as_of_idx: int
    returns_history: np.ndarray
    prices_history: np.ndarray
    dates: np.ndarray
    tickers: list[str]


class BacktestDataExtractor:
    """Loads processed data once and serves slices up to a moving pause date.

    Parameters
    ----------
    start_date:
        First date to include in the loaded dataset.
    pause_date:
        Current "as-of" date for extraction (must be within loaded date range).
        If None, defaults to start_date.
    final_end_date:
        Last date to include in the loaded dataset.
    proc_dir:
        Optional override of processed data folder. Defaults to configs.PROC_DIR.
    """

    def __init__(
        self,
        start_date: DateLike,
        pause_date: DateLike | None = None,
        final_end_date: DateLike | None = None,
        *,
        proc_dir: Path | None = None,
    ) -> None:
        self.start_date = _to_timestamp(start_date)
        self.final_end_date = _to_timestamp(final_end_date) if final_end_date is not None else None
        self.pause_date = _to_timestamp(pause_date) if pause_date is not None else self.start_date

        self.proc_dir = Path(proc_dir) if proc_dir is not None else Path(configs.PROC_DIR)
        self.returns_path = self.proc_dir / getattr(
            configs, "EXTRACTOR_RETURNS_FILENAME", configs.RETURNS_DAILY_FILENAME
        )
        self.prices_path = self.proc_dir / getattr(
            configs, "EXTRACTOR_PRICES_FILENAME", configs.PRICES_ADJ_CLOSE_FILENAME
        )

        self._loaded = False
        self._dates: np.ndarray | None = None
        self._tickers: list[str] | None = None
        self._returns_all: np.ndarray | None = None
        self._prices_all: np.ndarray | None = None
        self._pause_idx: int | None = None

    # -----------------
    # Loading & alignment
    # -----------------

    def load(self) -> None:
        """Load and align returns + prices once from disk."""

        if not self.returns_path.exists():
            raise FileNotFoundError(f"Missing returns file: {self.returns_path}")
        if not self.prices_path.exists():
            raise FileNotFoundError(f"Missing prices file: {self.prices_path}")

        returns_df = pd.read_parquet(self.returns_path)
        prices_df = pd.read_parquet(self.prices_path)

        # Normalize indices
        returns_df.index = pd.to_datetime(returns_df.index)
        prices_df.index = pd.to_datetime(prices_df.index)
        returns_df.sort_index(inplace=True)
        prices_df.sort_index(inplace=True)

        # Limit date range
        end_ts = self.final_end_date if self.final_end_date is not None else returns_df.index.max()
        end_ts = min(end_ts, returns_df.index.max(), prices_df.index.max())

        returns_df = returns_df.loc[(returns_df.index >= self.start_date) & (returns_df.index <= end_ts)]
        prices_df = prices_df.loc[(prices_df.index >= self.start_date) & (prices_df.index <= end_ts)]

        # Align on common dates and tickers
        common_dates = returns_df.index.intersection(prices_df.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between returns and prices in the requested range.")

        common_tickers = list(set(returns_df.columns).intersection(set(prices_df.columns)))
        if len(common_tickers) == 0:
            raise ValueError("No overlapping tickers between returns and prices.")

        if getattr(configs, "EXTRACTOR_SORT_TICKERS", True):
            common_tickers = sorted(common_tickers)

        returns_df = returns_df.loc[common_dates, common_tickers]
        prices_df = prices_df.loc[common_dates, common_tickers]

        # Handle missing values (v1 defaults live in configs.py)
        prices_fill_method = getattr(configs, "EXTRACTOR_PRICES_FILL_METHOD", "ffill_bfill")
        if prices_fill_method != "ffill_bfill":
            raise ValueError(f"Unsupported EXTRACTOR_PRICES_FILL_METHOD={prices_fill_method!r}")

        prices_df = prices_df.ffill().bfill()
        returns_df = returns_df.fillna(getattr(configs, "EXTRACTOR_RETURNS_FILL_VALUE", 0.0))

        self._dates = common_dates.to_numpy()
        self._tickers = list(common_tickers)
        self._returns_all = returns_df.to_numpy(dtype=float)
        self._prices_all = prices_df.to_numpy(dtype=float)

        # Initialize pause index
        self.set_pause_date(self.pause_date)

        self._loaded = True

    # -----------------
    # Pause date control
    # -----------------

    def set_pause_date(self, pause_date: DateLike) -> None:
        """Set the current pause date (as-of) to the last available date <= pause_date."""

        if self._dates is None:
            # allow calling before load(); will be applied after load()
            self.pause_date = _to_timestamp(pause_date)
            return

        pause_ts = _to_timestamp(pause_date)

        dates = pd.to_datetime(self._dates)
        # Find insertion point to keep sorted order; then step back one.
        # This yields the last index where dates[idx] <= pause_ts.
        idx = int(dates.searchsorted(pause_ts, side="right") - 1)
        if idx < 0:
            raise ValueError(
                f"pause_date={pause_ts.date()} is before the first available date {dates[0].date()}"
            )
        self._pause_idx = idx

    # -----------------
    # Extraction API
    # -----------------

    def extract(self) -> ExtractedWindow:
        """Return tensors for the window [start_date, pause_date] (inclusive)."""

        if not self._loaded:
            self.load()

        assert self._dates is not None
        assert self._tickers is not None
        assert self._returns_all is not None
        assert self._prices_all is not None
        assert self._pause_idx is not None

        end = self._pause_idx
        returns_history = self._returns_all[: end + 1]
        prices_history = self._prices_all[: end + 1]
        dates = self._dates[: end + 1]

        return ExtractedWindow(
            as_of_idx=end,
            returns_history=returns_history,
            prices_history=prices_history,
            dates=dates,
            tickers=list(self._tickers),
        )

    def add_data(self, n_days: int = 1) -> ExtractedWindow:
        """Advance the pause date by `n_days` trading days and return the new window."""

        if n_days < 1:
            raise ValueError("n_days must be >= 1")

        if not self._loaded:
            self.load()

        assert self._pause_idx is not None
        assert self._dates is not None

        self._pause_idx = min(self._pause_idx + int(n_days), len(self._dates) - 1)
        return self.extract()

    # -----------------
    # Convenience helpers
    # -----------------

    @property
    def tickers(self) -> list[str]:
        if not self._loaded:
            self.load()
        assert self._tickers is not None
        return list(self._tickers)

    @property
    def dates(self) -> np.ndarray:
        if not self._loaded:
            self.load()
        assert self._dates is not None
        return self._dates.copy()


if __name__ == "__main__":
    # Minimal sanity demo (prints shapes). Example:
    #   ./venv/bin/python data_pipeline/data_extractor.py 2020-01-01 2020-03-01 2020-12-31
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        raise SystemExit(
            "Usage: python data_pipeline/data_extractor.py <start_date> <pause_date> [final_end_date]"
        )

    start = args[0]
    pause = args[1]
    final_end = args[2] if len(args) == 3 else None

    ex = BacktestDataExtractor(start_date=start, pause_date=pause, final_end_date=final_end)
    w = ex.extract()
    print(
        "Extracted:",
        f"dates={w.dates.shape}",
        f"tickers={len(w.tickers)}",
        f"returns={w.returns_history.shape}",
        f"prices={w.prices_history.shape}",
        f"as_of_idx={w.as_of_idx}",
    )
