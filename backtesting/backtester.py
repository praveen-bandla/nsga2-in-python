from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSummary:
    start_date: str
    end_date: str
    n_days: int
    n_assets: int
    total_return_portfolio: float
    total_return_spy: float
    annualized_return_portfolio: float
    annualized_return_spy: float


class PortfolioBacktester:
    """Backtest a single fixed weight vector vs SPY.

    Sub-steps performed by `run()`:
    1) Load frozen weights (ticker, weight)
    2) Load daily asset log returns and SPY close prices
    3) Align on common trading days, then slice to the configured date window
    4) Convert asset log returns -> simple returns and compute portfolio daily returns
    5) Compute SPY daily returns from close prices
    6) Build equity curves starting at `initial_equity`

    This is intentionally minimal (v1) and designed to be extended later.
    """

    def __init__(
        self,
        *,
        returns_path: Path | None = None,
        spy_prices_path: Path | None = None,
        weights_csv_path: Path | None = None,
        start_date: str,
        end_date: str,
        initial_equity: float = 1.0,
        spy_initial_equity: float | None = None,
        trading_days_per_year: int = 252,
        returns_df: pd.DataFrame | None = None,
        spy_close: pd.Series | None = None,
        weights_df: pd.DataFrame | None = None,
    ) -> None:
        self.returns_path = Path(returns_path) if returns_path is not None else None
        self.spy_prices_path = Path(spy_prices_path) if spy_prices_path is not None else None
        self.weights_csv_path = Path(weights_csv_path) if weights_csv_path is not None else None
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.initial_equity = float(initial_equity)
        self.spy_initial_equity = float(self.initial_equity if spy_initial_equity is None else spy_initial_equity)
        self.trading_days_per_year = int(trading_days_per_year)

        self._returns_df = returns_df
        self._spy_close = spy_close
        self._weights_df = weights_df

    def run(self) -> tuple[pd.DataFrame, BacktestSummary]:
        # 1) Load inputs
        weights_df = self._weights_df if self._weights_df is not None else self._load_weights()
        returns_df = self._returns_df if self._returns_df is not None else self._load_returns()
        spy_close = self._spy_close if self._spy_close is not None else self._load_spy_close()

        # 2) Align portfolio/benchmark to the same trading days
        returns_df, spy_close = self._align_dates(returns_df, spy_close)

        # 3) Apply the configured backtest window
        returns_df, spy_close = self._slice_window(returns_df, spy_close)

        # 4) Align weight vector to available return columns (drop missing tickers, renormalize)
        aligned_tickers, w = self._align_weights_to_returns(weights_df, returns_df.columns)
        returns_df = returns_df.loc[:, aligned_tickers].fillna(0.0)

        # 5) Realized daily return series
        port_r = self._portfolio_simple_returns(returns_df, w)
        spy_r = self._spy_simple_returns(spy_close)

        # 6) Equity curves
        port_equity = self._equity_curve(port_r, initial_equity=self.initial_equity)
        spy_equity = self._equity_curve(spy_r, initial_equity=self.spy_initial_equity)

        out = pd.DataFrame(
            {
                "portfolio_return": port_r,
                "portfolio_equity": port_equity,
                "spy_return": spy_r,
                "spy_equity": spy_equity,
            },
            index=returns_df.index,
        )
        out.index.name = "date"

        summary = self._summarize(out, n_assets=int(len(w)))
        return out, summary

    def _load_weights(self) -> pd.DataFrame:
        """Load the frozen weight vector from CSV."""
        weights_df = pd.read_csv(self.weights_csv_path)
        return weights_df

    def _load_returns(self) -> pd.DataFrame:
        """Load daily log returns for all assets from parquet."""
        returns_df = pd.read_parquet(self.returns_path)
        returns_df.index = pd.to_datetime(returns_df.index)
        returns_df.sort_index(inplace=True)
        return returns_df

    def _load_spy_close(self) -> pd.Series:
        """Load SPY close prices from parquet as a 1D Series."""
        spy_df = pd.read_parquet(self.spy_prices_path)
        spy_df.index = pd.to_datetime(spy_df.index)
        spy_df.sort_index(inplace=True)
        if "SPY" not in spy_df.columns:
            raise ValueError("benchmark_spy.parquet must contain a 'SPY' column")
        return spy_df.loc[:, ["SPY"]].astype(float).squeeze("columns")

    def _align_dates(self, returns_df: pd.DataFrame, spy_close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Restrict both series to their common trading days."""
        common_dates = returns_df.index.intersection(spy_close.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between returns and SPY benchmark")
        return returns_df.loc[common_dates], spy_close.loc[common_dates]

    def _slice_window(self, returns_df: pd.DataFrame, spy_close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Apply the configured backtest start/end date window."""
        mask = (returns_df.index >= pd.Timestamp(self.start_date)) & (returns_df.index <= pd.Timestamp(self.end_date))
        returns_df = returns_df.loc[mask]
        spy_close = spy_close.loc[mask]
        if len(returns_df) < 2:
            raise ValueError("Backtest window too small after filtering")
        return returns_df, spy_close

    def _align_weights_to_returns(
        self,
        weights_df: pd.DataFrame,
        returns_columns: pd.Index,
    ) -> tuple[list[str], np.ndarray]:
        """Drop tickers not present in returns and renormalize weights."""
        tickers = weights_df["ticker"].astype(str).tolist()
        w = weights_df["weight"].to_numpy(dtype=float)
        if w.ndim != 1 or len(w) == 0:
            raise ValueError("Weights vector is empty or invalid")

        available = [t for t in tickers if t in set(returns_columns)]
        if len(available) == 0:
            raise ValueError("None of the tickers in the weights CSV exist in returns_daily.parquet")
        if len(available) != len(tickers):
            missing = sorted(set(tickers) - set(available))
            print(
                f"Warning: dropping {len(missing)} missing tickers from weights: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )

        idx = [tickers.index(t) for t in available]
        w = w[idx]
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("Weights sum to <= 0 after ticker alignment")
        w = w / w_sum

        return available, w

    def _portfolio_simple_returns(self, returns_df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
        """Convert asset log returns -> simple returns, then compute portfolio return series."""
        log_r = returns_df.to_numpy(dtype=float)
        simple_r = np.expm1(log_r)
        return (simple_r @ w).reshape(-1)

    def _spy_simple_returns(self, spy_close: pd.Series) -> np.ndarray:
        """Compute SPY simple returns from close prices."""
        return spy_close.pct_change().fillna(0.0).to_numpy(dtype=float).reshape(-1)

    def _equity_curve(self, r: np.ndarray, *, initial_equity: float) -> np.ndarray:
        """Build an equity curve from returns, starting at the provided initial equity."""
        return float(initial_equity) * np.cumprod(1.0 + r)

    def _summarize(self, df: pd.DataFrame, *, n_assets: int) -> BacktestSummary:
        n_days = int(len(df))
        ann_factor = float(self.trading_days_per_year)

        port_equity_end = float(df["portfolio_equity"].iloc[-1])
        spy_equity_end = float(df["spy_equity"].iloc[-1])

        # Use returns relative to the configured initial equities.
        port_growth = port_equity_end / self.initial_equity
        spy_growth = spy_equity_end / self.spy_initial_equity

        total_ret_port = port_growth - 1.0
        total_ret_spy = spy_growth - 1.0

        ann_ret_port = float(port_growth ** (ann_factor / n_days) - 1.0)
        ann_ret_spy = float(spy_growth ** (ann_factor / n_days) - 1.0)

        return BacktestSummary(
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            n_days=n_days,
            n_assets=int(n_assets),
            total_return_portfolio=float(total_ret_port),
            total_return_spy=float(total_ret_spy),
            annualized_return_portfolio=float(ann_ret_port),
            annualized_return_spy=float(ann_ret_spy),
        )
