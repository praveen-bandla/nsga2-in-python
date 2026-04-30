"""Sliding-window (expanding history) backtest with quarterly re-optimization.

This script implements a walk-forward / rebalancing backtest:

- Rebalance dates follow the S&P 500 quarterly refresh cadence:
  the 2nd Friday of Mar/Jun/Sep/Dec each year.
- At each rebalance day t_k, we recompute portfolio weights using an
  expanding history window from BACKTEST_START_DATE up to (t_k - 1 trading day)
  to avoid look-ahead.
- We then hold those weights from t_k (inclusive) until the day before the
  next rebalance date.
- Equity is chained: ending equity of one segment becomes the initial equity
  of the next segment.

Outputs:
- Writes one weights CSV per rebalance date under BACKTESTING_SLIDING_WEIGHTS_DIR.
- Writes a single stitched results CSV under BACKTESTING_RESULTS_DIR with a
  boolean `is_refresh_day` column.
"""

from __future__ import annotations
from pathlib import Path
import sys
import csv

import numpy as np
import pandas as pd

# Allow running as a script from repo root while still importing configs
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from configs import *
from backtesting.backtester import PortfolioBacktester
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution


def _second_friday(year: int, month: int) -> pd.Timestamp:
    """Return the 2nd Friday (calendar date) of a given year/month."""
    d0 = pd.Timestamp(year=year, month=month, day=1)
    friday = 4  # Monday=0 ... Friday=4
    days_until_friday = (friday - int(d0.weekday())) % 7
    first_friday = d0 + pd.Timedelta(days=int(days_until_friday))
    second_friday = first_friday + pd.Timedelta(days=7)
    return second_friday.normalize()


def _align_to_trading_day(target: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    if len(trading_days) == 0:
        return None

    target = pd.Timestamp(target).normalize()
    pos = trading_days.searchsorted(target, side="left")
    if pos >= len(trading_days):
        return None
    return pd.Timestamp(trading_days[pos]).normalize()


def compute_refresh_days(
    *,
    start_date: str,
    end_date: str,
    trading_days: pd.DatetimeIndex,
    months: tuple[int, ...],
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    refresh_days: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        for month in months:
            candidate = _second_friday(year, month)
            aligned = _align_to_trading_day(candidate, trading_days)
            if aligned is None:
                continue
            if aligned < start or aligned > end:
                continue
            refresh_days.append(aligned)

    # Ensure unique + sorted
    refresh_days = sorted(set(pd.Timestamp(d).normalize() for d in refresh_days))
    return refresh_days


def _previous_trading_day(target: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    if len(trading_days) == 0:
        return None
    target = pd.Timestamp(target).normalize()
    pos = trading_days.searchsorted(target, side="left") - 1
    if pos < 0:
        return None
    return pd.Timestamp(trading_days[pos]).normalize()


def _optimizer_inputs_from_returns(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if returns_df.empty:
        raise ValueError("Training returns window is empty")

    # Drop tickers with any missing values in the training window.
    clean = returns_df.dropna(axis=1, how="any")

    mean_returns = clean.mean().to_numpy(dtype=float)
    cov_matrix = clean.cov().to_numpy(dtype=float)
    std_returns = clean.std().to_numpy(dtype=float)
    tickers = list(clean.columns)
    return mean_returns, cov_matrix, std_returns, tickers


def _optimize_weights_from_history(returns_history: pd.DataFrame) -> pd.DataFrame:
    mean_returns, cov_matrix, std_returns, tickers = _optimizer_inputs_from_returns(returns_history)

    problem = PortfolioProblem(mean_returns, cov_matrix, std_returns)
    evolution = PortfolioEvolution(
        problem,
        num_of_generations=SLIDING_OPT_GENERATIONS,
        num_of_individuals=SLIDING_OPT_POPULATION,
        use_lou_selection=SLIDING_USE_LOU_SELECTION,
        use_lou_mutation=SLIDING_USE_LOU_MUTATION,
        use_lou_init=SLIDING_USE_LOU_INIT,
    )

    pareto_front = evolution.evolve()

    # Best Sharpe portfolio corresponds to minimal objective[0] (neg_sharpe)
    best_idx = min(range(len(pareto_front)), key=lambda i: pareto_front[i].objectives[0])
    best = pareto_front[best_idx]
    weights = np.asarray(best.features, dtype=float).reshape(-1)

    return pd.DataFrame({"ticker": tickers, "weight": weights})


def _write_weights_csv(weights_df: pd.DataFrame, path: Path) -> None:
    """Write weights to disk in the same (ticker, weight) format as run_portfolio.py."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "weight"])
        for ticker, weight in zip(weights_df["ticker"].astype(str), weights_df["weight"].to_numpy(dtype=float)):
            writer.writerow([ticker, float(weight)])


def main() -> None:
    overall_start = str(BACKTEST_START_DATE)
    overall_end = str(BACKTEST_END_DATE)

    # Load once (we will slice repeatedly for training/holding segments).
    returns_df = pd.read_parquet(BACKTEST_RETURNS_PATH)
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df.sort_index(inplace=True)

    spy_df = pd.read_parquet(BACKTEST_SPY_PRICES_PATH)
    spy_df.index = pd.to_datetime(spy_df.index)
    spy_df.sort_index(inplace=True)
    if "SPY" not in spy_df.columns:
        raise ValueError("benchmark_spy.parquet must contain a 'SPY' column")
    spy_close = spy_df.loc[:, ["SPY"]].astype(float).squeeze("columns")

    # Align to common trading days once (full history).
    common_days = returns_df.index.intersection(spy_close.index)
    if len(common_days) == 0:
        raise ValueError("No overlapping dates between returns and SPY benchmark")
    returns_all = returns_df.loc[common_days]
    spy_all = spy_close.loc[common_days]

    # Slice to the overall backtest window for evaluation.
    start_ts = pd.Timestamp(overall_start)
    end_ts = pd.Timestamp(overall_end)
    mask = (returns_all.index >= start_ts) & (returns_all.index <= end_ts)
    returns_bt = returns_all.loc[mask]
    spy_bt = spy_all.loc[mask]
    trading_days_bt = returns_bt.index
    trading_days_all = returns_all.index

    if len(trading_days_bt) < 2:
        raise ValueError("Backtest window too small after filtering")

    refresh_days = compute_refresh_days(
        start_date=overall_start,
        end_date=overall_end,
        trading_days=trading_days_bt,
        months=tuple(SLIDING_REBALANCE_MONTHS),
    )
    if len(refresh_days) == 0:
        raise ValueError("No refresh days found within the configured backtest window")

    # Enforce a no-look-ahead convention: compute weights using data up to the prior trading day.
    min_train_days = int(SLIDING_MIN_TRAIN_DAYS)

    weights_dir = Path(BACKTESTING_SLIDING_WEIGHTS_DIR)

    # Results output path
    BACKTESTING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = Path(BACKTEST_SLIDING_RESULTS_CSV_PATH)

    segments: list[pd.DataFrame] = []
    current_equity_port = float(BACKTEST_INITIAL_EQUITY)
    current_equity_spy = float(BACKTEST_INITIAL_EQUITY)

    # Optional initial segment: hold from BACKTEST_START_DATE until the day before
    # the first rebalance day, using weights trained on all data up to the prior trading day.
    first_rebalance = refresh_days[0]
    initial_hold_start = pd.Timestamp(overall_start).normalize()
    initial_hold_end = (pd.Timestamp(first_rebalance).normalize() - pd.Timedelta(days=1)).normalize()
    initial_hold_end = min(initial_hold_end, pd.Timestamp(overall_end).normalize())

    if initial_hold_start <= initial_hold_end:
        train_end0 = _previous_trading_day(initial_hold_start, trading_days_all)
        if train_end0 is not None:
            returns_train0 = returns_all.loc[:train_end0]
            if len(returns_train0) >= int(SLIDING_MIN_TRAIN_DAYS):
                weights_df0 = _optimize_weights_from_history(returns_train0)
                weights_path0 = Path(BACKTESTING_SLIDING_WEIGHTS_DIR) / f"weights_{initial_hold_start.date()}.csv"
                _write_weights_csv(weights_df0, weights_path0)

                bt0 = PortfolioBacktester(
                    start_date=str(initial_hold_start.date()),
                    end_date=str(initial_hold_end.date()),
                    initial_equity=current_equity_port,
                    spy_initial_equity=current_equity_spy,
                    trading_days_per_year=int(TRADING_DAYS_PER_YEAR),
                    returns_df=returns_bt,
                    spy_close=spy_bt,
                    weights_df=weights_df0,
                )
                seg0, _ = bt0.run()
                segments.append(seg0)
                current_equity_port = float(seg0["portfolio_equity"].iloc[-1])
                current_equity_spy = float(seg0["spy_equity"].iloc[-1])

    for i, rebalance_day in enumerate(refresh_days):
        hold_start = rebalance_day

        if i < len(refresh_days) - 1:
            next_rebalance = refresh_days[i + 1]
            # Hold until the day before the next rebalance (avoid overlapping dates across segments).
            hold_end = pd.Timestamp(next_rebalance) - pd.Timedelta(days=1)
        else:
            hold_end = pd.Timestamp(overall_end)

        hold_end = min(hold_end.normalize(), pd.Timestamp(overall_end).normalize())
        if hold_start.normalize() > hold_end:
            continue

        train_end = _previous_trading_day(hold_start, trading_days_all)
        if train_end is None:
            # No prior data to train on.
            continue

        returns_train = returns_all.loc[:train_end]
        if len(returns_train) < min_train_days:
            # Skip early quarters until we have enough data.
            continue

        # 1) Optimize weights on the expanding window.
        weights_df = _optimize_weights_from_history(returns_train)

        # 2) Persist weights for auditability.
        weights_path = weights_dir / f"weights_{hold_start.date()}.csv"
        _write_weights_csv(weights_df, weights_path)

        # 3) Realized segment returns/equity using the shared market data.
        bt = PortfolioBacktester(
            start_date=str(hold_start.date()),
            end_date=str(hold_end.date()),
            initial_equity=current_equity_port,
            spy_initial_equity=current_equity_spy,
            trading_days_per_year=int(TRADING_DAYS_PER_YEAR),
            returns_df=returns_bt,
            spy_close=spy_bt,
            weights_df=weights_df,
        )

        seg_df, _ = bt.run()
        segments.append(seg_df)
        current_equity_port = float(seg_df["portfolio_equity"].iloc[-1])
        current_equity_spy = float(seg_df["spy_equity"].iloc[-1])

    if len(segments) == 0:
        raise ValueError(
            "No segments were generated. "
            "This usually means SLIDING_MIN_TRAIN_DAYS is too large for the chosen date window."
        )

    out = pd.concat(segments).sort_index()

    refresh_set = set(pd.Timestamp(d).normalize() for d in refresh_days)
    out["is_refresh_day"] = [pd.Timestamp(d).normalize() in refresh_set for d in out.index]

    out.to_csv(results_path)

    print(f"Wrote sliding-window backtest results: {results_path}")
    print(f"Rebalance points used: {int(out['is_refresh_day'].sum())}")


if __name__ == "__main__":
    main()
