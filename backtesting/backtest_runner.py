from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script from repo root while still importing configs.
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import configs
from backtesting.backtester import PortfolioBacktester


def main() -> None:
    proc = Path(configs.PROC_DIR)

    returns_path = proc / configs.RETURNS_DAILY_FILENAME
    spy_path = proc / configs.BENCHMARK_SPY_FILENAME
    weights_path = Path(configs.BACKTESTING_WEIGHTS_DIR) / configs.BACKTESTING_WEIGHTS_FILENAME

    results_dir = Path(configs.BACKTESTING_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / configs.BACKTESTING_RESULTS_FILENAME

    bt = PortfolioBacktester(
        returns_path=returns_path,
        spy_prices_path=spy_path,
        weights_csv_path=weights_path,
        start_date=configs.BACKTEST_START_DATE,
        end_date=configs.BACKTEST_END_DATE,
        initial_equity=getattr(configs, "BACKTEST_INITIAL_EQUITY", 1.0),
        trading_days_per_year=getattr(configs, "TRADING_DAYS_PER_YEAR", 252),
    )

    df, summary = bt.run()
    df.to_csv(results_path)

    print(f"Wrote backtest results: {results_path}")
    print(f"Total return (Lou portfolio): {summary.total_return_portfolio:.2%}")
    print(f"Total return (SPY benchmark): {summary.total_return_spy:.2%}")
    print(f"Annualized return (Lou portfolio): {summary.annualized_return_portfolio:.2%}")
    print(f"Annualized return (SPY benchmark): {summary.annualized_return_spy:.2%}")


if __name__ == "__main__":
    main()
