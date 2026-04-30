from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script from repo root while still importing configs
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from configs import *
from backtesting.backtester import PortfolioBacktester


def main() -> None:
    BACKTESTING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bt = PortfolioBacktester(
        returns_path=BACKTEST_RETURNS_PATH,
        spy_prices_path=BACKTEST_SPY_PRICES_PATH,
        weights_csv_path=BACKTEST_WEIGHTS_CSV_PATH,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        initial_equity=BACKTEST_INITIAL_EQUITY,
        trading_days_per_year=TRADING_DAYS_PER_YEAR
    )

    df, summary = bt.run()
    df.to_csv(BACKTEST_RESULTS_CSV_PATH)

    print(f"Wrote backtest results: {BACKTEST_RESULTS_CSV_PATH}")
    print(f"Total return (Lou portfolio): {summary.total_return_portfolio:.2%}")
    print(f"Total return (SPY benchmark): {summary.total_return_spy:.2%}")
    print(f"Annualized return (Lou portfolio): {summary.annualized_return_portfolio:.2%}")
    print(f"Annualized return (SPY benchmark): {summary.annualized_return_spy:.2%}")


if __name__ == "__main__":
    main()
