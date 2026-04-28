from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


@dataclass(frozen=True)
class SeriesSummary:
	start_date: str
	end_date: str
	n_rows: int
	start_equity: float
	end_equity: float
	net_change: float
	net_return: float
	total_gain: float
	total_loss: float
	max_high: float
	max_low: float


def _find_latest_csv(results_dir: Path) -> Path:
	csv_paths = sorted(results_dir.glob("*.csv"))
	if not csv_paths:
		raise FileNotFoundError(f"No .csv files found in {results_dir}")
	return max(csv_paths, key=lambda p: p.stat().st_mtime)


def _compute_summary(
	df: pd.DataFrame,
	*,
	equity_col: str,
	return_col: str,
	initial_equity: float,
) -> SeriesSummary:
	if "date" not in df.columns:
		raise ValueError("Expected a 'date' column in results CSV")
	if equity_col not in df.columns:
		raise ValueError(f"Missing expected equity column: {equity_col}")
	if return_col not in df.columns:
		raise ValueError(f"Missing expected return column: {return_col}")

	df = df[["date", equity_col, return_col]].dropna().copy()
	if df.empty:
		raise ValueError("No non-null rows available to analyze")

	df = df.sort_values("date")

	start_date = str(df["date"].iloc[0])
	end_date = str(df["date"].iloc[-1])
	n_rows = int(len(df))

	equity = pd.to_numeric(df[equity_col], errors="coerce")
	daily_ret = pd.to_numeric(df[return_col], errors="coerce")

	start_equity = float(initial_equity)
	end_equity = float(equity.iloc[-1])
	net_change = end_equity - start_equity
	net_return = (end_equity / start_equity - 1.0) if start_equity != 0 else float("nan")

	equity_change = equity.diff()
	if len(equity_change):
		equity_change.iloc[0] = float(equity.iloc[0]) - start_equity
	total_gain = float(equity_change.where(equity_change > 0, 0.0).sum())
	total_loss = float((-equity_change.where(equity_change < 0, 0.0)).sum())

	max_high = float(equity.max())
	max_low = float(equity.min())

	return SeriesSummary(
		start_date=start_date,
		end_date=end_date,
		n_rows=n_rows,
		start_equity=start_equity,
		end_equity=end_equity,
		net_change=net_change,
		net_return=float(net_return),
		total_gain=total_gain,
		total_loss=total_loss,
		max_high=max_high,
		max_low=max_low,
	)


def _print_summary(name: str, s: SeriesSummary) -> None:
	text = (
		f"{name}\n"
		f"  Period: {s.start_date} -> {s.end_date} ({s.n_rows:,} rows)\n"
		f"  Start equity: {s.start_equity:,.2f}\n"
		f"  End equity:   {s.end_equity:,.2f}\n"
		f"  Net change:   {s.net_change:,.2f}\n"
		f"  Net return:   {s.net_return * 100:,.2f}%\n"
		f"  Total gain:   {s.total_gain:,.2f}\n"
		f"  Total loss:   {s.total_loss:,.2f}\n"
		f"  Max high:     {s.max_high:,.2f}\n"
		f"  Max low:      {s.max_low:,.2f}"
	)
	print(text)


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]

	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		import configs  # type: ignore
	except Exception as e:  # pragma: no cover
		raise RuntimeError(f"Failed to import configs.py from repo root: {repo_root}") from e

	initial_equity = float(getattr(configs, "BACKTEST_INITIAL_EQUITY", 1000.0))
	results_dir = Path(getattr(configs, "BACKTESTING_RESULTS_DIR", repo_root / "backtesting" / "results"))

	latest_csv = _find_latest_csv(results_dir)
	df = pd.read_csv(latest_csv)

	print(f"Loaded: {latest_csv.relative_to(repo_root)}\nInitial equity: {initial_equity:,.2f}")
	if "is_refresh_day" in df.columns:
		refresh_days = int(pd.to_numeric(df["is_refresh_day"], errors="coerce").fillna(False).astype(bool).sum())
		print(f"Refresh days: {refresh_days:,}")
	print()

	portfolio = _compute_summary(
		df,
		equity_col="portfolio_equity",
		return_col="portfolio_return",
		initial_equity=initial_equity,
	)
	_print_summary("Portfolio", portfolio)

	if "spy_equity" in df.columns and "spy_return" in df.columns:
		print()
		spy = _compute_summary(
			df,
			equity_col="spy_equity",
			return_col="spy_return",
			initial_equity=initial_equity,
		)
		_print_summary("SPY", spy)


if __name__ == "__main__":
	main()

