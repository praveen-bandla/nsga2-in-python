from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import ANALYSIS_EQUITY_CURVE_INPUT_CSV, ANALYSIS_EQUITY_CURVE_OUTPUT_PNG


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input CSV: {missing}. Found: {list(df.columns)}")


def main() -> None:
    input_csv: Path = ANALYSIS_EQUITY_CURVE_INPUT_CSV
    output_png: Path = ANALYSIS_EQUITY_CURVE_OUTPUT_PNG

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {input_csv}. "
            "Run the sliding-window backtest first, or update ANALYSIS_EQUITY_CURVE_INPUT_CSV in configs.py."
        )

    df = pd.read_csv(input_csv, parse_dates=["date"])
    _require_columns(df, ["date", "portfolio_equity", "spy_equity"])

    df = df.sort_values("date")

    # Local import so the repo can still be used without matplotlib unless this script is run.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["date"], df["portfolio_equity"], label="Portfolio", linewidth=2)
    ax.plot(df["date"], df["spy_equity"], label="SPY", linewidth=2)

    ax.set_title("Equity Progression: Portfolio vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)

    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()
