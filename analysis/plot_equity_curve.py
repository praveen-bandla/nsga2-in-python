from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt 

from configs import ANALYSIS_EQUITY_CURVE_INPUT_CSV, ANALYSIS_EQUITY_CURVE_OUTPUT_PNG

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main() -> None:
    input_csv: Path = ANALYSIS_EQUITY_CURVE_INPUT_CSV
    output_png: Path = ANALYSIS_EQUITY_CURVE_OUTPUT_PNG

    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values("date")

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["date"], df["portfolio_equity"], label="Portfolio", linewidth=2)
    ax.plot(df["date"], df["spy_equity"], label="SPY", linewidth=2)

    ax.set_title("Equity Progression: Portfolio vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Valuation ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)

    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()
