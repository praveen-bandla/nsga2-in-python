import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np

import configs
from portfolio.data import load_from_pipeline
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Portfolio Optimization with Modified NSGA-II"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Number of NSGA-II generations (default: 200)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true", help="Standard NSGA-II")
    mode.add_argument(
        "--selection", action="store_true", help="NSGA-II + selection optimization only"
    )
    return parser.parse_args()


def print_pareto_front(pareto_front, ticker_names):
    print(f"Pareto Front: {len(pareto_front)} optimal portfolios")
    print(f"{'#':>3} {'Sharpe':>10} {'Ann.Vol%':>10} {'DivRatio':>10} {'#Stocks':>8}")

    for i, ind in enumerate(pareto_front):
        sharpe = -ind.objectives[0]
        variance = ind.objectives[1]
        div_ratio = -ind.objectives[2]
        ann_sharpe = sharpe * math.sqrt(252)
        ann_vol = math.sqrt(abs(variance) * 252) * 100
        n_stocks = sum(1 for w in ind.features if w > 0.01)
        print(
            f"{i + 1:>3} {ann_sharpe:>10.4f} {ann_vol:>10.2f} {div_ratio:>10.4f} {n_stocks:>8}"
        )

    best_idx = min(
        range(len(pareto_front)), key=lambda i: pareto_front[i].objectives[0]
    )
    best = pareto_front[best_idx]
    weights = sorted(enumerate(best.features), key=lambda x: x[1], reverse=True)

    print(f"Best Sharpe portfolio (#{best_idx + 1}):")
    print("  Top 10 holdings:")
    for idx, w in weights[:10]:
        name = ticker_names[idx] if idx < len(ticker_names) else f"Stock_{idx}"
        print(f"    {name:>8}: {w * 100:.2f}%")


def main():
    args = parse_args()

    seed = getattr(configs, "RANDOM_SEED", None)
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    if args.baseline:
        lou_sel, lou_mut, lou_init = False, False, False
        mode_str = "Standard NSGA-II"
    elif args.selection:
        lou_sel, lou_mut, lou_init = True, False, False
        mode_str = "NSGA-II + selection optimization"
    else:
        lou_sel, lou_mut, lou_init = True, True, True
        mode_str = "Fully optimized NSGA-II (Lou 2023)"

    print(f"Mode: {mode_str}")
    print(f"Population: {args.population}, Generations: {args.generations}")

    mean_returns, cov_matrix, std_returns, ticker_names = load_from_pipeline()

    problem = PortfolioProblem(mean_returns, cov_matrix, std_returns)
    evolution = PortfolioEvolution(
        problem,
        num_of_generations=args.generations,
        num_of_individuals=args.population,
        use_lou_selection=lou_sel,
        use_lou_mutation=lou_mut,
        use_lou_init=lou_init,
    )

    print(f"Running {mode_str}...")
    start_time = time.time()
    pareto_front = evolution.evolve()
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    print_pareto_front(pareto_front, ticker_names)

    best_idx = min(
        range(len(pareto_front)), key=lambda i: pareto_front[i].objectives[0]
    )
    best = pareto_front[best_idx]

    weights_dir = Path(
        getattr(configs, "BACKTESTING_WEIGHTS_DIR", Path("backtesting") / "weights")
    )
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / getattr(
        configs,
        "BACKTESTING_WEIGHTS_FILENAME",
        "lou_fixed_portfolio_weights.csv",
    )

    with weights_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "weight"])
        for ticker, weight in zip(ticker_names, best.features, strict=True):
            writer.writerow([ticker, float(weight)])

    print(f"Saved frozen portfolio weights to: {weights_path}")


if __name__ == "__main__":
    main()
