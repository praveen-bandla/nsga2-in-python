"""
Portfolio optimization using Modified NSGA-II (Lou 2023).

Usage:
    python run_portfolio.py                 # fully optimized (all 3 Lou mods)
    python run_portfolio.py --baseline      # standard NSGA-II
    python run_portfolio.py --selection     # selection optimization only
"""

import argparse
import math
import time

from portfolio.data import load_from_pipeline
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution


def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio Optimization with Modified NSGA-II")
    parser.add_argument("--generations", type=int, default=200,
                        help="Number of NSGA-II generations (default: 200)")
    parser.add_argument("--population", type=int, default=100,
                        help="Population size (default: 100)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true",
                      help="Standard NSGA-II (no modifications)")
    mode.add_argument("--selection", action="store_true",
                      help="NSGA-II + selection optimization only")
    return parser.parse_args()


def print_pareto_front(pareto_front, ticker_names):
    """Display Pareto front results."""
    print(f"Pareto Front: {len(pareto_front)} optimal portfolios")
    print(f"{'#':>3} {'Sharpe':>10} {'Ann.Vol%':>10} {'DivRatio':>10} {'#Stocks':>8}")
    print(f"{'-'*3:>3} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}")

    for i, ind in enumerate(pareto_front):
        sharpe = -ind.objectives[0]
        variance = ind.objectives[1]
        div_ratio = -ind.objectives[2]
        ann_sharpe = sharpe * math.sqrt(252)
        ann_vol = math.sqrt(abs(variance) * 252) * 100
        n_stocks = sum(1 for w in ind.features if w > 0.01)
        print(f"{i+1:>3} {ann_sharpe:>10.4f} {ann_vol:>10.2f} {div_ratio:>10.4f} {n_stocks:>8}")

    best_idx = min(range(len(pareto_front)), key=lambda i: pareto_front[i].objectives[0])
    best = pareto_front[best_idx]
    print(f"Best Sharpe portfolio (#{best_idx+1}):")
    weights = sorted(enumerate(best.features), key=lambda x: x[1], reverse=True)
    print("  Top 10 holdings:")
    for idx, w in weights[:10]:
        name = ticker_names[idx] if idx < len(ticker_names) else f"Stock_{idx}"
        print(f"    {name:>8}: {w*100:.2f}%")


def main():
    args = parse_args()

    # Lou 2023 paper tests 3 configurations:
    # 1. Standard NSGA-II
    # 2. NSGA-II + selection optimization
    # 3. Fully optimized (selection + dynamic mutation + optimized init)
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


if __name__ == "__main__":
    main()
