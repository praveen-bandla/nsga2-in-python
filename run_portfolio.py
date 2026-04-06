"""
Run portfolio optimization using Modified NSGA-II (Lou 2023).

Usage:
    python run_portfolio.py --num-stocks 10 --generations 20 --population 30      
    python run_portfolio.py                                                       
    python run_portfolio.py --real-data
    python run_portfolio.py --baseline

This script:
    1. Loads or generates stock return data
    2. Sets up the 3-objective portfolio problem (Sharpe, Variance, Diversification)
    3. Runs the Modified NSGA-II with Lou 2023 improvements
    4. Displays the Pareto-optimal portfolio set
"""

import argparse
import math
import time

from portfolio.data import (
    DEFAULT_TICKERS,
    download_stock_data,
    compute_returns,
    compute_statistics,
    generate_synthetic_data,
)
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution


def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio Optimization with Modified NSGA-II")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real Yahoo Finance data instead of synthetic")
    parser.add_argument("--num-stocks", type=int, default=50,
                        help="Number of stocks for synthetic data (default: 50)")
    parser.add_argument("--generations", type=int, default=200,
                        help="Number of NSGA-II generations (default: 200)")
    parser.add_argument("--population", type=int, default=100,
                        help="Population size (default: 100)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable Lou 2023 adaptive mutation")
    parser.add_argument("--no-smart-init", action="store_true",
                        help="Disable Lou 2023 smart initialization")
    parser.add_argument("--no-refined-selection", action="store_true",
                        help="Disable Lou 2023 refined selection")
    parser.add_argument("--baseline", action="store_true",
                        help="Run standard NSGA-II (disable all Lou 2023 mods)")
    return parser.parse_args()


def print_pareto_front(pareto_front, num_assets, ticker_names=None):
    """Display Pareto front results."""
    print(f"Pareto Front: {len(pareto_front)} optimal portfolios")

    # Annualize for display (assuming daily data, 252 trading days)
    print(f"\n{'#':>3} {'Sharpe':>10} {'Ann.Vol%':>10} {'DivRatio':>10} {'#Stocks':>8}")
    print(f"{'-'*3:>3} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}")

    for i, ind in enumerate(pareto_front):
        sharpe = -ind.objectives[0]  # Negate back
        variance = ind.objectives[1]
        div_ratio = -ind.objectives[2]  # Negate back

        ann_sharpe = sharpe * math.sqrt(252)
        ann_vol = math.sqrt(variance * 252) * 100  # percentage
        n_stocks = sum(1 for w in ind.features if w > 0.01)  # >1% weight

        print(f"{i+1:>3} {ann_sharpe:>10.4f} {ann_vol:>10.2f} {div_ratio:>10.4f} {n_stocks:>8}")

    # Show the best portfolio by Sharpe
    best_sharpe_idx = min(range(len(pareto_front)),
                         key=lambda i: pareto_front[i].objectives[0])
    best = pareto_front[best_sharpe_idx]
    print(f"\nBest Sharpe portfolio (#{best_sharpe_idx+1}):")

    # Top 10 holdings
    weights = list(enumerate(best.features))
    weights.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top 10 holdings:")
    for idx, w in weights[:10]:
        name = ticker_names[idx] if ticker_names else f"Stock_{idx}"
        print(f"    {name:>8}: {w*100:.2f}%")


def main():
    args = parse_args()

    # Determine Lou 2023 modification flags
    if args.baseline:
        adaptive = False
        smart_init = False
        refined = False
        mode_str = "Standard NSGA-II (baseline)"
    else:
        adaptive = not args.no_adaptive
        smart_init = not args.no_smart_init
        refined = not args.no_refined_selection
        mods = []
        if adaptive:
            mods.append("adaptive mutation")
        if smart_init:
            mods.append("smart init")
        if refined:
            mods.append("refined selection")
        mode_str = f"Modified NSGA-II (Lou 2023: {', '.join(mods)})" if mods else "Standard NSGA-II"

    print(f"Mode: {mode_str}")
    print(f"Population: {args.population}, Generations: {args.generations}")

    # Load data
    ticker_names = None
    if args.real_data:
        prices = download_stock_data()
        returns = compute_returns(prices)
        mean_returns, cov_matrix, std_returns = compute_statistics(returns)
        ticker_names = list(prices.columns)
        num_assets = len(ticker_names)
        print(f"Using {num_assets} real stocks")
    else:
        num_assets = args.num_stocks
        mean_returns, cov_matrix, std_returns = generate_synthetic_data(num_stocks=num_assets)
        ticker_names = [f"S{i:02d}" for i in range(num_assets)]
        print(f"Using {num_assets} synthetic stocks")

    # Set up problem
    problem = PortfolioProblem(mean_returns, cov_matrix, std_returns)

    # Run optimizer
    evolution = PortfolioEvolution(
        problem,
        num_of_generations=args.generations,
        num_of_individuals=args.population,
        adaptive_mutation=adaptive,
        smart_init=smart_init,
        refined_selection=refined,
    )

    print(f"\nRunning {mode_str}...")
    start_time = time.time()
    pareto_front = evolution.evolve()
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")

    print_pareto_front(pareto_front, num_assets, ticker_names)


if __name__ == "__main__":
    main()
