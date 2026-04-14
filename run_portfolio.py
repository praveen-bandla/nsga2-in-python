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
    """
    Display the full Pareto front and three representative extreme portfolios.

    In multi-objective optimization there is no single best solution — the
    Pareto front is the answer. Every portfolio on it is optimal in the sense
    that improving one objective requires accepting a worse value on another.
    We highlight the three extremes so the trade-off structure is visible.
    """
    print(f"\nPareto Front: {len(pareto_front)} non-dominated portfolios")
    print(f"(All are equally 'optimal' — choice depends on investor preference)\n")
    print(f"{'#':>3} {'Ann.Ret%':>10} {'Ann.Vol%':>10} {'DivRatio':>10} {'#Stocks':>8}")
    print(f"{'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for i, ind in enumerate(pareto_front):
        ann_ret = -ind.objectives[0] * 252 * 100
        ann_vol = math.sqrt(abs(ind.objectives[1]) * 252) * 100
        div_ratio = -ind.objectives[2]
        n_stocks = sum(1 for w in ind.features if w > 0.01)
        print(f"{i+1:>3} {ann_ret:>10.2f} {ann_vol:>10.2f} {div_ratio:>10.4f} {n_stocks:>8}")

    # Three extremes that anchor the trade-off surface
    extremes = {
        "Max Return    (high return, likely higher risk)":
            max(range(len(pareto_front)), key=lambda i: -pareto_front[i].objectives[0]),
        "Min Variance  (lowest risk, likely lower return)":
            min(range(len(pareto_front)), key=lambda i:  pareto_front[i].objectives[1]),
        "Max Diversity (most diversified by asset vol)":
            min(range(len(pareto_front)), key=lambda i:  pareto_front[i].objectives[2]),
    }

    for label, idx in extremes.items():
        ind = pareto_front[idx]
        ann_ret = -ind.objectives[0] * 252 * 100
        ann_vol = math.sqrt(abs(ind.objectives[1]) * 252) * 100
        div_ratio = -ind.objectives[2]
        print(f"\n--- {label} ---")
        print(f"    Ann. Return: {ann_ret:.2f}%  |  Ann. Vol: {ann_vol:.2f}%  |  Div. Ratio: {div_ratio:.4f}")
        weights = sorted(enumerate(ind.features), key=lambda x: x[1], reverse=True)
        print("    Top 5 holdings:")
        for asset_idx, w in weights[:5]:
            name = ticker_names[asset_idx] if asset_idx < len(ticker_names) else f"Stock_{asset_idx}"
            print(f"      {name:>8}: {w*100:.2f}%")


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
