"""
Line-profiler script for run_portfolio.py.

Instruments the hot functions across the NSGA-II + Lou modifications pipeline
so we can see exactly where wall-time goes.

Usage:
    python profile_run.py                  # fully optimized (Lou mods)
    python profile_run.py --baseline       # standard NSGA-II
    python profile_run.py --selection      # selection only
    python profile_run.py --generations 20 --population 50   # quick smoke-test

Output: line-by-line timing printed to stdout, also saved to profile_results.txt
"""

import argparse
import math
import time
from line_profiler import LineProfiler

from portfolio.data import load_from_pipeline
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution, PortfolioNSGA2Utils
from nsga2.utils import NSGA2Utils
from nsga2.individual import Individual


def parse_args():
    parser = argparse.ArgumentParser(description="Line-profile Portfolio NSGA-II")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations (default: 50 for faster profiling)")
    parser.add_argument("--population", type=int, default=100,
                        help="Population size (default: 100)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--selection", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

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

    # ---- Wire up the line profiler ----------------------------------------
    lp = LineProfiler()

    # Core NSGA-II bottlenecks
    lp.add_function(NSGA2Utils.fast_nondominated_sort)
    lp.add_function(NSGA2Utils.calculate_crowding_distance)
    lp.add_function(Individual.dominates)

    # Lou modifications
    lp.add_function(PortfolioNSGA2Utils.create_children)
    lp.add_function(PortfolioNSGA2Utils._biased_sample)
    lp.add_function(PortfolioNSGA2Utils._mutate)
    lp.add_function(PortfolioNSGA2Utils._crossover)
    lp.add_function(PortfolioNSGA2Utils._tournament)

    # Objective evaluation (pure-Python math — likely slow)
    lp.add_function(PortfolioProblem.calculate_objectives)
    lp.add_function(PortfolioProblem._portfolio_variance_value)
    lp.add_function(PortfolioProblem.neg_return)
    lp.add_function(PortfolioProblem.portfolio_variance)
    lp.add_function(PortfolioProblem.neg_diversification_ratio)
    lp.add_function(PortfolioProblem.repair_weights)

    # Top-level evolve loop
    lp.add_function(PortfolioEvolution.evolve)

    # Wrap evolve so the profiler captures it
    profiled_evolve = lp(evolution.evolve)
    # -----------------------------------------------------------------------

    print(f"\nRunning {mode_str} with line profiler...")
    start = time.time()
    pareto_front = profiled_evolve()
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s  |  Pareto front size: {len(pareto_front)}")

    # Print to terminal
    lp.print_stats()

    # Also save to file for easy review
    output_file = "profile_results.txt"
    with open(output_file, "w") as f:
        lp.print_stats(stream=f)
    print(f"\nFull profile saved to {output_file}")


if __name__ == "__main__":
    main()
