"""
Profile the portfolio optimizer to find performance bottlenecks.

Usage:
    python profile_portfolio.py                 # cProfile (function-level)
    python profile_portfolio.py --line          # line_profiler (line-by-line on hot functions)
    python profile_portfolio.py --both          # run both
"""

import argparse
import cProfile
import pstats
import io
import time

from portfolio.data import load_from_pipeline
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution, PortfolioNSGA2Utils
from nsga2.utils import NSGA2Utils
from nsga2.individual import Individual

# Reduced parameters for profiling (full run takes ~15 min, this ~1-2 min)
PROFILE_GENERATIONS = 20
PROFILE_POPULATION = 30


def setup():
    """Load data and create problem + evolution objects."""
    mean_returns, cov_matrix, std_returns, ticker_names = load_from_pipeline()
    problem = PortfolioProblem(mean_returns, cov_matrix, std_returns)
    evolution = PortfolioEvolution(
        problem,
        num_of_generations=PROFILE_GENERATIONS,
        num_of_individuals=PROFILE_POPULATION,
        use_lou_selection=True,
        use_lou_mutation=True,
        use_lou_init=True,
    )
    return evolution


def run_optimization(evolution):
    """Run the optimizer and return the Pareto front."""
    return evolution.evolve()


def run_cprofile():
    """Profile with cProfile for function-level breakdown."""
    print(f"=== cProfile (pop={PROFILE_POPULATION}, gen={PROFILE_GENERATIONS}) ===")
    evolution = setup()

    profiler = cProfile.Profile()
    profiler.enable()
    pareto_front = run_optimization(evolution)
    profiler.disable()

    # Print top functions by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(40)
    output = stream.getvalue()
    print(output)

    # Also print by total time (self time, excludes subcalls)
    print("\n=== Top functions by self time ===")
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats('tottime')
    stats2.print_stats(30)
    print(stream2.getvalue())

    print(f"Pareto front: {len(pareto_front)} portfolios")
    return output


def run_line_profiler():
    """Profile with line_profiler for line-by-line breakdown of hot functions."""
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("line_profiler not installed. Run: pip install line_profiler")
        return None

    print(f"=== line_profiler (pop={PROFILE_POPULATION}, gen={PROFILE_GENERATIONS}) ===")
    evolution = setup()
    problem = evolution.utils.problem

    lp = LineProfiler()

    # Portfolio objective functions (likely #1 bottleneck)
    lp.add_function(problem._mat_vec)
    lp.add_function(problem._dot)
    lp.add_function(problem._portfolio_variance_value)
    lp.add_function(problem.neg_sharpe_ratio)
    lp.add_function(problem.portfolio_variance)
    lp.add_function(problem.neg_diversification_ratio)
    lp.add_function(problem.calculate_objectives)
    lp.add_function(problem.repair_weights)

    # NSGA-II core (likely #2-3 bottleneck)
    lp.add_function(NSGA2Utils.fast_nondominated_sort)
    lp.add_function(NSGA2Utils.calculate_crowding_distance)

    # Individual dominance check (called inside fast_nondominated_sort)
    lp.add_function(Individual.dominates)

    # Lou 2023 modifications
    lp.add_function(PortfolioNSGA2Utils._tournament)
    lp.add_function(PortfolioNSGA2Utils._biased_sample)
    lp.add_function(PortfolioNSGA2Utils._mutate)
    lp.add_function(PortfolioNSGA2Utils._crossover)
    lp.add_function(PortfolioNSGA2Utils.create_children)

    # Evolution loop
    lp.add_function(evolution.evolve.__func__)

    # Run with profiling
    lp_wrapper = lp(run_optimization)
    pareto_front = lp_wrapper(evolution)

    # Print results
    stream = io.StringIO()
    lp.print_stats(stream=stream)
    output = stream.getvalue()
    print(output)

    print(f"Pareto front: {len(pareto_front)} portfolios")
    return output


def main():
    parser = argparse.ArgumentParser(description="Profile portfolio optimizer")
    parser.add_argument("--line", action="store_true", help="Run line_profiler")
    parser.add_argument("--both", action="store_true", help="Run both cProfile and line_profiler")
    args = parser.parse_args()

    if args.both:
        cprofile_output = run_cprofile()
        print("\n" + "=" * 70 + "\n")
        line_output = run_line_profiler()
    elif args.line:
        line_output = run_line_profiler()
    else:
        cprofile_output = run_cprofile()


if __name__ == "__main__":
    main()
