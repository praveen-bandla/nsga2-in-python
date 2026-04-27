"""
Benchmark optimized portfolio optimization runtime stability.

Usage:
    python benchmark_optimized.py --runs 10
"""

import sys
sys.dont_write_bytecode = True

import argparse
import math
import statistics
import time

from portfolio.data import load_from_pipeline
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark optimized portfolio runtime')
    parser.add_argument('--runs', type=int, default=10, help='Number of timed runs to perform')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--generations', type=int, default=200)
    parser.add_argument('--population', type=int, default=100)
    return parser.parse_args()


def run_once(problem, generations, population):
    evolution = PortfolioEvolution(
        problem,
        num_of_generations=generations,
        num_of_individuals=population,
        use_lou_selection=True,
        use_lou_mutation=True,
        use_lou_init=True,
    )

    start = time.perf_counter()
    pareto_front = evolution.evolve()
    elapsed = time.perf_counter() - start

    return elapsed, pareto_front


def summarize_runtime(times):
    n = len(times)
    mean = statistics.mean(times)
    median = statistics.median(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0

    ci_half_width = 1.96 * se
    ci_low = mean - ci_half_width
    ci_high = mean + ci_half_width

    cv = std / mean if mean > 0 else float('nan')
    relative_margin_error = ci_half_width / mean if mean > 0 else float('nan')

    return {
        'n': n,
        'mean': mean,
        'median': median,
        'std': std,
        'se': se,
        'cv': cv,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'relative_margin_error': relative_margin_error,
        'min': min(times),
        'max': max(times),
    }


def main():
    args = parse_args()

    print('Loading data once, outside timed runs...')
    mean_returns, cov_matrix, std_returns, _ = load_from_pipeline()
    problem = PortfolioProblem(mean_returns, cov_matrix, std_returns)

    print(f'Optimized NSGA-II benchmark')
    print(f'Runs: {args.runs}')
    print(f'Warmups: {args.warmups}')
    print(f'Population: {args.population}')
    print(f'Generations: {args.generations}')
    print()

    for i in range(args.warmups):
        print(f'Warmup {i + 1}/{args.warmups}...')
        run_once(problem, args.generations, args.population)

    times = []

    print()
    for i in range(args.runs):
        elapsed, _ = run_once(problem, args.generations, args.population)
        times.append(elapsed)
        print(f'Run {i + 1:>3}/{args.runs}: {elapsed:.4f}s')

    summary = summarize_runtime(times)

    print()
    print('=' * 70)
    print('OPTIMIZED RUNTIME SUMMARY')
    print('=' * 70)
    print(f'Runs:                 {summary["n"]}')
    print(f'Mean:                 {summary["mean"]:.4f}s')
    print(f'Median:               {summary["median"]:.4f}s')
    print(f'Std Dev:              {summary["std"]:.4f}s')
    print(f'Min:                  {summary["min"]:.4f}s')
    print(f'Max:                  {summary["max"]:.4f}s')
    print(f'Coefficient of Var:   {summary["cv"] * 100:.2f}%')
    print(f'95% CI:               [{summary["ci_low"]:.4f}s, {summary["ci_high"]:.4f}s]')
    print(f'Relative Error:       {summary["relative_margin_error"] * 100:.2f}%')
    print('=' * 70)

    if summary['cv'] < 0.05 and summary['relative_margin_error'] < 0.05:
        verdict = 'Very stable benchmark'
    elif summary['cv'] < 0.10 and summary['relative_margin_error'] < 0.10:
        verdict = 'Reasonably stable benchmark'
    else:
        verdict = 'Noisy benchmark, increase runs or reduce system noise'

    print(f'Verdict:              {verdict}')


if __name__ == '__main__':
    main()