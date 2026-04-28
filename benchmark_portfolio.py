"""
Optimized benchmark script: Run portfolio optimization directly without subprocess.
Avoids process creation, module re-imports, and JIT recompilation overhead.
"""

import argparse
import math
import statistics
import time
from pathlib import Path
from tqdm import tqdm

# Import once at module level (cached for all runs)
from portfolio.data import load_from_pipeline
from portfolio.problem import PortfolioProblem
from portfolio.optimizer import PortfolioEvolution


def run_optimization_once(generations, population, use_lou_selection, use_lou_mutation, use_lou_init, silent=True):
    """
    Run portfolio optimization once.
    Returns elapsed time (excludes data loading on first run).
    """
    mean_returns, cov_matrix, std_returns, ticker_names = load_from_pipeline()
    
    problem = PortfolioProblem(mean_returns, cov_matrix, std_returns)
    evolution = PortfolioEvolution(
        problem,
        num_of_generations=generations,
        num_of_individuals=population,
        use_lou_selection=use_lou_selection,
        use_lou_mutation=use_lou_mutation,
        use_lou_init=use_lou_init,
    )
    
    # Redirect tqdm output if silent
    if silent:
        old_position = tqdm._instances  # Save tqdm state
        tqdm._instances.clear()
    
    start = time.time()
    pareto_front = evolution.evolve()
    elapsed = time.time() - start
    
    if silent:
        tqdm._instances = old_position
    
    return elapsed, pareto_front


def benchmark(num_runs, mode="", generations=200, population=100):
    """
    Run portfolio optimization multiple times and collect statistics.
    
    Args:
        num_runs: Number of times to run
        mode: "baseline", "selection", or "" (fully optimized)
        generations: Number of generations
        population: Population size
    """
    mode_config = {
        "baseline": (False, False, False),
        "selection": (True, False, False),
        "": (True, True, True),
    }
    
    use_lou_selection, use_lou_mutation, use_lou_init = mode_config.get(mode, (True, True, True))
    
    mode_label = {
        "baseline": "Standard NSGA-II",
        "selection": "Selection Optimization",
        "": "Fully Optimized (Lou 2023)"
    }.get(mode, mode)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {mode_label}")
    print(f"{'='*70}")
    print(f"Generations: {generations}, Population: {population}")
    print(f"Runs: {num_runs}")
    print(f"{'='*70}\n")

    times = []
    pareto_fronts = []
    
    # Pre-warm: Run once (JIT compilation, data loading)
    print("⏳ Pre-warming (JIT compilation, data loading)...")
    try:
        _, _ = run_optimization_once(generations, population, use_lou_selection, use_lou_mutation, use_lou_init, silent=True)
    except Exception as e:
        print(f"❌ Pre-warm failed: {e}")
        return None

    # Actual benchmark runs
    print(f"Running {num_runs} benchmark runs...\n")
    with tqdm(total=num_runs, desc="Benchmarking", unit="run") as pbar:
        for i in range(num_runs):
            try:
                elapsed, pareto_front = run_optimization_once(
                    generations, population, use_lou_selection, use_lou_mutation, use_lou_init, silent=True
                )
                times.append(elapsed)
                pareto_fronts.append(pareto_front)
            except Exception as e:
                print(f"\n❌ Run {i+1} failed: {e}")
            pbar.update(1)

    if not times:
        print("❌ No successful runs!")
        return None

    # Calculate statistics
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

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(times)}/{num_runs} successful runs")
    print(f"{'='*70}")
    print(f"Total Time:    {sum(times):.2f}s")
    print(f'Mean:        {mean:.4f}s')
    print(f'Median:      {median:.4f}s')
    print(f'Std Dev:     {std:.4f}s')
    print(f'CV:          {cv * 100:.2f}%')
    print(f'95% CI:      [{ci_low:.4f}s, {ci_high:.4f}s]')
    print(f'Rel. Error:  {relative_margin_error * 100:.2f}%')
    print(f"{'='*70}\n")

    return {
        'n': n,
        'mean': mean,
        'median': median,
        'std': std,
        'cv': cv,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'relative_margin_error': relative_margin_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fast Portfolio Optimization Benchmark (in-process, no subprocess overhead)"
    )
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of times to run (default: 10)")
    parser.add_argument("--generations", type=int, default=200,
                        help="Number of generations (default: 200)")
    parser.add_argument("--population", type=int, default=100,
                        help="Population size (default: 100)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 3 modes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true", help="Benchmark standard NSGA-II")
    mode.add_argument("--selection", action="store_true", help="Benchmark selection optimization")
    mode.add_argument("--optimized", action="store_true", help="Benchmark fully optimized")
    
    args = parser.parse_args()

    if args.compare:
        # Run all 3 modes
        results = {}
        results["baseline"] = benchmark(args.runs, "baseline", args.generations, args.population)
        results["selection"] = benchmark(args.runs, "selection", args.generations, args.population)
        results["optimized"] = benchmark(args.runs, "", args.generations, args.population)

        # Print comparison
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Mode':<30} {'Avg Time':>12} {'Min':>10} {'Max':>10} {'StdDev':>10}")
        print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        for mode_name in ["baseline", "selection", "optimized"]:
            res = results[mode_name]
            if res:
                label = {
                    "baseline": "Standard NSGA-II",
                    "selection": "Selection Opt",
                    "optimized": "Fully Optimized"
                }[mode_name]
                print(f"{label:<30} {res['mean']:>12.3f}s {res['ci_low']:>10.3f}s {res['ci_high']:>10.3f}s {res['std']:>10.3f}s")

        # Calculate speedup
        if results["baseline"] and results["optimized"]:
            speedup = results["baseline"]["mean"] / results["optimized"]["mean"]
            improvement = (1 - 1/speedup) * 100
            print(f"\n{'Speedup (Optimized vs Baseline):':<40} {speedup:.2f}x ({improvement:+.1f}%)")
        
        if results["selection"] and results["optimized"]:
            speedup_sel = results["selection"]["mean"] / results["optimized"]["mean"]
            print(f"{'Speedup (Optimized vs Selection):':<40} {speedup_sel:.2f}x")

    else:
        # Single mode
        if args.baseline:
            mode = "baseline"
        elif args.selection:
            mode = "selection"
        else:
            mode = ""

        benchmark(args.runs, mode, args.generations, args.population)


if __name__ == "__main__":
    main()