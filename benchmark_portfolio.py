"""
Benchmark script: Run run_portfolio.py multiple times and collect statistics.
Shows progress bar and prints average, min, max execution times.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm


def run_portfolio_once(args_list):
    """Run run_portfolio.py once with given arguments. Returns elapsed time."""
    cmd = [sys.executable, "run_portfolio.py"] + args_list
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n⚠️  Error in run: {result.stderr}")
            return None
        return elapsed
    except subprocess.TimeoutExpired:
        print("\n⚠️  Run timed out (>600s)")
        return None
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        return None


def benchmark(num_runs, mode="", generations=None, population=None):
    """
    Run portfolio optimization multiple times and collect statistics.
    
    Args:
        num_runs: Number of times to run
        mode: "" (default/fully optimized), "--baseline", or "--selection"
        generations: Optional number of generations
        population: Optional population size
    """
    args = []
    if mode:
        args.append(mode)
    if generations:
        args.extend(["--generations", str(generations)])
    if population:
        args.extend(["--population", str(population)])

    mode_label = {
        "": "Fully Optimized (Lou 2023)",
        "--baseline": "Standard NSGA-II",
        "--selection": "Selection Optimization"
    }.get(mode, mode)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {mode_label}")
    print(f"{'='*60}")
    if generations:
        print(f"Generations: {generations}")
    if population:
        print(f"Population: {population}")
    print(f"Runs: {num_runs}")
    print(f"{'='*60}\n")

    times = []
    with tqdm(total=num_runs, desc="Running", unit="run") as pbar:
        for i in range(num_runs):
            elapsed = run_portfolio_once(args)
            if elapsed is not None:
                times.append(elapsed)
            pbar.update(1)

    if not times:
        print("❌ No successful runs!")
        return None

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = sorted(times)[len(times) // 2]
    total_time = sum(times)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(times)}/{num_runs} successful runs")
    print(f"{'='*60}")
    print(f"Total Time:    {total_time:.2f}s")
    print(f"Average Time:  {avg_time:.2f}s")
    print(f"Median Time:   {median_time:.2f}s")
    print(f"Min Time:      {min_time:.2f}s")
    print(f"Max Time:      {max_time:.2f}s")
    print(f"Std Dev:       {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f}s")
    print(f"{'='*60}\n")

    return {
        "runs": len(times),
        "total": total_time,
        "avg": avg_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "times": times
    }


def main():
    parser = argparse.ArgumentParser(description="Portfolio Optimization Benchmark")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of times to run (default: 3)")
    parser.add_argument("--generations", type=int, default=None,
                        help="Number of generations (default: use run_portfolio.py default)")
    parser.add_argument("--population", type=int, default=None,
                        help="Population size (default: use run_portfolio.py default)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 3 modes (baseline, selection, optimized)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true", help="Benchmark standard NSGA-II")
    mode.add_argument("--selection", action="store_true", help="Benchmark selection optimization")
    mode.add_argument("--optimized", action="store_true", help="Benchmark fully optimized")
    
    args = parser.parse_args()

    # Check if run_portfolio.py exists
    if not Path("run_portfolio.py").exists():
        print("❌ Error: run_portfolio.py not found in current directory")
        sys.exit(1)

    if args.compare:
        # Run all 3 modes
        results = {}
        results["baseline"] = benchmark(args.runs, "--baseline", args.generations, args.population)
        results["selection"] = benchmark(args.runs, "--selection", args.generations, args.population)
        results["optimized"] = benchmark(args.runs, "", args.generations, args.population)

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Mode':<25} {'Avg Time':>12} {'Min':>10} {'Max':>10}")
        print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10}")
        
        for mode_name in ["baseline", "selection", "optimized"]:
            res = results[mode_name]
            if res:
                label = {
                    "baseline": "Standard NSGA-II",
                    "selection": "Selection Opt",
                    "optimized": "Fully Optimized"
                }[mode_name]
                print(f"{label:<25} {res['avg']:>12.2f}s {res['min']:>10.2f}s {res['max']:>10.2f}s")

        # Calculate speedup
        if results["baseline"] and results["optimized"]:
            speedup = results["baseline"]["avg"] / results["optimized"]["avg"]
            print(f"\n{'Speedup (Optimized vs Baseline):':<40} {speedup:.2f}x")

    else:
        # Single mode
        if args.baseline:
            mode = "--baseline"
        elif args.selection:
            mode = "--selection"
        else:
            mode = ""

        benchmark(args.runs, mode, args.generations, args.population)


if __name__ == "__main__":
    main()