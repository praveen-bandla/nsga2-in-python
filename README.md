# NSGA-II Portfolio Optimizer

An optimized implementation of NSGA-II for multi-objective portfolio optimization with faster performance and benchmark comparison capabilities.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Improvements](#key-improvements)
- [Setup](#setup)
- [NSGA-II Algorithm](#nsga-ii-algorithm)
- [Data Pipeline](#data-pipeline)
- [Running Optimizations](#running-optimizations)
- [Results & Performance](#results--performance)
- [Authors](#authors)

---

## Overview

This project implements an optimized version of NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective portfolio optimization, based on modifications introduced in the paper "Optimizing Portfolios with Modified NSGA-II Solutions" by Kaiyuan Lou available [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10393316).

### Project Goals

1. **Performance Optimization**: Demonstrate that the optimized algorithm runs significantly faster than the baseline implementation while maintaining solution quality
2. **Benchmark Comparison**: Compare optimized portfolio results against the S&P 500 benchmark to validate practical effectiveness of the modifications

### Components

1. **Optimized NSGA-II Library**: Highly efficient multi-objective optimization algorithm supporting unlimited objectives and dimensions
2. **Data Pipeline**: Automated financial data download and processing with S&P 500 and SPY benchmark tracking
3. **Portfolio Optimizer**: Multi-objective portfolio analysis integrated with the NSGA-II algorithm

---

## Project Structure

The repository splits into the **core pipeline** (everything needed to run the optimizer) and a small **testing & validation** group (used to verify the speed claim and backtest the resulting portfolios). Auxiliary EDA notebooks and demo scripts are not listed here.

Each entry is tagged with its origin:
- **`(upstream)`** — code inherited from the forked NSGA-II repository (see [Authors](#authors)). Not our work.
- **`(upstream, modified)`** — upstream code that we edited for performance or to support Lou's modifications.
- **`(ours)`** — written by us for this project.

```
nsga2-in-python/
│
├── nsga2/                  # base NSGA-II library             (upstream, modified)
├── portfolio/              # 3-obj portfolio + Lou 2023 mods  (ours)
├── data_pipeline/          # S&P 500 + SPY downloader         (ours)
├── data/                   # pipeline output (parquet files)  (ours)
│
├── backtesting/            # walk-forward backtests vs SPY    (ours)
├── models/                 # baseline weighting models        (ours)
├── profile_portfolio.py    # verifies the speedup claim       (ours)
│
├── run_portfolio.py        # main entry point                 (ours)
├── configs.py              # global paths/dates/tickers       (ours)
├── setup_cython.py         # compiles portfolio Cython kernel (ours)
└── requirements.txt
```

> **Start here:** `run_portfolio.py` → uses `portfolio/` → built on `nsga2/` → fed by `data_pipeline/` (output in `data/`).

### Core

`nsga2/` — base NSGA-II library (problem-agnostic). **Origin: upstream, with our modifications.** The package structure and core algorithm are inherited; we vectorized the dominance check, switched individuals to NumPy arrays, and later added a Cython sort kernel in `portfolio/cython_ops.pyx` rather than editing this package directly. Subclassed by `portfolio/`.
- `problem.py` — `Problem` class: stores objective functions, variable count, and per-variable bounds; generates initial individuals by uniform sampling within those bounds.
- `individual.py` — `Individual`: holds a candidate's decision vector, objective values, Pareto rank, crowding distance, and the vectorized Pareto-dominance check used by the sort.
- `population.py` — thin container holding the list of individuals and their partition into Pareto fronts.
- `utils.py` — `NSGA2Utils`: the genetic operators — fast non-dominated sort, crowding-distance computation, tournament selection, SBX crossover, polynomial mutation.
- `evolution.py` — `Evolution`: drives the main loop for *N* generations (sort → crowding → tournament select → crossover + mutate → elitist truncation).

`portfolio/` — portfolio problem definition and Lou's modified evolution loop.
- `problem.py` — `PortfolioProblem` (subclasses `Problem`): defines the three objectives (Sharpe ratio, variance, diversification ratio) and enforces the simplex constraint (weights ≥ 0, sum to 1).
- `optimizer.py` — `PortfolioNSGA2Utils` + `PortfolioEvolution`: extends the base NSGA-II with Lou's three modifications (refined selection, dynamic mutation, optimized initialization).
- `data.py` — `load_from_pipeline()`: reads the pipeline's parquet outputs (returns, covariance, std devs) and packages them for `PortfolioProblem`.
- `cython_ops.pyx` — Cython kernel for the fast non-dominated sort (the main per-generation hot loop). Compiled in-place by `setup_cython.py`.
- `numba_ops.py` — Numba-JIT kernels for SBX crossover and Gaussian mutation (per-gene loops with RNG, where Numba beats NumPy).

`data_pipeline/` — S&P 500 + SPY data download and processing.
- `data_loader.py` — `SP500Pipeline`: sequential batched download of adjusted-close prices from Yahoo Finance.
- `data_loader_threaded.py` — same pipeline, parallelized with Python threads for faster downloads.
- `data_extractor.py` — computes daily/monthly log returns, annualized covariance matrices, asset volatilities, and pulls the SPY benchmark.
- `tickers.json` — `SP500_TICKERS` (full constituents) and `TEST_TICKERS` (small subset for fast iteration).

Top-level scripts and config:
- `run_portfolio.py` — main entry point. Runs the optimizer in `--baseline`, `--selection`, or fully-optimized (default) mode.
- `configs.py` — global paths, date range, batch size, default ticker list.
- `setup_cython.py` — compiles `portfolio/cython_ops.pyx` in place.
- `requirements.txt` — pinned dependencies.
- `data/` — pipeline output: `raw/batch_XX.parquet` and `processed/{prices, returns, covariance, benchmark_spy}.parquet`.

### Testing & validation

- `profile_portfolio.py` — cProfile-based timing of each optimizer mode; used to verify the speedup claim from Lou's modifications.
- `backtesting/` — walk-forward and sliding-window backtests of the optimized weights against SPY.
- `models/` — baseline weighting models (e.g. price-weighted) consumed by `backtesting/` as comparison points.

---

## Key Improvements

This implementation extends the original NSGA-II with targeted enhancements from "Optimizing Portfolios with Modified NSGA-II Solutions" by Kaiyuan Lou:

| Weakness in base NSGA-II | Lou's Fix | Effect |
|-----|-----|-----|
| Only 2 objectives (return, variance) | Add CVaR as 3rd objective | Captures tail/downside risk that variance misses |
| Random uniform initialization | Structured seeding: large population → sort → select best N | Fewer wasted early generations, faster convergence |
| Fixed mutation rate p_m = 1/n | Exponentially decaying p_m and σ (decay ~ 0.995^gen) | Exploration early, precision late—balanced search |
| Symmetric crowding distance | Weighted crowding (higher weight on hardest objective) | Better spread on the hardest-to-cover dimension |
| Uniform tournament candidate selection | Bias candidate selection by crowding distance^b | Favors less crowded regions, improves diversity |
| No benchmark comparison | - | Practical validation against market baseline (Ours) |
| No optimizations | - | Apply advanced optimizations in Python (Ours) |

### Impact of Enhancements

- **Better Risk Modeling**: Three objectives capture realistic portfolio risk including downside tail events
- **Faster Convergence**: Intelligent initialization + adaptive parameters reduce computational waste in early generations
- **Improved Diversity**: Weighted crowding and refined selection maintain frontier spread where it matters most
- **Portfolio-Ready**: Built-in constraints and benchmark comparison for real-world deployment

Based on: [Original NSGA-II Repository](https://github.com/smkalami/nsga2-in-python)

---

## Setup

### 1. Create and Activate Virtual Environment (macOS)

From the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick verification:

```bash
which python
python -V
```

To deactivate later:

```bash
deactivate
```

### 2. Compile cython setup

```bash
python nsga2-in-python/setup_cython.py build_ext --inplace
```

### 3. Download data

#### 3.1. Default batch downloader
```bash
python nsga2-in-python/data_pipeline/data_loader.py
```

#### 3.2. Download with Threading
```bash
python nsga2-in-python/data_pipeline/data_loader_threaded.py
```

---

## NSGA-II Algorithm

### What is NSGA-II?

The Non-dominated Sorting Genetic Algorithm II (NSGA-II) is a fast and elitist multi-objective genetic algorithm that efficiently solves optimization problems with multiple competing objectives. It's particularly suited for portfolio optimization where we need to balance multiple objectives (return, risk, tail-risk) simultaneously.

### Core Algorithm (Base NSGA-II)

The base implementation (`nsga2/` module) includes:
- **Selection**: Tournament selection with configurable probability
- **Crossover**: Simulated Binary Crossover (SBX) with distribution index
- **Mutation**: Polynomial mutation with distribution index
- **Diversity**: Fast non-dominated sorting + crowding distance calculation

### Lou's Portfolio Optimizations

The enhanced version (`portfolio/`) adds three targeted modifications:

1. **Refined Selection**: Bias tournament candidate selection toward less crowded solutions, proportional to crowding distance
2. **Dynamic Mutation**: Adapt mutation probability and step size based on Pareto front density and generation number
3. **Optimized Initialization**: Start with a large population, sort by rank+crowding, then select best N individuals

These work together to balance exploration and exploitation for portfolio optimization specifically.

### Python API

#### Class: Problem

Defined in `nsga2/problem.py`. Defines a multi-objective optimization problem.

**Arguments:**

- `objectives`: List of objective functions to optimize
- `num_of_variables`: Number of decision variables
- `variables_range`: Bounds for each variable `[(min1, max1), (min2, max2), ...]`
- `same_range` (optional, default=`False`): If `True`, all variables share identical bounds
- `expand` (optional, default=`True`): If `True`, functions receive unpacked variables `f(x, y, z)`; if `False`, receive vector `f([x, y, z])`

#### Class: Evolution

Defined in `nsga2/evolution.py`. Executes the NSGA-II algorithm.

**Arguments:**

- `problem`: Problem instance
- `num_of_generations` (default=1000): Number of generations to evolve
- `num_of_individuals` (default=100): Population size
- `num_of_tour_particips` (default=2): Tournament participants
- `tournament_prob` (default=0.9): Tournament selection probability
- `crossover_param` (default=2): SBX distribution index
- `mutation_param` (default=2): Polynomial mutation index

**Methods:**

- `evolve()`: Execute the algorithm, returns best individuals from final generation

### Test Problems

Example test problems (MOP2, MOP4) are included in `moo_test_problems.py` and can be run with `main.py` to verify the algorithm on standard multi-objective optimization benchmarks.

---

## Data Pipeline

### Overview

The `SP500Pipeline` (in `data_pipeline/data_loader.py`) automates the download and processing of S&P 500 financial data from Yahoo Finance, preparing it for multi-objective optimization and benchmark comparison.

### What the Pipeline Does

1. Downloads adjusted close prices in configurable batches (batch size set in `configs.py`)
2. Saves raw batches to `data/raw/batch_XX.parquet`
3. Concatenates into single price matrix: `data/processed/prices_adj_close.parquet`
4. Computes key metrics:
   - Daily log returns → `data/processed/returns_daily.parquet`
   - Monthly log returns → `data/processed/returns_monthly.parquet`
   - Annualized covariance matrices → `data/processed/covariance_daily.parquet`, `covariance_monthly.parquet`
   - Asset volatilities → used for diversification ratio computation
5. Downloads SPY benchmark → `data/processed/benchmark_spy.parquet` (for performance comparison)

These inputs power the three-objective portfolio optimization:
- **Objective 1**: Sharpe Ratio (maximize risk-adjusted return)
- **Objective 2**: Portfolio Variance (minimize risk)
- **Objective 3**: Diversification Ratio (maximize weighted average volatility / portfolio volatility)

### Configuration

**Ticker lists** (`data_pipeline/tickers.json`):
- `SP500_TICKERS`: Full S&P 500 constituents
- `TEST_TICKERS`: Smaller subset for testing

**Global settings** (`configs.py`):
- `DEFAULT_TICKERS`: Default ticker list to use
- `BATCH_SIZE`: Batch size for downloads

### Running Data Downloads

#### Download Full S&P 500 Dataset

```bash
./venv/bin/python data_pipeline/data_loader.py
```

#### Test Run (Small Subset)

```bash
./venv/bin/python data_pipeline/data_loader.py --test
```

### Data Output Structure

**Raw data:**
- `data/raw/batch_00.parquet`, `data/raw/batch_01.parquet`, ...

**Processed data:**
- `data/processed/prices_adj_close.parquet`
- `data/processed/returns_daily.parquet`
- `data/processed/returns_monthly.parquet`
- `data/processed/covariance_daily.parquet`
- `data/processed/covariance_monthly.parquet`
- `data/processed/benchmark_spy.parquet` (SPY benchmark for comparison)

**Check raw data size:**

```bash
du -sh data/raw
du -h data/raw/* | sort -h
```

---

## Running Optimizations

### Portfolio Optimization Modes

The optimizer supports three modes to compare vanilla NSGA-II vs Lou's enhancements:

#### 1. Standard NSGA-II (Baseline)

```bash
./venv/bin/python run_portfolio.py --baseline
```

Runs unmodified NSGA-II with standard tournament selection and fixed parameters.

#### 2. With Selection Optimization

```bash
./venv/bin/python run_portfolio.py --selection
```

Adds Lou's Modification 1: Refined selection biased toward less crowded solutions.

#### 3. Fully Optimized (Default)

```bash
./venv/bin/python run_portfolio.py
```

Enables all three Lou modifications:
- Refined selection (Mod 1)
- Dynamic mutation (Mod 2)
- Optimized initialization (Mod 3)

### Configuration Options

```bash
./venv/bin/python run_portfolio.py --generations 300 --population 150
```

- `--generations`: Number of NSGA-II generations (default: 200)
- `--population`: Population size (default: 100)

### Profiling and Performance Comparison

To benchmark and profile the different modes:

```bash
./venv/bin/python profile_portfolio.py
```

This runs timing comparisons between modes to quantify the performance improvements from Lou's optimizations.

---

## Results & Performance

### Optimization Output

Running any mode produces a Pareto front of non-dominated portfolio solutions displayed in the terminal with:
- Annualized Sharpe Ratio
- Portfolio volatility (annualized %)
- Diversification ratio
- Number of assets held

Example output:
```
  #     Sharpe   Ann.Vol%  DivRatio  #Stocks
  1      1.2345      15.23    2.1456        45
  2      1.1987      14.89    2.3012        52
  ...
```

### Benchmark Comparison

Portfolio solutions can be compared against the S&P 500 benchmark (SPY) using the metrics:

- **Expected Return**: Projected annual return of optimized portfolio
- **Risk (Volatility)**: Standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return relative to risk-free rate
- **Diversification Ratio**: Ratio of weighted average volatility to portfolio volatility
- **Performance vs S&P 500**: Relative Sharpe ratio and volatility

### Mode Comparison

To understand the impact of Lou's modifications, compare outputs from:
- `--baseline`: Baseline performance of standard NSGA-II
- `--selection`: Improvement from refined selection
- (default): Total improvement from all three modifications

The fully optimized version should converge faster and produce more diverse, higher-quality Pareto fronts.

---

## Authors

**Original NSGA-II Implementation:**
- Pham Ngo Gia Bao, Ho Chi Minh University of Technology
- Tram Loi Quan, Ho Chi Minh University of Technology
- A/Prof. Quan Thanh Tho, Ho Chi Minh University of Technology (advisor)
- A/Prof. Akhil Garg, Shantou University (advisor)

**Portfolio Optimization & Performance Improvements:**
- Based on "Optimizing Portfolios with Modified NSGA-II Solutions" by Kaiyuan Lou

**Attribution:**
- Original NSGA-II by Wojciech Reszelewski and Kamil Mielnik
- Forked and enhanced from [smkalami/nsga2-in-python](https://github.com/smkalami/nsga2-in-python)

---
