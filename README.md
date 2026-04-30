# NSGA-II Portfolio Optimizer

An optimized NSGA-II workflow for multi-objective portfolio optimization, data download, and SPY benchmark backtesting.

## Quick Start

If you only want to run the portfolio optimizer and backtest it, start here.

### 1. Create a virtual environment

From the repo root:

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Install from `requirements.txt` after activating the virtual environment.

### 2. Build the Cython extension

For best performance, build the Cython extension before running the optimizer:

```bash
python setup_cython.py build_ext --inplace
```

### 3. Download data

Small test dataset:

```bash
python data_pipeline/data_loader.py --test
```

Full dataset:

```bash
python data_pipeline/data_loader.py
```

Optional threaded downloader:

```bash
python data_pipeline/data_loader_threaded.py
```

### 4. Run the portfolio optimizer

Default fully optimized run:

```bash
python run_portfolio.py
```

Smaller smoke test:

```bash
python run_portfolio.py --generations 20 --population 30
```

This writes the selected portfolio weights to:

```text
backtesting/weights/lou_fixed_portfolio_weights.csv
```

### 5. Run the SPY benchmark backtest

```bash
python backtesting/backtest_runner.py
```

This writes results to:

```text
backtesting/results/fixed_lou_vs_spy.csv
```

### 6. Plot the equity curve

```bash
python analysis/plot_equity_curve.py
```

This writes:

```text
analysis/equity_curve_portfolio_vs_spy.png
```

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Notes](#setup-notes)
- [NSGA-II Algorithm](#nsga-ii-algorithm)
- [Data Pipeline](#data-pipeline)
- [Running Optimizations](#running-optimizations)
- [Backtesting](#backtesting)
- [Results & Performance](#results--performance)
- [Authors](#authors)

---

## Overview

This project implements an optimized version of NSGA-II (Non-dominated Sorting Genetic Algorithm II) for portfolio optimization, based on a fork of the original `smkalami/nsga2-in-python` project and extended with portfolio-specific workflow code.

### Project Goals

1. **Performance optimization**: Speed up the NSGA-II workflow while preserving useful portfolio solutions.
2. **Benchmark comparison**: Compare optimized portfolio weights against the SPY benchmark.
3. **Reproducible workflow**: Provide a repo-level path from raw market data to optimized weights to backtest outputs.

### Components

1. **NSGA-II library**: Base multi-objective optimization primitives in `nsga2/`
2. **Portfolio optimizer**: Portfolio-specific objectives and Lou-style optimizer changes in `portfolio/`
3. **Data pipeline**: Yahoo Finance download and preprocessing in `data_pipeline/`
4. **Backtesting**: Portfolio-vs-SPY evaluation in `backtesting/`

---

## Project Structure

```text
nsga2-in-python/
├── nsga2/                  # Base NSGA-II implementation
├── portfolio/              # Portfolio objectives + optimized evolution loop
├── data_pipeline/          # Data download and preprocessing
├── backtesting/            # Portfolio vs SPY backtests
├── analysis/               # Plotting / analysis scripts
├── models/                 # Baseline model experiments
├── data/                   # Generated parquet outputs
├── run_portfolio.py        # Main optimizer entry point
├── profile_portfolio.py    # Performance profiling
├── configs.py              # Paths, dates, batch size, seeds
├── setup_cython.py         # Cython build step for best performance
└── requirements.txt
```

### Recommended entry points

- `run_portfolio.py`: optimize a portfolio
- `backtesting/backtest_runner.py`: compare saved weights vs SPY
- `analysis/plot_equity_curve.py`: visualize backtest results

---

## Setup Notes

### Cython build

The optimizer can run without the compiled Cython extension, but Quick Start includes it because it improves performance. Run this from the repo root:

```bash
python -m pip install Cython setuptools
python setup_cython.py build_ext --inplace
```

If the extension is not built, the optimizer falls back to the Python implementation.

### Important path note

All commands in this README assume you are already in the repo root. Use:

```bash
python setup_cython.py build_ext --inplace
```

not:

```bash
python nsga2-in-python/setup_cython.py build_ext --inplace
```

---

## NSGA-II Algorithm

### What is NSGA-II?

NSGA-II is a multi-objective evolutionary algorithm that searches for a Pareto front of non-dominated solutions. In this repo it is used to balance competing portfolio objectives rather than optimize a single scalar score.

### Core algorithm

The base implementation in `nsga2/` includes:

- tournament selection
- crossover and mutation operators
- non-dominated sorting
- crowding-distance-based diversity preservation

### Portfolio formulation used in this repo

The current portfolio optimizer in `portfolio/problem.py` minimizes three objectives:

1. **Negative Sharpe ratio**
2. **Portfolio variance**
3. **Negative diversification ratio**

So, in practical terms, the optimizer is trying to:

- maximize Sharpe ratio
- minimize variance
- maximize diversification ratio

### Optimizer changes in `portfolio/`

The portfolio workflow adds three main enhancements:

1. **Refined selection**: biased tournament candidate sampling based on crowding distance
2. **Dynamic mutation**: generation/density-dependent mutation probability and sigma
3. **Optimized initialization**: oversized initial population followed by rank/crowding selection

### About `main.py`

`main.py` is not the maintained entry point for the repo’s current workflow. For a working end-to-end run, use:

- `python data_pipeline/data_loader.py --test`
- `python run_portfolio.py`
- `python backtesting/backtest_runner.py`

---

## Data Pipeline

### Overview

The pipeline downloads Yahoo Finance price data, stores raw parquet batches, and writes processed return/covariance outputs used by the optimizer and backtests.

### What the pipeline writes

Raw outputs:

- `data/raw/batch_00.parquet`, `data/raw/batch_01.parquet`, ...

Processed outputs:

- `data/processed/prices_adj_close.parquet`
- `data/processed/returns_daily.parquet`
- `data/processed/returns_monthly.parquet`
- `data/processed/covariance_daily.parquet`
- `data/processed/covariance_monthly.parquet`
- `data/processed/benchmark_spy.parquet`

### Configuration

Important settings live in `configs.py`, including:

- ticker universe
- date range
- batch size
- output directories
- random seed

---

## Running Optimizations

### Modes

#### 1. Baseline NSGA-II

```bash
python run_portfolio.py --baseline
```

#### 2. Selection-only optimization

```bash
python run_portfolio.py --selection
```

#### 3. Fully optimized mode

```bash
python run_portfolio.py
```

### Common options

```bash
python run_portfolio.py --generations 300 --population 150
```

- `--generations`: number of generations, default `200`
- `--population`: population size, default `100`

### Output

The optimizer prints a Pareto front summary including:

- annualized Sharpe ratio
- annualized volatility
- diversification ratio
- number of stocks held

It also saves the best-Sharpe portfolio weights CSV for backtesting.

### Profiling

```bash
python profile_portfolio.py
```

This profiles the optimizer and helps compare performance across implementations.

---

## Backtesting

### Fixed-weight backtest vs SPY

After generating portfolio weights with `run_portfolio.py`, run:

```bash
python backtesting/backtest_runner.py
```

Outputs:

- CSV results in `backtesting/results/`
- terminal summary for total and annualized returns

### Equity-curve plot

```bash
python analysis/plot_equity_curve.py
```

This reads the configured backtest CSV and produces a PNG plot in `analysis/`.

---

## Results & Performance

### Optimization output

Typical terminal output looks like:

```text
  #     Sharpe   Ann.Vol%  DivRatio  #Stocks
  1      1.2345      15.23    2.1456        45
  2      1.1987      14.89    2.3012        52
  ...
```

### Benchmark comparison

The backtest compares the optimized portfolio against SPY using metrics such as:

- total return
- annualized return
- equity curve progression

### Backtesting (Sliding Window) + Equity Curve

To run the walk-forward (quarterly) **sliding-window** backtest vs SPY (expanding history, re-optimizes weights each refresh date):

```bash
./venv/bin/python backtesting/backtest_sliding_window_runner.py
```

Default outputs:

- Stitched backtest CSV: `backtesting/results/sliding_window_lou_vs_spy.csv`
- One weights CSV per rebalance date: `backtesting/weights/sliding_window/weights_YYYY-MM-DD.csv`

To print a quick summary for the latest CSV in `backtesting/results/`:

```bash
./venv/bin/python analysis/analysis.py
```

To generate an equity curve image (Portfolio vs SPY):

```bash
./venv/bin/python analysis/plot_equity_curve.py
```

This saves `analysis/equity_curve_portfolio_vs_spy.png` by default.

---

## Authors

**Original NSGA-II Implementation:**

- Pham Ngo Gia Bao, Ho Chi Minh University of Technology
- Tram Loi Quan, Ho Chi Minh University of Technology
- A/Prof. Quan Thanh Tho, Ho Chi Minh University of Technology (advisor)
- A/Prof. Akhil Garg, Shantou University (advisor)

**Portfolio optimization workflow:**

- Extended from the upstream NSGA-II codebase for portfolio optimization and benchmarking

**Attribution:**

- Original NSGA-II by Wojciech Reszelewski and Kamil Mielnik
- Forked and enhanced from [smkalami/nsga2-in-python](https://github.com/smkalami/nsga2-in-python)
