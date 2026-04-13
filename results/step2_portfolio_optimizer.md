# Step 2: Portfolio Optimizer Results

## tldr
- **Algorithm**: Fully optimized NSGA-II (Lou 2023)
- **Modifications**: Refined selection, dynamic mutation, optimized initialization
- **Population**: 100 individuals
- **Generations**: 200
- **Stock universe**: 465 S&P 500 stocks, 2513 trading days (~10 years)
- **Objectives**: Neg Sharpe ratio (min), Portfolio variance (min), Neg diversification ratio (min)
- **Runtime**: 939.94 seconds (pure Python baseline)

## Pareto Front (23 optimal portfolios)

| # | Sharpe | Ann.Vol% | DivRatio | #Stocks |
|---|--------|----------|----------|---------|
| 1 | 0.6436 | 17.26 | 1.9518 | 31 |
| 2 | 0.6274 | 16.32 | 1.9295 | 24 |
| 3 | 0.6550 | 16.74 | 1.9163 | 33 |
| 4 | 0.7592 | 17.17 | 1.9074 | 29 |
| 5 | 0.6827 | 16.85 | 1.9017 | 28 |
| 6 | 0.7669 | 16.38 | 1.8959 | 34 |
| 7 | 0.7992 | 16.79 | 1.8931 | 36 |
| 8 | 0.7396 | 16.33 | 1.8713 | 37 |
| 9 | 0.6324 | 16.22 | 1.8649 | 24 |
| 10 | 0.6520 | 16.04 | 1.8612 | 34 |
| 11 | 0.7779 | 16.50 | 1.8609 | 37 |
| 12 | 0.7738 | 16.37 | 1.8588 | 38 |
| 13 | 0.8477 | 17.37 | 1.8549 | 34 |
| 14 | 0.7441 | 16.27 | 1.8523 | 40 |
| 15 | 0.6926 | 15.89 | 1.8450 | 25 |
| 16 | 0.7293 | 16.21 | 1.8446 | 31 |
| 17 | 0.9297 | 16.66 | 1.8437 | 30 |
| 18 | 0.7679 | 15.71 | 1.8365 | 39 |
| 19 | 0.8870 | 16.38 | 1.8203 | 29 |
| 20 | 0.7653 | 15.45 | 1.8042 | 29 |
| 21 | 0.8271 | 16.08 | 1.7970 | 40 |
| 22 | 0.8478 | 16.08 | 1.7962 | 39 |
| 23 | 0.9246 | 16.21 | 1.7887 | 23 |

## Best Sharpe Portfolio (#17, Sharpe = 0.93)
| Stock | Weight |
|-------|--------|
| LLY | 13.14% |
| CPRT | 3.94% |
| JPM | 3.64% |
| PG | 2.66% |
| KMB | 2.59% |
| DLR | 2.51% |
| WMT | 2.50% |
| ODFL | 2.41% |
| MCK | 2.14% |
| EPAM | 2.03% |

## Pareto Front Tradeoffs
- **High diversification** (#1, DR=1.95): 31 stocks, lower Sharpe (0.64), moderate volatility (17.3%)
- **High Sharpe** (#17, Sharpe=0.93): 30 stocks, moderate diversification (1.84), moderate volatility (16.7%)
- **Low volatility** (#20, Vol=15.5%): 29 stocks, moderate Sharpe (0.77), lower diversification (1.80)

## Comparison to Lou 2023

### Setup Differences
| | Lou 2023 | Our Implementation |
|---|----------|-------------------|
| Stock universe | 26 stocks, various sectors | 465 S&P 500 stocks |
| Objectives | 2: Return (max), Volatility (min) | 3: Sharpe (max), Variance (min), Diversification (max) |
| Population | 3,000 initial → scaled down | 1,000 initial → 100 (10x multiplier) |
| Generations | ~400 | 200 |
| Validation | MCMC 10-year projection | Pareto front analysis |

### Lou's Results (from paper Figs. 1-5)
- **Pareto front range**: Return 0.26-0.40, Volatility 0.20-0.26
- **Sharpe ratios**: ~1.25-1.60 (annualized)
- **Standard NSGA-II**: Clustered in center of front, sparse at extremes
- **Selection optimized**: Better distribution, broader coverage
- **Fully optimized**: Entire front dominates the other two algorithms

### Lou's MCMC 10-Year Projection (Tables I-III)
| Algorithm | P(return > 0) | P(return >= 5x) | P(return >= 10x) |
|-----------|---------------|-----------------|------------------|
| Standard NSGA-II | 94.69% | 31.05% | 11.29% |
| Selection optimized | 94.81% | 34.09% | 14.19% |
| Fully optimized | 97.91% | 50.53% | 24.46% |

### Our Results vs Lou's
- **Sharpe range**: Ours 0.63-0.93 vs Lou's ~1.25-1.60. Lower because 465 stocks dilute concentration compared to Lou's focused 26-stock universe.
- **Volatility range**: Ours 15.5-17.4% vs Lou's 20-26%. Lower because more diversification is possible with 465 stocks.
- **Diversification ratio**: Ours 1.79-1.95 (Lou does not report this — it is our third objective).
- **Portfolio concentration**: Ours 23-40 stocks per portfolio vs Lou's likely fewer. More stocks in the universe allows the optimizer to spread weight.

### Lou's Three Modifications (as implemented)
1. **Refined Selection (Section III-A)**: P(i) = c(i)^b / Σc(j)^b — biases tournament candidate selection toward less crowded solutions. b=0.5 used.
2. **Dynamic Mutation (Section III-B)**: Mutation probability and sigma increase as Pareto front gets denser, with per-generation decay (0.995^gen).
3. **Optimized Initialization (Section III-C)**: 10x population for one generation, select down to standard size via non-dominated sorting + crowding distance.

## Baseline Performance (for optimization journey)
| Metric | Value |
|--------|-------|
| Runtime | 939.94s (~15 min) |
| Pareto front size | 23 portfolios |
| Stock universe | 465 stocks |
| Population x Generations | 100 x 200 |
| Init population (10x) | 1,000 |