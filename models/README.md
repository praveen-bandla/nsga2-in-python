# Models (v1)

This folder contains **portfolio models/strategies** that can be “plugged into” the backtesting runner.

In **v1**, a model’s job is simple:

- It **looks at historical returns up to a rebalance date**.
- It **outputs a vector of target portfolio weights** across the available tickers.

The backtester will take those weights and compute realized portfolio performance on the *next* period’s returns.

---

## What data does a model see?

Models should only use information available up to the decision time (no look-ahead).

The backtester will load these artifacts created by the data pipeline:

- `data/processed/returns_daily.parquet` — daily **log returns** for each ticker

Some models may also require:

- `data/processed/prices_adj_close.parquet` — daily close prices for each ticker

In **this v1 design**, we assume:

- We use **daily** returns and rebalance/trade **every day**.
- Models do **not** receive any benchmark series as input.

### Core input to models (tensor-based)

Models in this folder should **not** read Parquet files directly.

Instead, the backtesting runner will load the Parquet data, align dates/tickers, handle missing values, then pass a **tensor-like** structure to the model.

Recommended core input:

- `returns_history: np.ndarray`
    - shape: `(T, N)`
    - `T`: number of days available up to (and including) the decision time
    - `N`: number of tickers/assets
    - values: **daily log returns**

Alongside the tensor, the backtester should pass metadata so the model can interpret dimensions:

- `tickers: list[str]` of length `N`
- `dates: np.ndarray` of shape `(T,)` (e.g., `datetime64[D]`) or a list of Python `datetime` objects

If you later use PyTorch/JAX, you can keep the exact same shapes and just swap `np.ndarray` for your preferred tensor type.

### Optional additional input (for some models)

Some strategies need access to the **current price level** (e.g., price-weighted baselines). For those, the backtester should pass:

- `prices_history: np.ndarray`
    - shape: `(T, N)`
    - aligned to the same `dates` and `tickers` as `returns_history`
    - values: close prices (positive floats)

### Important note: log returns

Your pipeline computes log returns:

- log return: `r_log = log(P_t / P_{t-1})`
- simple return: `r_simple = exp(r_log) - 1`

Some models may prefer simple returns. Converting is fine as long as you do it consistently.

---

## What decision is a model making?

A model decides **target weights** `w_t` at each rebalance date/day `t`.

- `w_t` is a 1D vector of floats (length `N`)
- for a long-only, fully-invested portfolio:
  - `w_t >= 0` for all tickers
  - `w_t.sum() == 1.0`

The backtester interprets `w_t` as: “If I could rebalance at day `t`, what fraction of my portfolio value would I allocate to each asset for the next day?”

### Does the model manage cash/shares?

Not in v1.

- There is no explicit cash account, no share counts, and no order execution.
- Rebalancing is implicit: if weights change, we assume we can trade to those new weights at the rebalance time.

This keeps v1 simple and is standard for research backtests.

---

## Minimal model interface (v1)

All models should expose a class with this shape:

- `predict_weights(...)` — required: produce weights for the next day

Recommended interface:

```python
class BaseModel:
    name: str = "base"

    def predict_weights(
        self,
        as_of_idx,
        returns_history,
        prices_history=None,
        dates=None,
        tickers=None,
    ):
        """Return target weights at `as_of_idx` using only data up to that index.

        Parameters
        ----------
        as_of_idx : int
            Index of the current decision day in the `returns_history` time axis.
        returns_history : np.ndarray
            Daily log returns up to and including `as_of_idx`, shape (T, N).
        prices_history : np.ndarray | None
            Optional close prices up to and including `as_of_idx`, shape (T, N).
            Aligned with the same `dates`/`tickers` ordering as `returns_history`.
        dates : array-like | None
            Optional date vector aligned to returns_history.
        tickers : list[str] | None
            Optional ticker list aligned to the N assets.

        Returns
        -------
        np.ndarray
            1D weights vector of length N.
        """
        raise NotImplementedError
```

### Ticker ordering convention (important)

To keep the tensor interface simple and consistent across all models, **we assume assets are always ordered alphabetically by ticker**.

- The backtester will sort tickers alphabetically before creating `returns_history`.
- The model must output weights in **the same alphabetical order**.

If `prices_history` is provided, it uses the same ticker ordering.

If `tickers` is passed in, it is expected to already be sorted.

### Weight output rules (v1)

To keep models interchangeable, `predict_weights` should:

- return weights for **exactly** the `N` assets in `returns_history.shape[1]`
- contain no missing values (`np.isfinite(weights).all()`)
- be deterministic given inputs (unless your backtest intentionally sets a random seed)

---

## Suggested file structure for each model

Create one file per model:

- `models/equal_weight.py`
- `models/mean_variance.py`
- `models/risk_parity.py`
- etc.

Each file should contain:

- one public class (the model)
- small helper functions if needed

Example template:

```python
from __future__ import annotations

import numpy as np


class EqualWeightModel:
    """Baseline: equal-weight across all available tickers."""

    name: str = "equal_weight"

    def predict_weights(
        self,
        as_of_idx: int,
        returns_history: np.ndarray,
        prices_history: np.ndarray | None = None,
        dates=None,
        tickers=None,
    ) -> np.ndarray:
        n = int(returns_history.shape[1])
        if n == 0:
            raise ValueError("No tickers provided")
        w = np.ones(n, dtype=float) / n
        return w
```

---

## How models plug into backtesting

The backtesting runner (planned location: `backtesting/main.py`) will:

1. Load processed **daily** returns data.
2. Split into train/test by config dates.
3. Instantiate your model class.
4. For each trading day in the test window:
    - call `model.predict_weights(as_of_idx=t, returns_history=returns[:t+1], prices_history=prices[:t+1])`
   - compute next-period realized portfolio return from those weights
6. Compare the portfolio equity curve to SPY.

---

## Notes / v1 simplifications

- No transaction costs (optional to add later via turnover).
- No slippage, no execution constraints.
- No leverage or shorting unless explicitly allowed in the backtester + models.
- Use SPY as the benchmark (not “reconstruct S&P 500 weights”).

---

## Recommended first models

To build confidence in the backtester quickly, implement these first:

1. `EqualWeightModel` — sanity baseline.
2. `BuyAndHoldTopNModel` — trivial ranking rule (e.g., based on past momentum) to ensure the whole pipeline works.
3. Your “real” model(s).
