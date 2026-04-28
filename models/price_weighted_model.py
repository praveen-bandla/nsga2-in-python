from __future__ import annotations

import numpy as np


class PriceWeightedModel:
    """Price-weighted baseline.

    Produces weights proportional to the *current* price level:

        w_i(t) = P_i(t) / sum_j P_j(t)

    Notes
    -----
    - Requires `prices_history` to be passed in by the backtester.
    - This is NOT S&P 500 market-cap weighting. It is simply a baseline that
      uses price levels.
    - Assumes ticker ordering is alphabetical and consistent across inputs.
    """

    name: str = "price_weighted"

    def predict_weights(
        self,
        as_of_idx: int,
        returns_history: np.ndarray,
        prices_history: np.ndarray | None = None,
        dates=None,
        tickers=None,
    ) -> np.ndarray:
        if prices_history is None:
            raise ValueError(
                "PriceWeightedModel requires prices_history with shape (T, N)."
            )

        prices_history = np.asarray(prices_history, dtype=float)
        if prices_history.ndim != 2:
            raise ValueError("prices_history must have shape (T, N).")

        if not (0 <= as_of_idx < prices_history.shape[0]):
            raise IndexError("as_of_idx is out of bounds for prices_history.")

        prices_t = prices_history[as_of_idx]

        # Guardrails: treat NaNs/infs as zero and disallow negative prices.
        prices_t = np.where(np.isfinite(prices_t), prices_t, 0.0)
        prices_t = np.clip(prices_t, 0.0, None)

        denom = float(prices_t.sum())
        n_assets = int(prices_t.shape[0])
        if n_assets == 0:
            raise ValueError("No assets provided (N=0).")

        if denom <= 0.0:
            return np.ones(n_assets, dtype=float) / n_assets

        return prices_t / denom
