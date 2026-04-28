# Data Pipeline (Extractor)

This folder contains utilities related to data ingestion and preparation.

- `data_loader.py` downloads raw prices and writes processed parquet outputs.
- `data_extractor.py` loads those processed outputs **once** and serves **tensor slices** for backtesting.

---

## What `data_extractor.py` does

`BacktestDataExtractor` is designed for backtesting loops where you need to repeatedly ask:

> “Given all data up to today, give me the arrays I need for my model.”

Key idea:

- It reads from the processed folder (`configs.PROC_DIR`) only **one time** per run.
- It keeps the full `[start_date, final_end_date]` range in memory.
- It maintains a moving **pause date** (the current *as-of* day) and returns only the window up to that pause.

This avoids re-reading Parquet on every simulated trading day.

---

## Inputs it reads (from `data/processed/`)

Produced by running `data_loader.py`:

- `returns_daily.parquet` (log returns)
- `prices_adj_close.parquet` (prices)

The processed directory path is taken from `configs.PROC_DIR`.

Extractor defaults (filenames + missing-value handling) are defined in `configs.py`:

- `EXTRACTOR_RETURNS_FILENAME`
- `EXTRACTOR_PRICES_FILENAME`
- `EXTRACTOR_RETURNS_FILL_VALUE`
- `EXTRACTOR_PRICES_FILL_METHOD`
- `EXTRACTOR_SORT_TICKERS`

---

## Output format (for models)

The extractor returns an `ExtractedWindow` with:

- `returns_history: np.ndarray` with shape `(T, N)` (daily log returns)
- `prices_history: np.ndarray` with shape `(T, N)` (daily close prices)
- `dates: np.ndarray` with shape `(T,)`
- `tickers: list[str]` of length `N`
- `as_of_idx: int` (index of the current decision day in this window)

Ticker ordering convention:

- tickers are sorted **alphabetically**
- both `returns_history` and `prices_history` columns match that order
- model outputs must follow the same order

---

## Typical usage (backtesting loop)

```python
from data_pipeline.data_extractor import BacktestDataExtractor
from models.price_weighted_model import PriceWeightedModel

extractor = BacktestDataExtractor(
    start_date="2020-01-01",
    pause_date="2020-01-31",
    final_end_date="2024-12-31",
)

model = PriceWeightedModel()

# Initial window
window = extractor.extract()
weights = model.predict_weights(
    as_of_idx=window.as_of_idx,
    returns_history=window.returns_history,
    prices_history=window.prices_history,
    dates=window.dates,
    tickers=window.tickers,
)

# Advance one trading day at a time
for _ in range(10):
    window = extractor.add_data(1)
    weights = model.predict_weights(
        as_of_idx=window.as_of_idx,
        returns_history=window.returns_history,
        prices_history=window.prices_history,
        dates=window.dates,
        tickers=window.tickers,
    )
```

---

## Notes (v1 simplifications)

- `prices_history` is forward-filled/back-filled to avoid NaNs.
- `returns_history` fills missing values with `0.0`.

If you prefer a stricter approach (dropping tickers/dates with missing data or returning a mask), we can adjust the extractor later.
