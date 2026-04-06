# Group Work README (Data Pipeline)

This repo includes a simple data pipeline in `data_pipeline/data_loader.py` that downloads price data from Yahoo Finance (via `yfinance`), saves raw parquet batches, and writes processed returns/covariance outputs.

## 1) Create + activate the virtual environment (macOS)

From the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick checks:

```bash
which python
python -V
```

To deactivate later:

```bash
deactivate
```

## 2) What the pipeline does

`SP500Pipeline` (in `data_pipeline/data_loader.py`) will:

1. Download adjusted close prices in batches (batch size is controlled by `BATCH_SIZE` in `configs.py`).
2. Save each raw batch into `data/raw/batch_XX.parquet`.
3. Concatenate batches into one price matrix and save it as `data/processed/prices_adj_close.parquet`.
4. Compute:
   - daily log returns → `data/processed/returns_daily.parquet`
   - monthly log returns (month-end) → `data/processed/returns_monthly.parquet`
   - annualized covariance matrices → `data/processed/covariance_daily.parquet` and `data/processed/covariance_monthly.parquet`
5. Download SPY benchmark close prices → `data/processed/benchmark_spy.parquet`

Ticker lists come from `data_pipeline/tickers.json`:
- `SP500_TICKERS` = full list
- `TEST_TICKERS` = small list

Defaults are set in `configs.py`:
- `DEFAULT_TICKERS = SP500_TICKERS`

## 3) Run downloads

### Full dataset (S&P 500 list)

From repo root:

```bash
./venv/bin/python data_pipeline/data_loader.py
```

This uses the default tickers (full S&P 500 list).

### Test run (small ticker list)

```bash
./venv/bin/python data_pipeline/data_loader.py --test
```

This runs the exact same pipeline, but swaps the ticker list to `TEST_TICKERS`.

## 4) Where outputs go

Raw batch parquet files:
- `data/raw/batch_00.parquet`, `data/raw/batch_01.parquet`, ...

Processed parquet files:
- `data/processed/prices_adj_close.parquet`
- `data/processed/returns_daily.parquet`
- `data/processed/returns_monthly.parquet`
- `data/processed/covariance_daily.parquet`
- `data/processed/covariance_monthly.parquet`
- `data/processed/benchmark_spy.parquet`

## 5) Check raw file sizes

```bash
du -sh data/raw
```

Per-file (sorted):

```bash
du -h data/raw/* | sort -h
```
