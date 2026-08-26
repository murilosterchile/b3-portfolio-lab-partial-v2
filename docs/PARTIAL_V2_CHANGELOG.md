# Partial v2 — test build

This is an intermediate research build intended for local validation before the next full release.
It does **not** claim guaranteed investment returns.

## What changed

### Data pipeline

- Added automated B3 corporate-action ingestion for cash dividends/JCP, splits, reverse splits and
  bonus shares, with raw-response caching and retry/backoff.
- Added a separate adjusted analytical-price layer. Raw COTAHIST remains immutable.
- Invalid/non-positive raw closes are quarantined instead of being silently converted to zero.
- Added official CVM CAD + FCA ingestion to construct a historical ticker -> CNPJ -> `CD_CVM`
  registry, including historical/delisted names when FCA provides them.
- Fundamental joins remain point-in-time (`DT_RECEB <= trade_date`).
- Added an end-to-end `refresh-research-data` command so the research dataset can be rebuilt without
  manually maintaining the ticker/CVM bridge in the normal path.

### Features and ML

- Added gross profitability and balance-sheet-strength features from point-in-time CVM statements.
- Added same-date cross-sectional ranks for fundamental descriptors.
- Preserved training-only preprocessing and yearly walk-forward evaluation.
- Kept LightGBM + CatBoost + Ridge; this build does not add a neural network merely for complexity.

### Quantitative selection and risk

- Multifactor score now combines momentum, low volatility and point-in-time quality/profitability.
- QKP pair interaction is covariance-like: correlation is scaled by candidate volatility rather
  than penalizing two pairs equally just because their correlation is equal.
- Exact QKP semantics and the independent native reference solver are unchanged.
- Added a rank buffer/hysteresis option to reduce unnecessary boundary turnover.

### Backtest

- Automatically prefers total-return-adjusted analytical prices when they are available.
- Compares ML-only, momentum, multifactor and `research_v2` on the same OOS dates.
- Writes `data/gold/backtests/comparison.json`.
- Uses a fixed research gate; failed OOS performance remains a failure rather than triggering
  automatic retuning of the same holdout.

## Verification performed in the build environment

- `python3 -m compileall -q packages services scripts`: PASS.
- CMake Release build of `native/qkp`: PASS.
- `native/qkp/build/qkp_tests`: PASS (`qkp_tests: ok`).

The build environment used to package this ZIP did not contain Docker, Polars or PyArrow and did not
have network package installation available. Therefore the full Python `pytest` suite and a fresh
real-data B3 backtest **must be run locally** before treating this partial build as validated.

## Recommended first test

```bash
cp .env.example .env
make test
make refresh-research-data START_YEAR=2011 END_YEAR=2026
make backtest
cat data/gold/backtests/comparison.json
```

For a faster first pass when the raw COTAHIST/CVM files are already present, reuse the existing
`data/` directory and run the new corporate-action/CVM-registry steps followed by fundamentals and
backtest.
