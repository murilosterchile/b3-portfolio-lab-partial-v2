from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from portfolio_core.data_quality import DataQualityError, validate_prices
from portfolio_core.quant.risk import covariance_to_correlation, ledoit_wolf_covariance


def historical_correlation(
    *,
    tickers: list[str],
    data_dir: str,
    lookback_observations: int = 252,
) -> np.ndarray | None:
    """Estimate a shrinkage correlation matrix from locally ingested B3 data.

    Returns None when the local history is not sufficient. No network call occurs in API requests.
    """
    root = Path(data_dir) / "silver" / "b3_prices"
    parts = sorted(root.glob("year=*/part-000.parquet"))
    if not parts or len(tickers) < 2:
        return None
    try:
        frame = (
            pl.scan_parquet([str(p) for p in parts])
            .filter(pl.col("ticker").is_in(tickers))
            .select("trade_date", "ticker", "close")
            .collect()
            .sort(["trade_date", "ticker"])
        )
        if frame.is_empty():
            return None
        validate_prices(frame, context="historical correlation prices")
        wide = frame.pivot(index="trade_date", on="ticker", values="close").sort("trade_date")
        missing = [t for t in tickers if t not in wide.columns]
        if missing:
            return None
        values = (
            wide.select(tickers)
            .fill_null(strategy="forward")
            .tail(lookback_observations + 1)
            .to_numpy()
        )
        if values.shape[0] < 80:
            return None
        returns = values[1:] / values[:-1] - 1.0
        # Rows before a ticker's first observation cannot be marked and are not estimator inputs.
        complete = returns[np.all(np.isfinite(returns), axis=1)]
        if complete.shape[0] < 60:
            return None
        return covariance_to_correlation(ledoit_wolf_covariance(complete))
    except DataQualityError:
        raise
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None
