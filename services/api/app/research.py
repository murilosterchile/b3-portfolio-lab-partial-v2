from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from portfolio_core.data_quality import DataQualityError, validate_prices
from portfolio_core.quant.risk import covariance_to_correlation, ledoit_wolf_covariance


def historical_correlation(
    *,
    tickers: list[str],
    data_dir: str,
    as_of: date,
    lookback_observations: int = 252,
) -> np.ndarray | None:
    """Estimate a PIT shrinkage correlation matrix using only prices <= as_of."""
    adjusted_root = Path(data_dir) / "silver" / "b3_prices_adjusted"
    raw_root = Path(data_dir) / "silver" / "b3_prices"
    root = adjusted_root if list(adjusted_root.glob("year=*/part-000.parquet")) else raw_root
    parts = sorted(root.glob("year=*/part-000.parquet"))
    if not parts or len(tickers) < 2:
        return None
    try:
        scan = pl.scan_parquet([str(p) for p in parts]).filter(
            pl.col("ticker").is_in(tickers) & (pl.col("trade_date") <= pl.lit(as_of))
        )
        schema_names = scan.collect_schema().names()
        price_column = "adjusted_close" if "adjusted_close" in schema_names else "close"
        selected_columns = ["trade_date", "ticker", "close"]
        if price_column != "close":
            selected_columns.append(price_column)
        frame = scan.select(selected_columns).collect().sort(["trade_date", "ticker"])
        if frame.is_empty():
            return None
        validate_prices(frame, context="historical correlation prices")
        if price_column == "adjusted_close":
            invalid = frame.filter(
                pl.col(price_column).is_null()
                | ~pl.col(price_column).is_finite()
                | (pl.col(price_column) <= 0)
            )
            if not invalid.is_empty():
                raise DataQualityError("invalid adjusted prices in PIT covariance window")
        wide = frame.pivot(index="trade_date", on="ticker", values=price_column).sort("trade_date")
        missing = [ticker for ticker in tickers if ticker not in wide.columns]
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
        complete = returns[np.all(np.isfinite(returns), axis=1)]
        if complete.shape[0] < 60:
            return None
        return covariance_to_correlation(ledoit_wolf_covariance(complete))
    except DataQualityError:
        raise
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None
