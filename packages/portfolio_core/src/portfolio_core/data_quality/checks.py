from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl


class DataQualityError(ValueError):
    """Raised when an analytical result would use invalid or ambiguous data."""


def _preview(frame: pl.DataFrame, columns: Sequence[str], *, limit: int = 10) -> str:
    available = [column for column in columns if column in frame.columns]
    return repr(frame.select(available).head(limit).to_dicts())


def invalid_price_rows(frame: pl.DataFrame, *, price_column: str = "close") -> pl.DataFrame:
    if price_column not in frame.columns:
        raise DataQualityError(f"prices missing required column: {price_column}")
    return frame.filter(
        pl.col(price_column).is_null()
        | pl.col(price_column).is_nan()
        | pl.col(price_column).is_infinite()
        | (pl.col(price_column) <= 0)
    )


def validate_unique_rows(
    frame: pl.DataFrame,
    keys: Sequence[str],
    *,
    context: str,
) -> None:
    missing = set(keys) - set(frame.columns)
    if missing:
        raise DataQualityError(f"{context} missing uniqueness keys: {sorted(missing)}")
    duplicates = frame.group_by(list(keys)).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise DataQualityError(
            f"{context} has duplicate rows for {list(keys)}: "
            f"{_preview(duplicates, [*keys, 'len'])}"
        )


def validate_prices(frame: pl.DataFrame, *, context: str = "prices") -> None:
    required = {"ticker", "trade_date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise DataQualityError(f"{context} missing required columns: {sorted(missing)}")
    validate_unique_rows(frame, ["ticker", "trade_date"], context=context)
    invalid = invalid_price_rows(frame)
    if not invalid.is_empty():
        raise DataQualityError(
            f"{context} contains {invalid.height} non-finite or non-positive closes: "
            f"{_preview(invalid, ['ticker', 'trade_date', 'close'])}"
        )


def validate_finite_array(values: np.ndarray, *, context: str) -> None:
    array = np.asarray(values, dtype=float)
    invalid = np.argwhere(~np.isfinite(array))
    if invalid.size:
        preview = invalid[:10].tolist()
        raise DataQualityError(
            f"{context} contains {len(invalid)} NaN/Inf values at indices {preview}"
        )


def validate_correlation_matrix(correlation: np.ndarray, *, context: str = "correlation") -> None:
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DataQualityError(f"{context} must be a square matrix, got shape={matrix.shape}")
    validate_finite_array(matrix, context=context)
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise DataQualityError(f"{context} must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise DataQualityError(f"{context} diagonal must be approximately 1")
    if np.any(matrix < -1.0 - 1e-10) or np.any(matrix > 1.0 + 1e-10):
        raise DataQualityError(f"{context} entries must be in [-1, 1]")


def validate_fundamentals_point_in_time(
    frame: pl.DataFrame,
    *,
    prediction_column: str = "trade_date",
    received_column: str = "DT_RECEB",
) -> None:
    missing = {prediction_column, received_column} - set(frame.columns)
    if missing:
        raise DataQualityError(f"fundamentals join missing columns: {sorted(missing)}")
    future = frame.filter(
        pl.col(received_column).is_not_null()
        & (pl.col(received_column) > pl.col(prediction_column))
    )
    if not future.is_empty():
        raise DataQualityError(
            "future fundamentals detected (DT_RECEB > prediction date): "
            f"{_preview(future, ['ticker', 'CD_CVM', prediction_column, received_column])}"
        )


def validate_temporal_split(
    train: pl.DataFrame,
    evaluation: pl.DataFrame,
    *,
    date_column: str = "trade_date",
) -> None:
    if train.is_empty() or evaluation.is_empty():
        raise DataQualityError("temporal split cannot contain an empty train/evaluation frame")
    train_max = train[date_column].max()
    evaluation_min = evaluation[date_column].min()
    if train_max is None or evaluation_min is None or train_max >= evaluation_min:
        raise DataQualityError(
            f"target leakage risk: train max {train_max} must precede evaluation min {evaluation_min}"
        )
