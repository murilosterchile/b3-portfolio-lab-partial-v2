from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

# Common SGS series used by the prototype. Metadata should be reviewed before production use.
DEFAULT_SERIES = {
    "selic_target": 432,
    "ipca_monthly": 433,
    "usd_brl_sell": 1,
}

_MAX_DAILY_QUERY_DAYS = 3650


def download_sgs_series(
    code: int,
    *,
    start: date,
    end: date,
    base_url: str,
    timeout_seconds: float = 60.0,
) -> pl.DataFrame:
    url = f"{base_url.rstrip('/')}.{code}/dados"
    payload: list[dict[str, str]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_DAILY_QUERY_DAYS - 1), end)
        params = {
            "formato": "json",
            "dataInicial": chunk_start.strftime("%d/%m/%Y"),
            "dataFinal": chunk_end.strftime("%d/%m/%Y"),
        }
        response = httpx.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload.extend(response.json())
        chunk_start = chunk_end + timedelta(days=1)
    return pl.DataFrame(payload).select(
        pl.col("data").str.strptime(pl.Date, "%d/%m/%Y").alias("date"),
        pl.col("valor").str.replace(",", ".").cast(pl.Float64).alias("value"),
    )


def materialize_default_macro(
    *,
    start: date,
    end: date,
    base_url: str,
    data_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    for name, code in DEFAULT_SERIES.items():
        frame = download_sgs_series(code, start=start, end=end, base_url=base_url)
        out = data_dir / "silver" / "bcb" / f"series={name}" / "part-000.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.with_columns(pl.lit(code).alias("series_code")).write_parquet(out, compression="zstd")
        outputs.append(out)
    return outputs
