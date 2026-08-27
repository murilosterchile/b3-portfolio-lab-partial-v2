from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

# Common SGS series used by the prototype. Metadata should be reviewed before production use.
DEFAULT_SERIES = {
    # SGS 12 is the official CDI rate in percent per business day.
    "cdi_daily": 12,
    "selic_target": 432,
    "ipca_monthly": 433,
    "usd_brl_sell": 1,
}

_MAX_DAILY_QUERY_DAYS = 3650
_MAX_RESPONSE_ATTEMPTS = 3


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
        for attempt in range(_MAX_RESPONSE_ATTEMPTS):
            response = httpx.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            try:
                chunk_payload = response.json()
                if not isinstance(chunk_payload, list) or not all(
                    isinstance(item, dict) for item in chunk_payload
                ):
                    raise ValueError("unexpected BCB SGS response structure")
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == _MAX_RESPONSE_ATTEMPTS - 1:
                    raise RuntimeError("BCB SGS returned an invalid JSON response") from exc
                time.sleep(2**attempt)
                continue
            payload.extend(chunk_payload)
            break
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
