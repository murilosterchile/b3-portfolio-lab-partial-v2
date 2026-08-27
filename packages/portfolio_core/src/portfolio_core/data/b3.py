from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .common import DataArtifact, sha256_file, utc_now_iso, write_manifest

# Official B3 COTAHIST layout (revision 02, fixed-width 245-byte record).
# Positions below are Python 0-based half-open slices corresponding to B3's 1-based inclusive layout.
_SLICES: dict[str, tuple[int, int]] = {
    "record_type": (0, 2),
    "trade_date": (2, 10),
    "bdi_code": (10, 12),
    "ticker": (12, 24),
    "market_type": (24, 27),
    "issuer_short_name": (27, 39),
    "specification": (39, 49),
    "term_days": (49, 52),
    "currency": (52, 56),
    "open": (56, 69),
    "high": (69, 82),
    "low": (82, 95),
    "average": (95, 108),
    "close": (108, 121),
    "best_bid": (121, 134),
    "best_ask": (134, 147),
    "trades": (147, 152),
    "quantity": (152, 170),  # QUATOT: number of securities traded
    "traded_value_brl": (170, 188),  # VOLTOT: financial value in BRL cents
    "exercise_price": (188, 201),
    "correction_indicator": (201, 202),
    "maturity_date": (202, 210),
    "quote_factor": (210, 217),
    "exercise_points": (217, 230),
    "isin": (230, 242),
    "distribution_number": (242, 245),
}

_PRICE_FIELDS = {
    "open",
    "high",
    "low",
    "average",
    "close",
    "best_bid",
    "best_ask",
    "traded_value_brl",
    "exercise_price",
}

_EQUITY_SPEC = re.compile(r"^(ON|PN|PNA|PNB|PNC|PND|UNT)\b")


def annual_url(base_url: str, year: int) -> str:
    return f"{base_url.rstrip('/')}/COTAHIST_A{year}.ZIP"


def daily_url(base_url: str, ddmmyyyy: str) -> str:
    return f"{base_url.rstrip('/')}/COTAHIST_D{ddmmyyyy}.ZIP"


def _parse_int(raw: str) -> int:
    raw = raw.strip()
    return int(raw) if raw else 0


def _parse_price(raw: str) -> float:
    raw = raw.strip()
    return int(raw) / 100.0 if raw else 0.0


def parse_cotahist_line(line: str) -> dict[str, object] | None:
    if len(line) < 245 or line[:2] != "01":
        return None
    row: dict[str, object] = {}
    for name, (start, end) in _SLICES.items():
        raw = line[start:end]
        if name in _PRICE_FIELDS:
            row[name] = _parse_price(raw)
        elif name in {"market_type", "trades", "quantity", "quote_factor", "distribution_number"}:
            row[name] = _parse_int(raw)
        else:
            row[name] = raw.strip()
    # Compatibility alias for already materialized datasets and downstream
    # readers. COTAHIST ``volume`` has always represented VOLTOT (BRL), never
    # QUATOT; new code should use the dimensionally explicit canonical name.
    row["volume"] = row["traded_value_brl"]
    return row


def parse_cotahist_zip(zip_path: Path, *, equities_only: bool = True) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        txt_names = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
        if not txt_names:
            raise ValueError(f"No COTAHIST TXT file found inside {zip_path}")
        with archive.open(txt_names[0]) as raw_stream:
            text = io.TextIOWrapper(raw_stream, encoding="latin-1", newline="")
            for line in text:
                row = parse_cotahist_line(line.rstrip("\r\n"))
                if row is None:
                    continue
                if row["market_type"] != 10:  # cash market
                    continue
                if equities_only and not _EQUITY_SPEC.match(str(row["specification"]).strip()):
                    continue
                records.append(row)
    if not records:
        return pl.DataFrame()
    frame = pl.DataFrame(records).with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=True),
    )
    return frame.sort(["trade_date", "ticker"])


def download_cotahist(
    *,
    year: int,
    base_url: str,
    data_dir: Path,
    timeout_seconds: float = 120.0,
    force: bool = False,
) -> tuple[Path, DataArtifact]:
    raw_dir = data_dir / "raw" / "b3" / str(year)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"COTAHIST_A{year}.ZIP"
    url = annual_url(base_url, year)

    if not zip_path.exists() or force:
        partial = zip_path.with_suffix(".ZIP.part")
        with httpx.stream("GET", url, timeout=timeout_seconds, follow_redirects=True) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
        if not zipfile.is_zipfile(partial):
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "B3 did not return a valid ZIP. The legacy endpoint can occasionally be protected "
                "by anti-bot controls; download the annual ZIP manually from the B3 historical "
                "quotes page and place it at the expected path, then rerun with --no-download."
            )
        partial.replace(zip_path)

    artifact = DataArtifact(
        source="B3 COTAHIST",
        source_url=url,
        fetched_at=utc_now_iso(),
        sha256=sha256_file(zip_path),
        bytes=zip_path.stat().st_size,
        notes="Official B3 fixed-width historical quotations. Raw files are intentionally immutable.",
    )
    write_manifest(raw_dir / "manifest.json", artifact)
    return zip_path, artifact


def materialize_cotahist(
    zip_path: Path,
    *,
    year: int,
    data_dir: Path,
    equities_only: bool = True,
    batch_size: int = 100_000,
) -> Path:
    """Stream COTAHIST into Parquet without holding the full year in memory."""
    output_dir = data_dir / "silver" / "b3_prices" / f"year={year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "part-000.parquet"
    temp_path = output_path.with_suffix(".parquet.part")
    temp_path.unlink(missing_ok=True)

    batch: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None
    arrow_schema: pa.Schema | None = None

    def flush() -> None:
        nonlocal batch, writer, arrow_schema
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            arrow_schema = table.schema
            writer = pq.ParquetWriter(temp_path, arrow_schema, compression="zstd", use_dictionary=True)
        elif arrow_schema is not None:
            table = table.cast(arrow_schema)
        writer.write_table(table)
        batch = []

    try:
        with zipfile.ZipFile(zip_path) as archive:
            txt_names = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
            if not txt_names:
                raise ValueError(f"No COTAHIST TXT file found inside {zip_path}")
            with archive.open(txt_names[0]) as raw_stream:
                text = io.TextIOWrapper(raw_stream, encoding="latin-1", newline="")
                for line in text:
                    row = parse_cotahist_line(line.rstrip("\r\n"))
                    if row is None or row["market_type"] != 10:
                        continue
                    if equities_only and not _EQUITY_SPEC.match(str(row["specification"]).strip()):
                        continue
                    row["trade_date"] = datetime.strptime(str(row["trade_date"]), "%Y%m%d").date()
                    batch.append(row)
                    if len(batch) >= batch_size:
                        flush()
                flush()
    finally:
        if writer is not None:
            writer.close()

    if not temp_path.exists():
        raise RuntimeError("No equity cash-market rows were materialized from the COTAHIST archive")
    temp_path.replace(output_path)
    return output_path
