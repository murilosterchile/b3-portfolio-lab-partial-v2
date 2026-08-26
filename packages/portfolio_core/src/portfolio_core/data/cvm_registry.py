from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable

import httpx
import polars as pl
from portfolio_core.data_quality import DataQualityError

from .common import DataArtifact, sha256_file, utc_now_iso, write_manifest

CVM_CAD_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
CVM_FCA_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS"

_TICKER_RE = re.compile(r"^[A-Z]{4}\d{1,2}$")


def _norm_column(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _column(frame: pl.DataFrame, *candidates: str, required: bool = False) -> str | None:
    lookup = {_norm_column(name): name for name in frame.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    if required:
        raise DataQualityError(
            f"CVM registry file missing one of columns {candidates}; found={frame.columns}"
        )
    return None


def _clean_cnpj_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.replace_all(r"\D", "")
        .str.pad_start(14, "0")
    )


def _download(url: str, path: Path, *, force: bool = False, timeout: float = 180.0) -> Path:
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with httpx.stream(
        "GET",
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "b3-portfolio-lab/0.1 research"},
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
    partial.replace(path)
    return path


def download_cvm_cad(*, data_dir: Path, force: bool = False) -> Path:
    path = data_dir / "raw" / "cvm_registry" / "cad_cia_aberta.csv"
    _download(CVM_CAD_URL, path, force=force)
    write_manifest(
        path.parent / "cad_manifest.json",
        DataArtifact(
            source="CVM Cadastro de Companhias Abertas",
            source_url=CVM_CAD_URL,
            fetched_at=utc_now_iso(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            notes="Official CVM open-data company register; used for CNPJ -> CD_CVM.",
        ),
    )
    return path


def download_cvm_fca(*, year: int, data_dir: Path, force: bool = False) -> Path:
    url = f"{CVM_FCA_BASE_URL}/fca_cia_aberta_{year}.zip"
    path = data_dir / "raw" / "cvm_registry" / "fca" / str(year) / f"fca_cia_aberta_{year}.zip"
    _download(url, path, force=force)
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"CVM FCA download is not a valid ZIP: {path}")
    write_manifest(
        path.parent / "manifest.json",
        DataArtifact(
            source="CVM FCA",
            source_url=url,
            fetched_at=utc_now_iso(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            notes="Official structured Formulario Cadastral used for historical ticker mapping.",
        ),
    )
    return path


def _read_latin1_csv_bytes(raw: bytes) -> pl.DataFrame:
    return pl.read_csv(
        io.StringIO(raw.decode("latin-1")),
        separator=";",
        infer_schema_length=0,
        null_values=["", "NA", "N/A", "nan"],
    )


def parse_cvm_cad(path: Path) -> pl.DataFrame:
    frame = _read_latin1_csv_bytes(path.read_bytes())
    cnpj_col = _column(frame, "cnpj_cia", "cnpj_companhia", required=True)
    code_col = _column(frame, "cd_cvm", required=True)
    social_col = _column(frame, "denom_social", "denom_cia", required=True)
    commercial_col = _column(frame, "denom_comerc", required=False)
    status_col = _column(frame, "sit", "sit_reg", required=False)
    registration_col = _column(frame, "dt_reg", required=False)
    cancellation_col = _column(frame, "dt_cancel", required=False)
    selected = frame.select(
        _clean_cnpj_expr(cnpj_col).alias("CNPJ"),
        pl.col(code_col).cast(pl.Utf8).str.strip_chars().alias("CD_CVM"),
        pl.col(social_col).cast(pl.Utf8).str.strip_chars().alias("DENOM_SOCIAL"),
        (
            pl.col(commercial_col).cast(pl.Utf8).str.strip_chars()
            if commercial_col
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("DENOM_COMERC"),
        (
            pl.col(status_col).cast(pl.Utf8).str.strip_chars()
            if status_col
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("SITUACAO"),
        _date_expr(registration_col).alias("DT_REG"),
        _date_expr(cancellation_col).alias("DT_CANCEL"),
    ).filter(
        pl.col("CNPJ").str.len_chars() == 14
    ).filter(
        pl.col("CD_CVM").is_not_null() & (pl.col("CD_CVM") != "")
    )
    return selected.unique(subset=["CNPJ", "CD_CVM"], keep="last").sort("CNPJ")


def _zip_csv(zip_path: Path, needle: str) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if needle in _norm_column(Path(name).name) and name.lower().endswith(".csv")
        ]
        if not names:
            raise DataQualityError(f"FCA ZIP has no CSV matching {needle!r}: {zip_path}")
        with archive.open(names[0]) as stream:
            return _read_latin1_csv_bytes(stream.read())


def _date_expr(column: str | None) -> pl.Expr:
    if column is None:
        return pl.lit(None, dtype=pl.Date)
    return pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)


def parse_fca_ticker_history(zip_path: Path) -> pl.DataFrame:
    """Parse FCA valor_mobiliario into deterministic CNPJ/ticker validity records."""
    frame = _zip_csv(zip_path, "valor_mobiliario")
    cnpj_col = _column(frame, "cnpj_companhia", "cnpj_cia", required=True)
    ticker_col = _column(frame, "codigo_negociacao", required=True)
    market_col = _column(frame, "mercado", required=False)
    start_col = _column(
        frame, "data_inicio_negociacao", "data_inicio_listagem", required=False
    )
    end_col = _column(frame, "data_fim_negociacao", "data_fim_listagem", required=False)
    ref_col = _column(frame, "data_referencia", required=False)
    out = frame.select(
        _clean_cnpj_expr(cnpj_col).alias("CNPJ"),
        pl.col(ticker_col).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("ticker"),
        (
            pl.col(market_col).cast(pl.Utf8).str.strip_chars()
            if market_col
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("market"),
        _date_expr(start_col).alias("valid_from"),
        _date_expr(end_col).alias("valid_to"),
        _date_expr(ref_col).alias("reference_date"),
    ).filter(
        (pl.col("CNPJ").str.len_chars() == 14)
        & pl.col("ticker").str.contains(r"^[A-Z]{4}\d{1,2}$")
    )
    # Some FCA vintages omit the explicit start date. The form reference date is a conservative
    # availability fallback; we never backdate a ticker based on a later form.
    out = out.with_columns(
        pl.coalesce([pl.col("valid_from"), pl.col("reference_date")]).alias("valid_from")
    ).drop_nulls(["valid_from"])
    return out.unique(
        subset=["CNPJ", "ticker", "valid_from", "valid_to"], keep="last"
    ).sort(["ticker", "valid_from"])


def parse_fca_sector_history(zip_path: Path) -> pl.DataFrame:
    """Parse point-in-time sector labels from FCA geral."""
    frame = _zip_csv(zip_path, "geral")
    cnpj_col = _column(frame, "cnpj_companhia", "cnpj_cia", required=True)
    ref_col = _column(frame, "data_referencia", required=True)
    sector_col = _column(frame, "setor_atividade", required=True)
    return (
        frame.select(
            _clean_cnpj_expr(cnpj_col).alias("CNPJ"),
            _date_expr(ref_col).alias("reference_date"),
            pl.col(sector_col).cast(pl.Utf8).str.strip_chars().alias("sector"),
        )
        .drop_nulls(["CNPJ", "reference_date", "sector"])
        .filter((pl.col("sector") != "") & (pl.col("CNPJ").str.len_chars() == 14))
        .unique(subset=["CNPJ", "reference_date"], keep="last")
        .sort(["CNPJ", "reference_date"])
    )


def _merge_intervals(history: pl.DataFrame) -> pl.DataFrame:
    """Collapse repeated FCA submissions without inventing validity across gaps.

    FCA is re-filed periodically, so the same CNPJ/ticker/start-date can appear in many yearly
    archives. We collapse only records that share the same explicit start date. Distinct start
    dates remain distinct intervals, which avoids bridging a ticker through a delisting/relisting
    gap or a later ticker reuse.
    """
    if history.is_empty():
        return pl.DataFrame(
            schema={
                "CNPJ": pl.Utf8,
                "ticker": pl.Utf8,
                "valid_from": pl.Date,
                "valid_to": pl.Date,
            }
        )
    return (
        history.select("CNPJ", "ticker", "valid_from", "valid_to")
        .drop_nulls(["CNPJ", "ticker", "valid_from"])
        .group_by(["CNPJ", "ticker", "valid_from"])
        .agg(pl.col("valid_to").max().alias("valid_to"))
        .sort(["ticker", "valid_from"])
    )



def _coalesce_same_issuer_intervals(frame: pl.DataFrame) -> pl.DataFrame:
    """Merge overlapping/adjacent intervals only for the same ticker and CVM issuer."""
    rows: list[dict[str, object]] = []
    for group in frame.sort(["ticker", "CD_CVM", "valid_from"]).partition_by(
        ["ticker", "CD_CVM"], maintain_order=True
    ):
        current: dict[str, object] | None = None
        for row in group.iter_rows(named=True):
            if current is None:
                current = dict(row)
                continue
            current_end = current["valid_to"]
            next_start = row["valid_from"]
            if current_end is not None and next_start is not None and next_start <= current_end:
                row_end = row["valid_to"]
                if row_end is not None and (current_end is None or row_end > current_end):
                    current["valid_to"] = row_end
                continue
            rows.append(current)
            current = dict(row)
        if current is not None:
            rows.append(current)
    return pl.DataFrame(rows, schema=frame.schema) if rows else frame.head(0)

def build_historical_issuer_bridge(
    *,
    cad: pl.DataFrame,
    ticker_history: pl.DataFrame,
    sector_history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a no-fuzzy-match ticker -> CD_CVM bridge from official CVM identifiers."""
    collapsed = _merge_intervals(ticker_history)
    if {"DT_REG", "DT_CANCEL"}.issubset(cad.columns):
        bridge = collapsed.join(
            cad.select("CNPJ", "CD_CVM", "DT_REG", "DT_CANCEL"),
            on="CNPJ",
            how="inner",
        ).filter(
            (pl.col("DT_REG").is_null() | (pl.col("DT_REG") <= pl.col("valid_from")))
            & (pl.col("DT_CANCEL").is_null() | (pl.col("valid_from") <= pl.col("DT_CANCEL")))
        ).drop("DT_REG", "DT_CANCEL")
    else:
        bridge = collapsed.join(cad.select("CNPJ", "CD_CVM"), on="CNPJ", how="inner")
    bridge = bridge.with_columns(
        pl.col("valid_to").fill_null(date(2999, 12, 31)),
        pl.lit("Unknown").alias("sector"),
        pl.lit("CVM FCA + CAD").alias("mapping_source"),
    )
    bridge = _coalesce_same_issuer_intervals(bridge)
    if sector_history is not None and not sector_history.is_empty():
        # Sector may only be attached if it was already known by the beginning of the validity
        # interval; otherwise keep Unknown instead of leaking a later classification backward.
        sector = sector_history.sort(["CNPJ", "reference_date"])
        bridge = bridge.sort(["CNPJ", "valid_from"]).join_asof(
            sector,
            left_on="valid_from",
            right_on="reference_date",
            by="CNPJ",
            strategy="backward",
            check_sortedness=False,
            suffix="_fca",
        ).with_columns(
            pl.coalesce([pl.col("sector_fca"), pl.col("sector")]).alias("sector")
        ).drop(["sector_fca", "reference_date"])
    result = bridge.select(
        "ticker", "CD_CVM", "valid_from", "valid_to", "sector", "CNPJ", "mapping_source"
    ).sort(["ticker", "valid_from"])
    # Reject overlapping ownership by different CVM issuers; do not silently choose one.
    indexed = result.with_row_index("_row")
    overlap = indexed.join(indexed, on="ticker", how="inner", suffix="_right").filter(
        (pl.col("_row") < pl.col("_row_right"))
        & (pl.col("valid_from") <= pl.col("valid_to_right"))
        & (pl.col("valid_from_right") <= pl.col("valid_to"))
        & (pl.col("CD_CVM") != pl.col("CD_CVM_right"))
    )
    if not overlap.is_empty():
        raise DataQualityError(
            "CVM FCA produced overlapping ticker ownership intervals: "
            f"{overlap.select('ticker', 'CD_CVM', 'valid_from', 'valid_to', 'CD_CVM_right', 'valid_from_right', 'valid_to_right').head(10).to_dicts()}"
        )
    return result


def materialize_cvm_registry(
    *,
    data_dir: Path,
    years: Iterable[int],
    force: bool = False,
) -> dict[str, object]:
    cad_path = download_cvm_cad(data_dir=data_dir, force=force)
    cad = parse_cvm_cad(cad_path)
    tickers: list[pl.DataFrame] = []
    sectors: list[pl.DataFrame] = []
    used_years: list[int] = []
    for year in sorted(set(int(value) for value in years)):
        path = download_cvm_fca(year=year, data_dir=data_dir, force=force)
        tickers.append(parse_fca_ticker_history(path))
        sectors.append(parse_fca_sector_history(path))
        used_years.append(year)
    ticker_history = pl.concat(tickers, how="vertical_relaxed") if tickers else pl.DataFrame()
    sector_history = pl.concat(sectors, how="vertical_relaxed") if sectors else pl.DataFrame()
    bridge = build_historical_issuer_bridge(
        cad=cad,
        ticker_history=ticker_history,
        sector_history=sector_history,
    )
    root = data_dir / "silver" / "cvm_registry"
    root.mkdir(parents=True, exist_ok=True)
    ticker_path = root / "ticker_history.parquet"
    sector_path = root / "sector_history.parquet"
    bridge_path = root / "issuer_bridge.parquet"
    ticker_history.write_parquet(ticker_path, compression="zstd")
    sector_history.write_parquet(sector_path, compression="zstd")
    bridge.write_parquet(bridge_path, compression="zstd")
    manifest = {
        "source": "CVM CAD + FCA",
        "generated_at": utc_now_iso(),
        "years": used_years,
        "cad_rows": cad.height,
        "ticker_history_rows": ticker_history.height,
        "sector_history_rows": sector_history.height,
        "bridge_rows": bridge.height,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
