from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import polars as pl

from .common import DataArtifact, sha256_file, utc_now_iso, write_manifest


def cvm_document_url(base_url: str, document: str, year: int) -> str:
    doc = document.upper()
    if doc not in {"ITR", "DFP"}:
        raise ValueError("document must be ITR or DFP")
    return f"{base_url.rstrip('/')}/{doc}/DADOS/{doc.lower()}_cia_aberta_{year}.zip"


def download_cvm_document(
    *,
    document: str,
    year: int,
    base_url: str,
    data_dir: Path,
    timeout_seconds: float = 120.0,
) -> Path:
    url = cvm_document_url(base_url, document, year)
    raw_dir = data_dir / "raw" / "cvm" / document.lower() / str(year)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / Path(url).name
    if not target.exists():
        response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
        response.raise_for_status()
        target.write_bytes(response.content)
    if not zipfile.is_zipfile(target):
        raise RuntimeError(f"CVM resource is not a ZIP: {url}")
    write_manifest(
        raw_dir / "manifest.json",
        DataArtifact(
            source=f"CVM {document.upper()}",
            source_url=url,
            fetched_at=utc_now_iso(),
            sha256=sha256_file(target),
            bytes=target.stat().st_size,
            notes="Official CVM open data. DT_RECEB is preserved for point-in-time joins.",
        ),
    )
    return target


def extract_cvm_statements(zip_path: Path, *, data_dir: Path, document: str, year: int) -> list[Path]:
    wanted_markers = ("_DRE_con_", "_BPA_con_", "_BPP_con_", "_DFC_MI_con_", "_DFC_MD_con_")
    outputs: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        metadata_name = f"{document.lower()}_cia_aberta_{year}.csv"
        if metadata_name not in archive.namelist():
            raise ValueError(f"CVM archive is missing document metadata: {metadata_name}")
        with archive.open(metadata_name) as stream:
            metadata = pl.read_csv(
                io.BytesIO(stream.read()),
                separator=";",
                encoding="latin1",
                infer_schema_length=5000,
            )
        metadata_keys = ["CNPJ_CIA", "DT_REFER", "VERSAO"]
        metadata = metadata.with_columns(pl.col("VERSAO").cast(pl.Int64, strict=False))
        metadata = metadata.group_by(metadata_keys).agg(pl.col("DT_RECEB").min())

        for name in archive.namelist():
            if not name.lower().endswith(".csv") or not any(marker in name for marker in wanted_markers):
                continue
            with archive.open(name) as stream:
                raw = stream.read()
            statement = next((m.strip("_").split("_")[0] for m in wanted_markers if m in name), "statement")
            frame = pl.read_csv(
                io.BytesIO(raw),
                separator=";",
                encoding="latin1",
                infer_schema_length=5000,
                ignore_errors=True,
                quote_char=None,
            )
            frame = frame.with_columns(pl.col("VERSAO").cast(pl.Int64, strict=False))
            frame = frame.join(metadata, on=metadata_keys, how="left", validate="m:1")
            if frame.get_column("DT_RECEB").null_count():
                raise ValueError(f"CVM statement rows without DT_RECEB metadata: {name}")
            keep = [
                col
                for col in [
                    "CNPJ_CIA",
                    "CD_CVM",
                    "DENOM_CIA",
                    "DT_REFER",
                    "DT_INI_EXERC",
                    "DT_FIM_EXERC",
                    "DT_RECEB",
                    "VERSAO",
                    "CD_CONTA",
                    "DS_CONTA",
                    "VL_CONTA",
                    "ORDEM_EXERC",
                ]
                if col in frame.columns
            ]
            frame = frame.select(keep).with_columns(pl.lit(statement).alias("STATEMENT_TYPE"))
            for col in ("DT_REFER", "DT_INI_EXERC", "DT_FIM_EXERC", "DT_RECEB"):
                if col in frame.columns:
                    frame = frame.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
            if "VL_CONTA" in frame.columns:
                frame = frame.with_columns(
                    pl.col("VL_CONTA")
                    .cast(pl.Utf8)
                    .str.replace_all("\\.", "")
                    .str.replace(",", ".")
                    .cast(pl.Float64, strict=False)
                )
            out = (
                data_dir
                / "silver"
                / "cvm"
                / f"document={document.lower()}"
                / f"statement={statement.lower()}"
                / f"year={year}"
                / "part-000.parquet"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(out, compression="zstd")
            outputs.append(out)
    return outputs
