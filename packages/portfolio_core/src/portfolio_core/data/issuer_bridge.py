from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import polars as pl
from portfolio_core.data_quality import DataQualityError

REQUIRED_BRIDGE_COLUMNS = {"ticker", "CD_CVM", "valid_from", "valid_to", "sector"}
_LEGAL_NAME_TOKENS = {"A", "CIA", "COMPANHIA", "S", "SA"}


def validate_issuer_bridge(frame: pl.DataFrame) -> None:
    """Reject ambiguous ticker ownership intervals before a point-in-time join."""
    invalid = frame.filter(pl.col("valid_from") > pl.col("valid_to"))
    if not invalid.is_empty():
        raise DataQualityError(
            f"issuer bridge has invalid validity ranges: {invalid.head(10).to_dicts()}"
        )
    indexed = frame.with_row_index("_bridge_row")
    overlaps = indexed.join(indexed, on="ticker", how="inner", suffix="_right").filter(
        (pl.col("_bridge_row") < pl.col("_bridge_row_right"))
        & (pl.col("valid_from") <= pl.col("valid_to_right"))
        & (pl.col("valid_from_right") <= pl.col("valid_to"))
    )
    if not overlaps.is_empty():
        preview = overlaps.select(
            "ticker",
            "CD_CVM",
            "valid_from",
            "valid_to",
            "CD_CVM_right",
            "valid_from_right",
            "valid_to_right",
        ).head(10).to_dicts()
        raise DataQualityError(
            "issuer bridge has multiple CD_CVM mappings valid for the same ticker/date: "
            f"{preview}"
        )


def normalize_issuer_name(value: str) -> str:
    """Normalize issuer names without inventing aliases between B3 and CVM entities."""
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[A-Z0-9]+", ascii_name.upper())
    normalized = ["BANCO" if token == "BCO" else token for token in tokens]
    return "".join(token for token in normalized if token not in _LEGAL_NAME_TOKENS)


def suggest_issuer_bridge(
    b3_issuers: pl.DataFrame,
    cvm_issuers: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create safe automatic mappings and an auditable review report.

    Only a unique exact normalized name or a unique containment match is accepted
    automatically. Similarity scores are evidence for review, never an automatic join.
    """
    required_b3 = {"ticker", "issuer_short_name", "first_trade_date"}
    required_cvm = {"CD_CVM", "DENOM_CIA"}
    if missing := required_b3 - set(b3_issuers.columns):
        raise ValueError(f"B3 issuers missing columns: {sorted(missing)}")
    if missing := required_cvm - set(cvm_issuers.columns):
        raise ValueError(f"CVM issuers missing columns: {sorted(missing)}")

    cvm_entries: list[dict[str, str]] = []
    seen_cvm: set[tuple[str, str]] = set()
    for row in cvm_issuers.select("CD_CVM", "DENOM_CIA").drop_nulls().iter_rows(named=True):
        code = str(row["CD_CVM"]).strip()
        name = str(row["DENOM_CIA"]).strip()
        key = (code, name)
        if not code or not name or key in seen_cvm:
            continue
        seen_cvm.add(key)
        cvm_entries.append(
            {"CD_CVM": code, "DENOM_CIA": name, "normalized_name": normalize_issuer_name(name)}
        )

    bridge_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, object]] = []
    for row in b3_issuers.sort("ticker").iter_rows(named=True):
        ticker = str(row["ticker"]).strip().upper()
        issuer_name = str(row["issuer_short_name"]).strip()
        normalized = normalize_issuer_name(issuer_name)
        first_trade_date = str(row["first_trade_date"])

        exact = [entry for entry in cvm_entries if entry["normalized_name"] == normalized]
        contained = [
            entry
            for entry in cvm_entries
            if len(normalized) >= 5
            and (normalized in entry["normalized_name"] or entry["normalized_name"] in normalized)
        ]
        match: dict[str, str] | None = None
        match_type = ""
        if normalized and len(exact) == 1:
            match, match_type = exact[0], "exact_name"
        elif normalized and not exact and len(contained) == 1:
            match, match_type = contained[0], "unique_name_containment"

        if match is not None:
            bridge_rows.append(
                {
                    "ticker": ticker,
                    "CD_CVM": match["CD_CVM"],
                    "valid_from": first_trade_date,
                    "valid_to": "",
                    "sector": "Unknown",
                }
            )
            review_rows.append(
                {
                    "ticker": ticker,
                    "issuer_short_name": issuer_name,
                    "candidate_CD_CVM": match["CD_CVM"],
                    "candidate_DENOM_CIA": match["DENOM_CIA"],
                    "match_type": match_type,
                    "similarity": 1.0
                    if match_type == "exact_name"
                    else round(SequenceMatcher(None, normalized, match["normalized_name"]).ratio(), 4),
                    "status": "auto_added",
                }
            )
            continue

        ranked = sorted(
            cvm_entries,
            key=lambda entry: (
                SequenceMatcher(None, normalized, entry["normalized_name"]).ratio(),
                entry["DENOM_CIA"],
            ),
            reverse=True,
        )[:3]
        if not ranked:
            review_rows.append(
                {
                    "ticker": ticker,
                    "issuer_short_name": issuer_name,
                    "candidate_CD_CVM": "",
                    "candidate_DENOM_CIA": "",
                    "match_type": "no_candidate",
                    "similarity": 0.0,
                    "status": "unresolved",
                }
            )
        else:
            status = "ambiguous" if exact or len(contained) > 1 else "review_required"
            for candidate in ranked:
                review_rows.append(
                    {
                        "ticker": ticker,
                        "issuer_short_name": issuer_name,
                        "candidate_CD_CVM": candidate["CD_CVM"],
                        "candidate_DENOM_CIA": candidate["DENOM_CIA"],
                        "match_type": "name_similarity",
                        "similarity": round(
                            SequenceMatcher(
                                None, normalized, candidate["normalized_name"]
                            ).ratio(),
                            4,
                        ),
                        "status": status,
                    }
                )

    bridge_schema = {
        "ticker": pl.Utf8,
        "CD_CVM": pl.Utf8,
        "valid_from": pl.Utf8,
        "valid_to": pl.Utf8,
        "sector": pl.Utf8,
    }
    review_schema = {
        "ticker": pl.Utf8,
        "issuer_short_name": pl.Utf8,
        "candidate_CD_CVM": pl.Utf8,
        "candidate_DENOM_CIA": pl.Utf8,
        "match_type": pl.Utf8,
        "similarity": pl.Float64,
        "status": pl.Utf8,
    }
    return (
        pl.DataFrame(bridge_rows, schema=bridge_schema),
        pl.DataFrame(review_rows, schema=review_schema),
    )


def load_issuer_bridge(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    missing = REQUIRED_BRIDGE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Issuer bridge missing columns: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError(
            "Issuer bridge is empty. Add reviewed ticker/CD_CVM mappings with validity dates first."
        )
    parsed = frame.with_columns(
        pl.col("CD_CVM").cast(pl.Utf8),
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
        pl.col("valid_from").str.strptime(pl.Date, "%Y-%m-%d", strict=True),
        pl.when(pl.col("valid_to").is_null() | (pl.col("valid_to") == ""))
        .then(pl.lit("2999-12-31"))
        .otherwise(pl.col("valid_to"))
        .str.strptime(pl.Date, "%Y-%m-%d", strict=True)
        .alias("valid_to"),
    )
    validate_issuer_bridge(parsed)
    return parsed
