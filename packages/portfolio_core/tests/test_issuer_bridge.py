from datetime import date

import polars as pl
from portfolio_core.data.issuer_bridge import (
    normalize_issuer_name,
    suggest_issuer_bridge,
)


def test_normalize_issuer_name_handles_legal_suffixes_and_accents() -> None:
    assert normalize_issuer_name("BCO BRASIL S.A.") == "BANCOBRASIL"
    assert normalize_issuer_name("Companhia Energética S.A.") == "ENERGETICA"


def test_suggest_bridge_auto_adds_only_unique_safe_matches() -> None:
    b3 = pl.DataFrame(
        {
            "ticker": ["BBAS3", "PETR4", "TEST3"],
            "issuer_short_name": ["BCO BRASIL", "PETROBRAS", "ENERGIA"],
            "first_trade_date": [date(2023, 1, 2)] * 3,
        }
    )
    cvm = pl.DataFrame(
        {
            "CD_CVM": ["1023", "9512", "1111", "2222"],
            "DENOM_CIA": [
                "BCO BRASIL S.A.",
                "PETROLEO BRASILEIRO S.A. PETROBRAS",
                "ALFA ENERGIA S.A.",
                "BETA ENERGIA S.A.",
            ],
        }
    )

    bridge, review = suggest_issuer_bridge(b3, cvm)

    assert bridge.select("ticker", "CD_CVM").rows() == [("BBAS3", "1023"), ("PETR4", "9512")]
    assert bridge.get_column("valid_from").to_list() == ["2023-01-02", "2023-01-02"]
    assert review.filter(pl.col("ticker") == "TEST3").get_column("status").unique().to_list() == [
        "ambiguous"
    ]
    assert "TEST3" not in bridge.get_column("ticker").to_list()
