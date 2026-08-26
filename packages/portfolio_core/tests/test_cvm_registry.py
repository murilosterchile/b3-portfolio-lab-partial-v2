import zipfile
from datetime import date
from pathlib import Path

import polars as pl
from portfolio_core.data.cvm_registry import (
    build_historical_issuer_bridge,
    parse_fca_sector_history,
    parse_fca_ticker_history,
)


def _fca_zip(path: Path) -> None:
    value_csv = (
        "CNPJ_Companhia;Codigo_Negociacao;Mercado;Data_Inicio_Negociacao;Data_Fim_Negociacao;Data_Referencia\n"
        "12.345.678/0001-90;ABCD3;BOLSA;2018-04-01;;2024-12-31\n"
    )
    general_csv = (
        "CNPJ_Companhia;Data_Referencia;Setor_Atividade\n"
        "12.345.678/0001-90;2018-01-01;Industria\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fca_cia_aberta_valor_mobiliario_2024.csv", value_csv)
        archive.writestr("fca_cia_aberta_geral_2024.csv", general_csv)


def test_fca_cad_builds_identifier_based_bridge(tmp_path: Path) -> None:
    archive = tmp_path / "fca.zip"
    _fca_zip(archive)
    tickers = parse_fca_ticker_history(archive)
    sectors = parse_fca_sector_history(archive)
    cad = pl.DataFrame(
        {
            "CNPJ": ["12345678000190"],
            "CD_CVM": ["99999"],
            "DENOM_SOCIAL": ["EMPRESA TESTE SA"],
            "DENOM_COMERC": ["TESTE"],
            "SITUACAO": ["ATIVO"],
        }
    )

    bridge = build_historical_issuer_bridge(
        cad=cad, ticker_history=tickers, sector_history=sectors
    )

    row = bridge.row(0, named=True)
    assert row["ticker"] == "ABCD3"
    assert row["CD_CVM"] == "99999"
    assert row["CNPJ"] == "12345678000190"
    assert row["mapping_source"] == "CVM FCA + CAD"
    assert row["sector"] == "Industria"


def test_bridge_uses_cvm_registration_valid_at_ticker_start() -> None:
    cad = pl.DataFrame(
        {
            "CNPJ": ["12345678000190", "12345678000190"],
            "CD_CVM": ["100", "200"],
            "DT_REG": [date(1990, 1, 1), date(2010, 1, 1)],
            "DT_CANCEL": [date(2000, 1, 1), None],
        }
    )
    tickers = pl.DataFrame(
        {
            "CNPJ": ["12345678000190"],
            "ticker": ["ABCD3"],
            "valid_from": [date(2018, 4, 1)],
            "valid_to": [None],
        }
    )

    bridge = build_historical_issuer_bridge(cad=cad, ticker_history=tickers)

    assert bridge.get_column("CD_CVM").to_list() == ["200"]
