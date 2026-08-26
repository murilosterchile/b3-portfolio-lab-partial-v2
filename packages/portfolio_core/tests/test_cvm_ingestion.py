import zipfile
from datetime import date

import polars as pl
from portfolio_core.data.cvm import extract_cvm_statements


def test_extract_cvm_statements_attaches_document_received_date(tmp_path) -> None:
    archive_path = tmp_path / "dfp_cia_aberta_2025.zip"
    metadata = (
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;DT_RECEB\n"
        "00.000.000/0001-91;2025-12-31;1;BCO BRASIL S.A.;001023;2026-02-11\n"
    )
    statement = (
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CD_CONTA;DS_CONTA;VL_CONTA\n"
        "00.000.000/0001-91;2025-12-31;1;BCO BRASIL S.A.;001023;3.01;Receita;1000\n"
        '00.000.000/0001-91;2025-12-31;1;BCO BRASIL S.A.;001023;6.02.13;'
        '"AFAC" e/ou aporte de capital;2500\n'
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dfp_cia_aberta_2025.csv", metadata.encode("latin1"))
        archive.writestr("dfp_cia_aberta_DRE_con_2025.csv", statement.encode("latin1"))

    outputs = extract_cvm_statements(
        archive_path, data_dir=tmp_path / "data", document="DFP", year=2025
    )

    frame = pl.read_parquet(outputs[0])
    assert frame.get_column("DT_RECEB").to_list() == [date(2026, 2, 11)] * 2
    assert frame.get_column("DS_CONTA").to_list() == ["Receita", '"AFAC" e/ou aporte de capital']
    assert frame.get_column("VL_CONTA").to_list() == [1000.0, 2500.0]


def test_extract_cvm_statements_accepts_empty_statement(tmp_path) -> None:
    archive_path = tmp_path / "dfp_cia_aberta_2026.zip"
    metadata = (
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;DT_RECEB\n"
        "00.000.000/0001-91;2026-12-31;1;BCO BRASIL S.A.;001023;2027-02-11\n"
    )
    empty_statement = (
        "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;CD_CVM;CD_CONTA;DS_CONTA;VL_CONTA\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dfp_cia_aberta_2026.csv", metadata.encode("latin1"))
        archive.writestr("dfp_cia_aberta_DFC_MD_con_2026.csv", empty_statement.encode("latin1"))

    outputs = extract_cvm_statements(
        archive_path, data_dir=tmp_path / "data", document="DFP", year=2026
    )

    frame = pl.read_parquet(outputs[0])
    assert frame.is_empty()
    assert frame.schema["VERSAO"] == pl.Int64
