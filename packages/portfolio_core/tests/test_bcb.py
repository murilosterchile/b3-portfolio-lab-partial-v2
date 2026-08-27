from datetime import date

import httpx
import polars as pl
import pytest
from portfolio_core.data.bcb import download_sgs_series
from portfolio_core.features.regime import attach_selic_candidate


def test_download_sgs_series_splits_ranges_longer_than_ten_years(monkeypatch) -> None:
    requested_ranges: list[tuple[str, str]] = []

    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> httpx.Response:
        requested_ranges.append((params["dataInicial"], params["dataFinal"]))
        request = httpx.Request("GET", url, params=params)
        day = params["dataInicial"]
        return httpx.Response(200, request=request, json=[{"data": day, "valor": "1,5"}])

    monkeypatch.setattr(httpx, "get", fake_get)

    frame = download_sgs_series(
        432,
        start=date(2011, 1, 1),
        end=date(2026, 8, 26),
        base_url="https://api.bcb.gov.br/dados/serie/bcdata.sgs",
    )

    assert requested_ranges == [
        ("01/01/2011", "28/12/2020"),
        ("29/12/2020", "26/08/2026"),
    ]
    assert frame["value"].to_list() == [1.5, 1.5]


def test_selic_candidate_cannot_be_available_on_reference_date() -> None:
    regime = pl.DataFrame({"trade_date": [date(2025, 1, 2), date(2025, 1, 3)]})
    selic = pl.DataFrame({"date": [date(2025, 1, 2)], "value": [12.25]})
    attached = attach_selic_candidate(regime, selic)
    assert attached["regime_selic_target"].to_list() == [None, 12.25]
    with pytest.raises(ValueError, match="at least one"):
        attach_selic_candidate(regime, selic, availability_lag_days=0)
