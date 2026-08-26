from datetime import date

import httpx
from portfolio_core.data.bcb import download_sgs_series


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
