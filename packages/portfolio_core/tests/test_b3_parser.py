from portfolio_core.data.b3 import parse_cotahist_line


def _put(buf: list[str], start: int, end: int, value: str) -> None:
    text = value.ljust(end - start)[: end - start]
    buf[start:end] = list(text)


def test_parse_official_fixed_width_line() -> None:
    buf = [" "] * 245
    _put(buf, 0, 2, "01")
    _put(buf, 2, 10, "20260821")
    _put(buf, 10, 12, "02")
    _put(buf, 12, 24, "TEST3")
    _put(buf, 24, 27, "010")
    _put(buf, 27, 39, "TEST CORP")
    _put(buf, 39, 49, "ON NM")
    _put(buf, 52, 56, "R$")
    for start, end, value in [
        (56, 69, "0000000001234"),
        (69, 82, "0000000001300"),
        (82, 95, "0000000001200"),
        (95, 108, "0000000001250"),
        (108, 121, "0000000001280"),
        (121, 134, "0000000001270"),
        (134, 147, "0000000001290"),
        (170, 188, "000000000123450000"),
    ]:
        _put(buf, start, end, value)
    _put(buf, 147, 152, "00123")
    _put(buf, 152, 170, "000000000000010000")
    _put(buf, 210, 217, "0000001")
    _put(buf, 230, 242, "BRTESTACNOR0")
    _put(buf, 242, 245, "001")
    row = parse_cotahist_line("".join(buf))
    assert row is not None
    assert row["ticker"] == "TEST3"
    assert row["market_type"] == 10
    assert row["close"] == 12.80
    assert row["trades"] == 123
    assert row["quantity"] == 10000
    assert row["traded_value_brl"] == 1_234_500.0
    assert row["volume"] == row["traded_value_brl"]  # non-destructive legacy alias
