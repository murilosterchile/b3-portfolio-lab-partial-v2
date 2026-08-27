import polars as pl
import pytest

from portfolio_core.quant.factors import composite_quant_score


def test_composite_quant_score_sums_sleeve_contributions() -> None:
    frame = pl.DataFrame(
        {
            "rank_momentum_12_1": [0.8],
            "rank_return_126d": [0.6],
            "rank_return_63d": [0.4],
            "rank_volatility_63d": [0.2],
            "rank_residual_volatility_63d": [0.4],
        }
    )

    scored = composite_quant_score(frame)

    contributions = scored.select(
        pl.sum_horizontal(
            "quant_contribution_momentum",
            "quant_contribution_low_volatility",
            "quant_contribution_quality",
        )
    ).item()
    assert scored["quant_score"].item() == pytest.approx(contributions)
