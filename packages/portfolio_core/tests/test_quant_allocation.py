import numpy as np

from portfolio_core.quant import cost_aware_allocation


def test_cost_aware_allocation_starts_with_feasible_sector_exposure() -> None:
    covariance = np.eye(6)

    weights = cost_aware_allocation(
        covariance,
        max_weight=0.25,
        sectors=["A", "A", "B", "B", "C", "C"],
        max_sector_weight=0.40,
        issuer_ids=["1", "2", "3", "4", "5", "6"],
        max_issuer_weight=0.25,
        liquidity_weight_caps=np.asarray([0.25, 0.25, 0.20, 0.20, 0.15, 0.15]),
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights[:2].sum() <= 0.40 + 1e-7
    assert weights[2:4].sum() <= 0.40 + 1e-7
    assert weights[4:].sum() <= 0.40 + 1e-7
