from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from portfolio_core.data_quality import DataQualityError, validate_finite_array


@dataclass(frozen=True)
class QKPInstance:
    names: tuple[str, ...]
    linear_values: np.ndarray
    pair_values: np.ndarray
    costs: np.ndarray
    capacity: float
    min_cardinality: int = 1
    max_cardinality: int | None = None
    sectors: tuple[str, ...] | None = None
    sector_max_count: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0:
            raise ValueError("QKP instance must contain at least one item")
        if self.linear_values.shape != (n,) or self.costs.shape != (n,):
            raise ValueError("linear_values and costs must match item count")
        if self.pair_values.shape != (n, n):
            raise ValueError("pair_values must be an n x n matrix")
        validate_finite_array(self.linear_values, context="QKP linear values")
        validate_finite_array(self.pair_values, context="QKP pair values")
        validate_finite_array(self.costs, context="QKP costs")
        if not np.isfinite(self.capacity):
            raise DataQualityError("QKP capacity must be finite")
        if not np.allclose(self.pair_values, self.pair_values.T, atol=1e-10):
            raise ValueError("pair_values must be symmetric")
        if np.any(self.costs < 0) or self.capacity < 0:
            raise ValueError("costs/capacity must be non-negative")
        max_card = self.max_cardinality if self.max_cardinality is not None else n
        if not (0 <= self.min_cardinality <= max_card <= n):
            raise ValueError("invalid cardinality bounds")
        if self.sectors is not None and len(self.sectors) != n:
            raise ValueError("sectors must match item count")


@dataclass(frozen=True)
class OptimizationResult:
    selected_indices: tuple[int, ...]
    selected_names: tuple[str, ...]
    objective: float
    total_cost: float
    status: str
    solver: str
    exact: bool
    gap: float | None = None
