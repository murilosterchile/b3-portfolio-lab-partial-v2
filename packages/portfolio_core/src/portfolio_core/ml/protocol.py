from __future__ import annotations

from dataclasses import dataclass
from datetime import date


DEFAULT_DEVELOPMENT_END_YEAR = 2025
DEFAULT_DIAGNOSTIC_YEAR = 2026
DEFAULT_LABEL_HORIZON_BARS = 63
DEFAULT_EXECUTION_LAG_BARS = 1


@dataclass(frozen=True)
class ResearchProtocol:
    """Pre-registered temporal governance for model development.

    2026 has already been observed during this research cycle. It is therefore
    diagnostic only and must never influence model, feature, target, cost,
    threshold, portfolio-size, or gate selection.
    """

    development_end_year: int = DEFAULT_DEVELOPMENT_END_YEAR
    diagnostic_year: int = DEFAULT_DIAGNOSTIC_YEAR
    label_horizon_bars: int = DEFAULT_LABEL_HORIZON_BARS
    execution_lag_bars: int = DEFAULT_EXECUTION_LAG_BARS

    def __post_init__(self) -> None:
        if self.development_end_year >= self.diagnostic_year:
            raise ValueError("development_end_year must precede diagnostic_year")
        if self.label_horizon_bars <= 0:
            raise ValueError("label_horizon_bars must be positive")
        if self.execution_lag_bars < 1:
            raise ValueError("execution_lag_bars must be at least 1 to prevent same-close execution")

    @property
    def development_knowledge_cutoff(self) -> date:
        return date(self.development_end_year, 12, 31)

    @property
    def diagnostic_start(self) -> date:
        return date(self.diagnostic_year, 1, 1)

    @property
    def diagnostic_end(self) -> date:
        return date(self.diagnostic_year, 12, 31)

    @property
    def warning(self) -> str:
        return (
            f"{self.diagnostic_year} has already been observed during research and is no longer "
            "a pristine holdout. It is diagnostic only."
        )
