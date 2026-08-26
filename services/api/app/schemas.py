from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class LoginRequest(RegisterRequest):
    pass


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticker: str
    issuer_id: str | None = None
    company: str
    sector: str
    price: float
    predicted_excess_return: float
    prediction_uncertainty: float
    volatility_annual: float
    ml_score: float
    quant_score: float
    liquidity_score: float
    explanation: str


class OptimizeRequest(BaseModel):
    budget: float = Field(gt=100.0, le=100_000_000.0)
    min_positions: int = Field(default=6, ge=1, le=30)
    max_positions: int = Field(default=10, ge=1, le=30)
    candidate_count: int = Field(default=30, ge=5, le=75)
    risk_aversion: float = Field(default=0.7, ge=0.0, le=5.0)
    uncertainty_penalty: float = Field(default=0.5, ge=0.0, le=3.0)
    allocation: Literal["hrp", "minvar", "inverse_vol", "cost_aware_minvar"] = "hrp"


class PositionOut(BaseModel):
    ticker: str
    issuer_id: str
    company: str
    sector: str
    weight: float
    amount: float
    price: float
    expected_excess_return: float
    volatility_annual: float


class OptimizeResponse(BaseModel):
    status: str
    solver: str
    exact: bool
    objective: float
    expected_excess_return: float
    expected_volatility: float
    positions: list[PositionOut]
    notes: list[str]
