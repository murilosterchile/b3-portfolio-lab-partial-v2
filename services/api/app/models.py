from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    __tablename__ = "user_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[User] = relationship(back_populates="sessions")


class AssetSnapshot(Base):
    __tablename__ = "asset_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of: Mapped[str] = mapped_column(String(10), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    company: Mapped[str] = mapped_column(String(128))
    sector: Mapped[str] = mapped_column(String(64), index=True)
    price: Mapped[float] = mapped_column(Float)
    predicted_excess_return: Mapped[float] = mapped_column(Float)
    prediction_uncertainty: Mapped[float] = mapped_column(Float)
    volatility_annual: Mapped[float] = mapped_column(Float)
    ml_score: Mapped[float] = mapped_column(Float)
    quant_score: Mapped[float] = mapped_column(Float)
    liquidity_score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text, default="")
    __table_args__ = (UniqueConstraint("as_of", "ticker", name="uq_asset_snapshot_date_ticker"),)


class Portfolio(Base):
    __tablename__ = "portfolios"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(120), default="Portfolio")
    budget: Mapped[float] = mapped_column(Float)
    risk_aversion: Mapped[float] = mapped_column(Float)
    objective: Mapped[float] = mapped_column(Float)
    solver: Mapped[str] = mapped_column(String(32))
    exact: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    positions: Mapped[list["PortfolioPosition"]] = relationship(cascade="all, delete-orphan")


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    ticker: Mapped[str] = mapped_column(String(16))
    weight: Mapped[float] = mapped_column(Float)
    expected_excess_return: Mapped[float] = mapped_column(Float)
    volatility_annual: Mapped[float] = mapped_column(Float)


Index("ix_asset_snapshot_latest", AssetSnapshot.as_of, AssetSnapshot.ml_score)
