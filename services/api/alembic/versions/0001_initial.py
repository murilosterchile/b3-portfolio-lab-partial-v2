"""initial schema

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_table("user_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("csrf_token", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_index("ix_user_sessions_token_hash", "user_sessions", ["token_hash"])
    op.create_index("ix_user_sessions_expires_at", "user_sessions", ["expires_at"])
    op.create_table("asset_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("as_of", sa.String(10), nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("company", sa.String(128), nullable=False),
        sa.Column("sector", sa.String(64), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("predicted_excess_return", sa.Float(), nullable=False),
        sa.Column("prediction_uncertainty", sa.Float(), nullable=False),
        sa.Column("volatility_annual", sa.Float(), nullable=False),
        sa.Column("ml_score", sa.Float(), nullable=False),
        sa.Column("quant_score", sa.Float(), nullable=False),
        sa.Column("liquidity_score", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.UniqueConstraint("as_of", "ticker", name="uq_asset_snapshot_date_ticker"),
    )
    op.create_index("ix_asset_snapshots_as_of", "asset_snapshots", ["as_of"])
    op.create_index("ix_asset_snapshots_ticker", "asset_snapshots", ["ticker"])
    op.create_index("ix_asset_snapshots_sector", "asset_snapshots", ["sector"])
    op.create_index("ix_asset_snapshot_latest", "asset_snapshots", ["as_of", "ml_score"])
    op.create_table("portfolios",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("budget", sa.Float(), nullable=False),
        sa.Column("risk_aversion", sa.Float(), nullable=False),
        sa.Column("objective", sa.Float(), nullable=False),
        sa.Column("solver", sa.String(32), nullable=False),
        sa.Column("exact", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table("portfolio_positions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("portfolio_id", sa.String(36), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("expected_excess_return", sa.Float(), nullable=False),
        sa.Column("volatility_annual", sa.Float(), nullable=False),
    )
    op.create_index("ix_portfolio_positions_portfolio_id", "portfolio_positions", ["portfolio_id"])


def downgrade() -> None:
    op.drop_table("portfolio_positions")
    op.drop_table("portfolios")
    op.drop_table("asset_snapshots")
    op.drop_table("user_sessions")
    op.drop_table("users")
