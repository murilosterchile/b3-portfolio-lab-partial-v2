"""persist point-in-time issuer identity on model snapshots

Revision ID: 0002
Revises: 0001
"""

from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("asset_snapshots", sa.Column("issuer_id", sa.String(32), nullable=True))
    op.create_index("ix_asset_snapshots_issuer_id", "asset_snapshots", ["issuer_id"])


def downgrade() -> None:
    op.drop_index("ix_asset_snapshots_issuer_id", table_name="asset_snapshots")
    op.drop_column("asset_snapshots", "issuer_id")
