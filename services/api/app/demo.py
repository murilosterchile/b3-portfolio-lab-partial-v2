from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AssetSnapshot


DEMO_ROWS = [
    ("WEGE3", "WEG", "Industrials", 38.70, 0.054, 0.019, 0.265, 91.0, 88.0, 96.0, "Momentum and profitability proxies are strong; valuation risk reduces confidence."),
    ("ITUB4", "Itau Unibanco", "Financials", 36.10, 0.041, 0.014, 0.218, 88.0, 82.0, 99.0, "Stable trend, strong liquidity and lower predicted uncertainty."),
    ("BBAS3", "Banco do Brasil", "Financials", 28.40, 0.047, 0.021, 0.283, 86.0, 84.0, 98.0, "High expected excess return with a larger macro sensitivity penalty."),
    ("PETR4", "Petrobras", "Energy", 33.90, 0.050, 0.027, 0.341, 85.0, 87.0, 100.0, "Momentum is favorable; commodity and governance exposure increase uncertainty."),
    ("EQTL3", "Equatorial", "Utilities", 35.20, 0.036, 0.013, 0.205, 84.0, 79.0, 94.0, "Defensive risk profile improves portfolio diversification."),
    ("PRIO3", "PRIO", "Energy", 42.50, 0.061, 0.031, 0.389, 83.0, 90.0, 96.0, "High alpha estimate but the uncertainty and energy correlation penalties are meaningful."),
    ("VALE3", "Vale", "Materials", 64.80, 0.045, 0.026, 0.333, 81.0, 83.0, 100.0, "Strong liquidity and momentum, balanced by commodity-cycle risk."),
    ("SUZB3", "Suzano", "Materials", 52.30, 0.039, 0.024, 0.318, 79.0, 80.0, 95.0, "FX sensitivity can diversify domestic exposures, with elevated volatility."),
    ("RADL3", "Raia Drogasil", "Consumer", 27.70, 0.032, 0.015, 0.229, 78.0, 72.0, 92.0, "Lower-volatility consumer exposure supports diversification."),
    ("RENT3", "Localiza", "Consumer", 48.10, 0.037, 0.022, 0.312, 77.0, 76.0, 95.0, "Improving trend; interest-rate sensitivity remains a risk factor."),
    ("ABEV3", "Ambev", "Consumer", 13.20, 0.025, 0.012, 0.196, 75.0, 68.0, 99.0, "Defensive volatility and liquidity offset a modest alpha estimate."),
    ("BBDC4", "Bradesco", "Financials", 15.90, 0.031, 0.020, 0.276, 74.0, 71.0, 99.0, "Positive recovery signal with more uncertainty than the leading financial names."),
    ("TOTS3", "TOTVS", "Technology", 32.40, 0.043, 0.023, 0.321, 80.0, 85.0, 91.0, "Growth and momentum signals are favorable; valuation raises model uncertainty."),
    ("VIVT3", "Telefonica Brasil", "Communication", 47.80, 0.028, 0.011, 0.181, 82.0, 70.0, 96.0, "Low volatility and defensive cash-flow profile provide diversification."),
    ("CPLE6", "Copel", "Utilities", 10.70, 0.030, 0.014, 0.211, 76.0, 74.0, 93.0, "Utility exposure has a relatively stable risk profile."),
    ("GGBR4", "Gerdau", "Materials", 18.60, 0.034, 0.025, 0.347, 72.0, 77.0, 96.0, "Cyclical exposure increases risk despite constructive momentum."),
    ("LREN3", "Lojas Renner", "Consumer", 17.30, 0.029, 0.026, 0.355, 69.0, 73.0, 94.0, "Potential upside is offset by rate and consumption sensitivity."),
    ("SBSP3", "Sabesp", "Utilities", 91.00, 0.035, 0.018, 0.248, 78.0, 81.0, 95.0, "Defensive characteristics and idiosyncratic catalysts improve diversification."),
]


def seed_demo(db: Session) -> int:
    if db.scalar(select(AssetSnapshot.id).limit(1)) is not None:
        return 0
    as_of = date.today().isoformat()
    for row in DEMO_ROWS:
        db.add(
            AssetSnapshot(
                as_of=as_of,
                ticker=row[0], issuer_id=row[0][:4], company=row[1], sector=row[2], price=row[3],
                predicted_excess_return=row[4], prediction_uncertainty=row[5],
                volatility_annual=row[6], ml_score=row[7], quant_score=row[8],
                liquidity_score=row[9], explanation=row[10],
            )
        )
    db.commit()
    return len(DEMO_ROWS)
