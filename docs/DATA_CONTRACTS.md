# Data contracts and leakage rules

1. Raw upstream artifacts are immutable and checksummed.
2. Every derived dataset has a separate raw/silver/gold path.
3. CVM statements may be used only when `DT_RECEB <= prediction_date`.
4. Preprocessing statistics and feature availability are fit/decided on training periods only.
5. Forward returns exist only as labels; they are never input features.
6. Historical universe identity should come from official CVM FCA/CAD validity data when available;
   current-listed-only membership is not an acceptable production backtest universe.
7. Raw COTAHIST `close` is not a total-shareholder-return series. Research promotion must use the
   governed adjusted analytical layer when the required B3 corporate actions are available.
8. Ticker/company mappings are governed identifiers. Official CNPJ/CD_CVM/FCA mappings take
   precedence; fuzzy name matching is fallback/review data, never a silent authoritative join.
9. Analytical prices must be finite, positive and unique by ticker/date. Invalid raw prices are
   quarantined; invalid prices actually used by a consumer are fatal.
10. When adjusted prices are unavailable, a change in COTAHIST `distribution_number` is treated as
    an unadjusted corporate-action boundary and that return is explicitly neutralized/reported.
    This fallback is not equivalent to total return.
11. Corporate-action adjustment is keyed by ISIN where available. Cash proceeds are applied using
    the contemporaneous raw cum-event price and stock events use backward share-unit adjustment.
12. Complex reorganizations that cannot be represented safely as a cash distribution or simple
    split/bonus must not be guessed; they remain excluded/quarantined until modeled explicitly.
13. Fundamental snapshots use only information available at the rebalance date. Same-receipt-date
    restatements are resolved deterministically before the as-of join.
14. Fold feature availability is learned from training data only. Features below the configured
    training coverage threshold are recorded and omitted from both fit and inference for that fold.
15. Correlation/QKP/risk inputs must be finite. Correlation matrices must be symmetric, bounded in
    [-1, 1], and have an approximately unit diagonal.
16. Transaction costs and turnover use the same one-way half-L1 convention, including cash.
17. Strategy parameters are not automatically changed after observing the final OOS gate. A failed
    gate stays failed; subsequent research requires a new development/validation split.

## Remaining economic limitations of partial v2

The new B3 corporate-action layer materially improves the old raw-COTAHIST backtest, but it is not a
complete security-master/accounting engine. Mergers, spin-offs, redemptions, subscription-right
exercise, taxes, stock lending and some historical delisted-company events may require additional
modeling or another governed source. The pipeline reports/fails rather than silently inventing cash
flows for unsupported cases.
