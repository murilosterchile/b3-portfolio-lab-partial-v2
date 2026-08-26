# Research basis — partial v2

This revision intentionally changes the project only where the previous backtest exposed a
methodological weakness. It is not an attempt to tune the same holdout until it becomes profitable.

## External implementations reviewed

### Microsoft Qlib

Repository: `microsoft/qlib` (MIT).

Useful design ideas adopted conceptually:

- strict separation between processors fit on the training interval and inference processors;
- cross-sectional normalization/ranking for equity prediction;
- explicit labels based on future returns rather than future prices as input features;
- research workflows built around time-aware evaluation rather than shuffled CV.

No Qlib source code is vendored into this repository.

### skfolio

Repository: `skfolio/skfolio` (BSD-3-Clause).

Useful design ideas adopted conceptually:

- compare sophisticated allocation against naive/equal-weight baselines;
- transaction costs and turnover are first-class portfolio metrics;
- robust covariance/risk estimators and hierarchical allocation are preferable to unconstrained
  sample mean-variance when the expected-return estimate is noisy;
- cross-sectional transforms and walk-forward validation are part of the modelling pipeline.

### Riskfolio-Lib

Repository: `dcajasn/Riskfolio-Lib` (BSD-3-Clause).

Reviewed as a reference for portfolio risk constraints and risk-budgeting conventions. The current
project keeps its smaller in-house allocation layer instead of adding a large dependency.

### Machine Learning in Finance: From Theory to Practice

Book/code repository: `mfrdixon/ML_Finance_Codes`, corresponding to:

Matthew F. Dixon, Igor Halperin, Paul Bilokon, *Machine Learning in Finance: From Theory to
Practice*, Springer.

The main principle carried into this project is that financial ML must be evaluated as a noisy,
non-stationary prediction problem with explicit out-of-sample validation and economic metrics;
model complexity alone is not evidence of economic value.

## Academic signals used

### Cross-sectional machine learning

Gu, Kelly and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial
Studies 33(5), 2223–2273. DOI: 10.1093/rfs/hhaa009.

The paper motivates tree-based nonlinear models for expected-return prediction and identifies
momentum, liquidity and volatility among dominant predictive signals. This supports keeping
LightGBM/CatBoost as the primary nonlinear models while evaluating the result cross-sectionally.

### Momentum

Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling Losers: Implications for Stock
Market Efficiency*, Journal of Finance 48(1), 65–91. DOI: 10.1111/j.1540-6261.1993.tb04702.x.

The partial v2 retains 12-1 momentum and medium-horizon confirmation as an independent baseline and
as one component of the non-ML multifactor score.

### Profitability / quality

Novy-Marx (2013), *The Other Side of Value: The Gross Profitability Premium*, Journal of Financial
Economics 108(1), 1–28.

The partial v2 adds `gross_profitability = gross_profit / assets`, calculated point-in-time from CVM
statements, plus a cross-sectional rank. It also retains ROE/ROA and balance-sheet strength proxies.

### Covariance shrinkage

Ledoit and Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*,
Journal of Multivariate Analysis 88(2), 365–411.

The existing Ledoit-Wolf estimator remains the covariance default. In v2 the QKP pair penalty is
scaled by candidate volatility as well as correlation, making it covariance-like rather than using
correlation alone.

### Naive diversification as a mandatory baseline

DeMiguel, Garlappi and Uppal (2009), *Optimal Versus Naive Diversification: How Inefficient is the
1/N Portfolio Strategy?*, Review of Financial Studies 22(5), 1915–1953.

Because estimation error can make sophisticated optimizers look better in-sample than out-of-sample,
the backtest now writes separate ML, momentum, multifactor and research-v2 results instead of
reporting only one optimized strategy.

## B3/CVM data engineering changes

### Corporate actions / total-return analytical price

The raw COTAHIST files remain immutable. A new layer uses the B3 listed-company supplemental
endpoint to cache cash dividends/JCP and stock actions. It then builds backward-adjusted analytical
prices by ISIN. Invalid raw prices are quarantined rather than converted to zero.

The adjusted price layer is used automatically by technical features and the backtester when
available. If it is absent, the old raw-COTAHIST fallback remains, but it is explicitly not a
production-quality total-return basis.

### Historical ticker mapping without manual fuzzy matching

The new CVM registry pipeline downloads:

- `cad_cia_aberta.csv` for CNPJ -> `CD_CVM`;
- yearly FCA structured files for CNPJ -> historical `Codigo_Negociacao` plus validity dates;
- FCA sector history when available.

This creates a deterministic historical `ticker -> CNPJ -> CD_CVM` bridge and reduces both manual
mapping and survivorship bias. Fuzzy name similarity remains only as a fallback for unresolved rows.

## Backtest changes

The default `make backtest` now compares four strategies on the same walk-forward OOS dates:

1. `ml_top10`: conservative ML alpha only;
2. `momentum_top10`: academic momentum baseline;
3. `multifactor_top10`: momentum + low-volatility + point-in-time quality;
4. `research_v2`: 65% ML percentile + 25% multifactor score + 10% low-uncertainty score.

`research_v2` uses a rank buffer of five positions. Existing holdings are retained while they remain
inside `top_k + buffer`, which reduces boundary churn without using future data.

The run writes `data/gold/backtests/comparison.json` and applies a fixed pass/fail research gate. A
failed gate remains a failed result; the code does not retune the same holdout to force positive
performance.

## Remaining limitations in this partial version

- The B3 listed-company supplemental endpoint is public but not a formally versioned API. Raw JSON
  is therefore cached for reproducibility. If an issuer root cannot be refreshed, that root is
  excluded into a corporate-action coverage quarantine instead of silently using raw prices; a
  stricter fail-all mode is also available.
- Delisted issuers can have incomplete corporate-action availability at that endpoint. The FCA
  bridge fixes historical identity but does not itself supply every cash distribution.
- Full tax treatment, stock lending, subscription-right exercise, mergers, spin-offs and complex
  reorganizations are not yet modeled as investor cash flows.
- The backtest comparison in this partial release still tests top-K selection. A full rolling QKP
  walk-forward backtest, with point-in-time covariance and turnover-aware QKP state, is the next
  major quantitative milestone.
- Positive historical OOS performance, if obtained, is evidence about the tested period only and is
  not a guarantee of future return.
