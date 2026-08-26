# Technical decision report - B3 Portfolio Lab

## 1. Objective

The prototype is designed as a credible base for a future commercial product while remaining runnable on a single development machine. The main design goal is not “use as much AI as possible”; it is to create a system in which every investment signal, optimization result and backtest can be reproduced and audited.

The pipeline intentionally separates four questions:

1. **Data:** what information was actually available at decision time?
2. **Signal:** which assets appear relatively attractive?
3. **Selection:** which combination of candidate assets maximizes utility under discrete constraints?
4. **Sizing:** how should capital be distributed among the selected assets?

This separation is important because a model with useful ranking power can still produce a poor portfolio, and a sophisticated optimizer cannot rescue a signal with no out-of-sample information.

## 2. Architectural style

The application is a **modular monolith**, not a microservice fleet. It contains clear modules and process boundaries but deploys locally as three services: PostgreSQL, API and web UI.

Why:

- microservices add networking, tracing, versioning and deployment failure modes before scale requires them;
- research workloads need cheap local iteration;
- internal module boundaries can later be extracted into independent services without rewriting domain code;
- quantitative logic lives in `portfolio_core` and does not depend on FastAPI or Next.js.

## 3. Language choices

### Python 3.12 for data/ML/quant

Python is used where the ecosystem is strongest: Polars/Arrow, scikit-learn, LightGBM, CatBoost, SciPy, PySCIPOpt and backtesting. Python 3.12 was chosen instead of a newer interpreter because binary PySCIPOpt/SCIP support is mature on mainstream platforms and reproducibility is more important than using the newest interpreter release.

### C++23 for the independent exact optimizer reference

A separate exact branch-and-bound implementation is provided in C++23. It is not the default production solver; its purposes are:

- independent verification of small QKP instances;
- a path to specialized optimization later;
- profiling without Python overhead;
- reducing dependence on a single solver implementation.

The fallback bound includes every positive remaining linear and pair coefficient and ignores restrictive constraints, which makes it loose but valid. Therefore any pruning preserves exactness.

### TypeScript + Next.js for UI

Next.js gives a production-capable React structure, strong TypeScript support and simple container deployment. The UI deliberately avoids a large component framework in the first prototype: fewer dependencies, smaller supply-chain surface and easier design control.

## 4. PostgreSQL decision

PostgreSQL is the system-of-record database for identities, sessions, portfolio state and model-facing application snapshots. Historical market files are retained as immutable Parquet objects.

This is a deliberate distinction between a **database** and a **data lake file format**. PostgreSQL remains the only database in the architecture. Loading decades of raw COTAHIST rows into the transactional database would provide little value for the prototype and make ingestion/backtests slower and operationally heavier.

Commercial scale can later add partitioned PostgreSQL/Timescale-style storage or an analytical warehouse if real product workloads justify it.

## 5. Official B3 historical integration

The system integrates the official annual COTAHIST files:

`https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{YYYY}.ZIP`

The official B3 layout defines 245-byte records and three record types (header, quotation and trailer). The parser implements the official field positions for date, BDI, ticker, market type, issuer, specification, OHLC, trades, quantity, volume, ISIN and related fields.

Data lineage rules:

- raw ZIP is never overwritten silently;
- SHA-256 and source URL are written to a manifest;
- parsed data is stored by year as Parquet/ZSTD;
- derived features are written to a distinct `gold` layer;
- no derived data is allowed to mutate raw source artifacts.

### Critical B3 limitation

B3 states that the historical quotations are not adjusted for inflation or corporate distributions such as dividends/bonuses/rights. Therefore the prototype can exercise real B3 ingestion and price-based research, but a commercial performance engine needs a rights-cleared corporate-action/total-return adjustment source before portfolio claims are made.

## 6. CVM and fundamentals

The CVM module downloads official ITR/DFP ZIP files and preserves `DT_RECEB`, the delivery/receipt date. This is essential: a 31 March financial statement cannot be used by a model on 1 April if it was filed later.

The design uses as-of joins with:

`statement.DT_RECEB <= prediction_date`

A ticker-to-CVM-company bridge is intentionally governed rather than silently inferred with fuzzy text matching. A false issuer mapping is more dangerous than a missing feature because it creates plausible but wrong fundamentals.

Future production work should maintain an approved bridge table with effective dates, ticker changes and corporate reorganizations.

## 7. BCB macro layer

The BCB SGS client supports the official public JSON service and initial series for SELIC target, monthly IPCA and USD/BRL. Macro data is kept as its own series so release timing/frequency can later be modeled explicitly.

## 8. Machine-learning design

The first model is an ensemble of:

- LightGBM;
- CatBoost;
- Ridge regression baseline.

This choice is deliberate. Financial cross-sectional datasets are heterogeneous, noisy and mostly tabular. Gradient-boosted trees are strong baselines and frequently outperform much more complex deep models on such data. Ridge acts as a low-variance sanity component.

The target is **future excess return**, not exact future price. The self-contained prototype subtracts the cross-sectional market return. A production research run should use a formally chosen benchmark such as IBOV/CDI depending on the product mandate.

### Features

The prototype generates:

- 5/21/63/126/252-day returns;
- 12-1 momentum;
- rolling volatility;
- distance to moving averages;
- liquidity/trade proxies;
- cross-sectional percentile ranks.

The fundamental engine is a governed extension point because accounting taxonomy and issuer mapping need review before ratios are trusted.

### Validation

Random train/test splits are prohibited. Evaluation is calendar walk-forward. Model preprocessing is fit only on the historical training subset. The backtest retrains before each test year and never uses future labels to select the model for that year.

Primary research metrics should include Rank IC, top-minus-bottom spread, hit rate, turnover and net portfolio results. R-squared is not a primary promotion metric.

## 9. Quantitative models

The repository contains primitives for several academically established ideas:

- **Momentum:** medium-term relative strength, inspired by the classic continuation literature.
- **Low volatility:** explicit volatility ranking as a risk-aware signal component.
- **Ledoit-Wolf covariance shrinkage:** stabilizes covariance estimates in limited samples.
- **Hierarchical Risk Parity (HRP):** default sizing method after discrete selection.
- **Minimum variance / inverse volatility:** transparent allocation baselines.
- **Black-Litterman posterior:** allows ML forecasts to become uncertain “views” rather than unconditional expected returns.

A commercial research process should compare every complex strategy against simple equal-weight and factor baselines. If the sophisticated pipeline does not beat a simple baseline after realistic costs, complexity should not be retained.

## 10. Exact Quadratic Knapsack selection

The QKP layer receives individual utility and pairwise interactions.

`p_i` combines expected excess return and model uncertainty.

`q_ij` represents pairwise risk/diversification. The prototype converts correlation into a symmetric penalty/reward matrix. Negative coefficients are allowed; this matters because risk cannot be represented faithfully by a QKP implementation restricted to non-negative interactions.

### Exact SCIP backend

The default backend uses SCIP through PySCIPOpt. Each product `x_i x_j` becomes a binary `y_ij` with the exact Fortet-style linearization constraints. This turns the binary quadratic objective into an exactly equivalent MILP and allows budget, cardinality and sector constraints.

The result is marked `exact=true` only when SCIP reports an optimal status.

### Independent exact B&B

The C++ and Python fallback solvers use an independent branch-and-bound. The Python implementation is differential-tested against brute-force enumeration on random small instances containing both positive and negative interactions. The C++ unit test is also compiled and executed during artifact verification.

The fallback is intentionally limited to small candidate sets because its upper bound prioritizes correctness over tightness. SCIP is the production-shaped backend for larger instances.

## 11. Why selection and allocation are separate

QKP answers “which assets belong in the portfolio?”. HRP/minimum variance answers “how much capital should each selected asset receive?”.

Separating these questions enables controlled experiments:

- ML + equal weight;
- ML + QKP + equal weight;
- ML + QKP + HRP;
- factor ranking + QKP + HRP;
- simple momentum + equal weight.

This makes it possible to prove whether QKP adds value rather than attributing all performance to the whole stack.

## 12. Backtesting

The backtester applies signals only after their timestamp, rebalances monthly, tracks one-way turnover and subtracts configurable transaction costs. The included walk-forward script trains models only on earlier years.

Before commercialization it should be extended with:

- corporate-action-adjusted total returns;
- delistings and ticker histories;
- bid/ask spread and liquidity-dependent slippage;
- taxes appropriate to the product context;
- benchmark series with identical timestamps;
- survivorship-safe investable-universe history.

## 13. Security model

Security is present from the prototype stage because retrofitting identity/security is expensive.

Implemented foundations:

- Argon2id password hashing;
- cryptographically random server-side session tokens;
- only token hashes stored in PostgreSQL;
- HttpOnly session cookie;
- SameSite cookies;
- CSRF token primitive;
- explicit CORS allowlist;
- restrictive security headers;
- production `Secure`/HSTS behavior;
- secrets supplied by environment variables;
- PostgreSQL `pgcrypto`/`citext` extensions available;
- container runs API/web as non-root users.

Before public deployment add TLS termination, centralized secrets, rate limiting, audit-event persistence, backup/restore drills, dependency scanning in CI, SAST and an external penetration test.

## 14. Front-end design

The UI is deliberately information-dense without looking like a trading terminal. It has:

- concise model/optimizer/risk/data status cards;
- ranked assets with ML score, expected alpha and volatility;
- an intuitive budget / portfolio-size / risk-aversion builder;
- explicit exact-solver status;
- portfolio weights and expected risk/return summary;
- research disclaimer.

The UI treats “exact optimum proved” as a first-class fact rather than hiding optimization status.

## 15. Scalability path

The modular monolith is expected to scale significantly before service extraction is necessary. A future evolution can be:

1. object storage for raw/derived Parquet;
2. dedicated batch workers for ingest/train/backtest;
3. PostgreSQL read replicas / partitioning for application data;
4. model registry/artifact store;
5. scheduler/orchestrator;
6. separate optimization worker if large solve queues appear;
7. CDN/edge delivery for the web application.

Module interfaces are already separated so this evolution does not require rewriting the quantitative core.

## 16. Commercialization gates

The codebase can serve as a technical base, but the following are explicit go/no-go gates before selling the service:

- rights to use/derive/redistribute B3 data in the intended business model;
- CVM/legal analysis of whether the service is research, analysis, advisory or another regulated activity;
- corporate-action-adjusted, survivorship-safe research dataset;
- repeated out-of-sample results across regimes;
- realistic cost/slippage model;
- privacy policy/LGPD controls;
- security review and incident-response process;
- model-risk governance and reproducible recommendation audit log.

## 17. What was verified while building this artifact

- The standalone C++23 exact QKP solver compiled successfully with CMake.
- Its included unit test passed.
- The Python exact branch-and-bound implementation was differential-tested against brute-force enumeration on 20 random eight-item QKP instances with negative pair interactions; objectives matched within numerical tolerance.
- The official B3 fixed-width positions in the parser were derived from the B3 layout documentation.

Full Docker/web dependency installation could not be executed inside the artifact-building sandbox because the sandbox has no outbound package-network access. The Compose images are therefore intended to perform dependency installation on the user's local machine, where Docker/Podman has internet access.

## 18. Primary references used for design

- B3 Historical Quotes: https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/cotacoes-historicas/
- B3 Historical Quotations Layout: https://www.b3.com.br/data/files/33/67/B9/50/D84057102C784E47AC094EA8/SeriesHistoricas_Layout.pdf
- CVM Open Data: https://dados.cvm.gov.br/
- BCB SGS public service: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados
- PySCIPOpt installation: https://pyscipopt.readthedocs.io/en/stable/install.html
- SCIP Optimization Suite: https://www.scipopt.org/
- Gu, Kelly & Xiu, *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies.
- Jegadeesh & Titman, *Returns to Buying Winners and Selling Losers*, Journal of Finance.
- Ledoit & Wolf, covariance shrinkage literature.
- Lopez de Prado, *Building Diversified Portfolios that Outperform Out of Sample* (HRP).
- Black & Litterman, global portfolio optimization framework.
