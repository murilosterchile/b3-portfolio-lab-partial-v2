# Quant/ML P1 implementation contract

This implementation keeps 2026 diagnostic-only. Any artifact used for selection must declare
`development_end_year=2025`, `diagnostic_year=2026`, and a knowledge cutoff no later than
`2025-12-31`.

## Governed sequence

1. Rebuild the PIT feature panel. Technical, fundamental, missingness, feature-age, liquidity,
   residual-volatility, investment and sector-neutral candidates are materialized causally.
2. Run `make diagnose-model-degradation`. It writes the complete window/half-life challenger
   table and only creates `models/selected_training_policy.json` when the pre-registered recent
   Rank-IC, net Top-10 spread and stability gates pass.
3. Run `make tune`. Tuning consumes that selected policy and uses development folds only.
4. Run `make train`. The saved metadata includes effective LightGBM bagging settings, annual
   feature coverage, OOF disagreement calibration bins, the selected policy and shrunk factor
   sleeve weights.
5. Run `make backtest`. This is the purged <=2025 development comparison. It includes 1/N,
   single-factor sleeves, equal and learned multifactor sleeves, ML, QKP and risk allocators.
6. Run `make qkp-ablation`. Candidate counts 20/30/50/75 use the same liquidity-first candidate
   rule and compare the prefilter, QKP selection and HRP allocation separately.
7. Only then run `make backtest USE_TRAINED_MODEL=1`. The 2026 report is diagnostic and cannot
   update any preceding artifact or acceptance gate.

## Deliberately gated components

- Market-based value and Black-Litterman are not primary strategies. There is no invented shares
  outstanding series; Black-Litterman now requires PIT market weights and explicit OOF view-error
  covariance.
- IPCA remains excluded until a governed release timestamp exists. SELIC has a mandatory
  availability lag of at least one calendar day and regime features remain challengers.
- Raw ensemble dispersion is persisted as `ensemble_disagreement`. It becomes
  `calibrated_uncertainty` only when inner time-ordered OOF bins show monotonically increasing
  absolute error; otherwise its portfolio penalty is zero.
- A negative result or failure to select a stable training policy is a valid research outcome.
  The pipeline stops instead of relaxing costs, gates or the 2026 boundary.
