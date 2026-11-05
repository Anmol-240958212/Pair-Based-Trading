# Project Progress

## Completed

- Loaded and standardised daily Capitaline data from 2015–2026.
- Created a fast Parquet copy containing 528,295 price observations.
- Joined all 206 price-series companies to sector and industry information from
  `STATIC_DATA_SHARANYA.xlsx`.
- Added point-in-time data-coverage and liquidity eligibility filters.
- Generated economically similar candidate pairs within the same industry.
- Implemented Engle–Granger cointegration testing, hedge-ratio estimation,
  spread z-scores and mean-reversion half-life calculations.
- Replaced Streamlit with a React/TypeScript/Plotly.js interface backed by a
  read-only FastAPI service.
- Added overview, robust pair-screening, on-demand pair diagnostics, formation
  and trading charts, trade logs, locked-test results, regime comparison, cost
  sensitivity and methodology views.
- Implemented an out-of-sample baseline pairs-trading backtest with next-day
  execution, transaction costs, entry and exit thresholds, stop-losses,
  maximum holding periods and trade-level results.
- Added a Jupyter notebook that uses the same reusable Python research code.
- Added automated tests for signal timing, stop-loss resets and exits.
- Added leakage-safe event labelling: every completed divergence trade becomes
  one ML row whose features are frozen on the signal date and whose later
  outcome supplies the binary target.
- Added chronological comparison pipelines for Logistic Regression, Decision
  Tree and Bagging Decision Trees.
- Generated 561 labelled divergence events from 17 completed semiannual
  trading windows (2018H1–2026H1); 153 satisfy the initial successful-reversion
  and after-cost profitability target, while 408 do not.
- Kept the incomplete 2026H2 period out of the labelled dataset.
- Added reverse-direction Engle-Granger tests, conservative bidirectional
  p-values, Benjamini-Hochberg FDR correction and Johansen confirmation.
- Added point-in-time pair, market, industry, volatility, momentum, divergence
  duration and liquidity features.
- Froze the validation-selected specification before evaluating 2022 onward:
  bidirectional Engle-Granger plus Johansen, Logistic Regression, profitability
  target and a 0.60 approval threshold.
- Completed the locked 2022H1-2026H1 walk-forward test and a ten-slot portfolio
  simulation with company-overlap controls and 5/10/20 bps cost sensitivity.
- Added a 10,000-run within-window permutation test. The ML filter's event-level
  improvement was not statistically significant (one-sided p = 0.242).
- Added API coverage to the automated suite; all 27 Python tests pass and the
  strict TypeScript production build succeeds.
- Documented that each annual workbook represents that year's F&O-company
  universe and that the supplied equity prices are corporate-action adjusted.
- Defined the empirical scope as adjusted underlying equities belonging to the
  F&O universe; futures rolls, margins and lot sizes are intentionally outside
  scope rather than silently approximated.
- Added a separately labelled Experiment 2 with boundary-safe entries, rolling
  hedge ratios, formation and signal-date stability diagnostics, a reversion
  classifier, expected-return regression, after-cost approval and
  volatility-aware portfolio sizing.
- Selected rolling rather than fixed hedging using only 2020-2021 validation
  evidence (ROC-AUC 0.716 versus 0.422).
- Evaluated Experiment 2 as a post-hoc 2022H1-2026H1 comparison. At 10 bps the
  two-stage filter reduced the dynamic baseline loss from -5.69% to -1.19% and
  drawdown from -9.04% to -2.79%, but remained unprofitable and insignificant
  (permutation p = 0.787).
- Added Experiment 3 with 90% formation coverage, 19,217 daily pair-level rows,
  next-day execution and explicit 20-day take-profit/stop-loss barrier labels.
- Compared Logistic Regression and histogram Gradient Boosting with purged,
  label-available walk-forward training and non-overlapping portfolio rules.
- The Experiment 3 Logistic portfolio is slightly positive at 10 bps (+0.89%,
  Sharpe 0.155, drawdown -2.85%) but negative at 20 bps and statistically
  insignificant under a moving-block bootstrap (p = 0.388).
- Added Experiment 4 comparing shrinkage Linear Discriminant Analysis,
  Gaussian Naive Bayes and a two-state Gaussian Markov regime filter combined
  with Logistic Regression on the unchanged daily panel.
- The Markov-regime portfolio is the best mathematical-model variant at 10 bps
  (+1.31%, Sharpe 0.222, drawdown -2.80%, 149 trades), improving on ordinary
  Logistic. It is negative at 20 bps and remains statistically insignificant
  under the moving-block bootstrap (p = 0.328).
- Added Experiment 5 with zero-, one- and three-day observation periods plus
  common-factor alignment, lead-lag, abnormal turnover, idiosyncratic shock and
  spread-acceleration confirmation features.
- Validation on 2020-2021 selected immediate entry, so the confirmation idea
  was not supported by the predeclared selection period. As a post-2022
  diagnostic, the three-day rule produced 67 trades, +2.48% at 10 bps, +1.12%
  at 20 bps, Sharpe 0.721, drawdown -0.78% and bootstrap p = 0.033. This is a
  promising rule to freeze and test on new data, not a fresh confirmed result.
- Downloaded 118,480 timestamped NSE corporate announcements covering 111 of
  the 112 companies in the three-day study; MCX returned no archive records and
  is excluded from like-for-like news comparisons rather than treated as having
  no news.
- Added Experiment 6 using announcements broadcast during the three-day
  trigger-to-confirmation interval. A one-sided material-event veto reduced
  executed trades from 67 to 41, improved later win rate from 65.7% to 68.3%
  and drawdown from -0.78% to -0.51%, but reduced 10-bps return from +2.48% to
  +2.16% and slightly worsened validation mean return.
- Added Experiment 7 with fully rebuilt labels for +1%/-2%, symmetric +/-0.5%,
  symmetric +/-1% and +1.5%/-1% payoff rules. The original +1%/-2% barrier was
  selected by 2020-2021 validation and remained strongest later. At 10 bps,
  symmetric +/-0.5% returned +0.70% and symmetric +/-1% returned +0.97%, both
  below the existing rule's +2.48%.
- Integrated the selected Markov-regime Logistic strategy into Pair Explorer.
  The interface now prioritises pairs with saved model activity and displays
  the three-day confirmation funnel, 0.60 probability decision, executed
  trades, pair-level realised equity and the classical baseline side by side.

## Initial Finding

A low formation-period cointegration p-value does not guarantee profitable
out-of-sample trading. In the January–June 2020 test, HCL Technologies–Infosys
and Bank of India–Punjab National Bank were profitable under the initial rules,
while ABB–Siemens and Hindalco–National Aluminium were not.

## Next Steps

- Write the formal paper around the frozen methodology and negative-but-useful
  out-of-sample result.
- Add exact intra-year F&O entry/removal dates if that source becomes available.
- Treat any future feature or classifier changes as a new experiment; do not
  retune against the already-opened 2022H1-2026H1 test period.
- Use completed 2026H2 or later observations as the next genuinely unseen
  evaluation period.
