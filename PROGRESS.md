# Project Progress

## Completed

- Loaded and standardised daily Capitaline data from 2015–2026.
- Created a fast Parquet copy containing 528,295 price observations.
- Joined price records to static company, sector and industry information.
- Added point-in-time data-coverage and liquidity eligibility filters.
- Generated economically similar candidate pairs within the same industry.
- Implemented Engle–Granger cointegration testing, hedge-ratio estimation,
  spread z-scores and mean-reversion half-life calculations.
- Built a Streamlit interface for selecting periods, industries and company
  pairs and viewing interactive formation-period diagnostics.
- Implemented an out-of-sample baseline pairs-trading backtest with next-day
  execution, transaction costs, entry and exit thresholds, stop-losses,
  maximum holding periods and trade-level results.
- Added a Jupyter notebook that uses the same reusable Python research code.
- Added automated tests for signal timing, stop-loss resets and exits.

## Initial Finding

A low formation-period cointegration p-value does not guarantee profitable
out-of-sample trading. In the January–June 2020 test, HCL Technologies–Infosys
and Bank of India–Punjab National Bank were profitable under the initial rules,
while ABB–Siemens and Hindalco–National Aluminium were not.

## Next Steps

- Complete the formal data-quality and static-company coverage audit.
- Extend the baseline test into a full rolling walk-forward backtest.
- Construct labelled divergence events and ML features.
- Train and validate the ML signal-filtering model without look-ahead leakage.
- Compare the baseline and ML-filtered strategies across market regimes.
