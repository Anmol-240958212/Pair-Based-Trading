# Research Methodology and Frozen Test Record

## Data and windows

- Daily Capitaline observations for 206 Indian F&O companies, January 2015 to
  October 2026. Prices are adjusted for corporate actions.
- Each annual workbook contains the companies that belonged to the F&O universe
  in that year; the universe is therefore not constructed by backfilling the
  latest company list across the full history.
- Candidate companies require at least 95% formation-period date coverage and
  500 observations.
- Candidate pairs are formed only inside the same Capitaline industry.
- Each completed experiment uses a trailing three-year formation period and the
  following six-month trading period.
- The incomplete 2026H2 period is not labelled or evaluated.

## Pair robustness rules

The event superset uses forward Engle-Granger p <= 0.05, a positive hedge ratio
and a spread half-life between 2 and 120 days. Every candidate also records:

- reverse-direction Engle-Granger p-value;
- conservative maximum of the two p-values;
- Benjamini-Hochberg false-discovery-rate adjustment;
- Johansen trace-test confirmation at 5%.

FDR at 10% retained only 14 events overall and no usable development training
sample, so it is reported as a robustness result rather than used as the primary
rule.

## ML development and frozen specification

- Development training outcomes: 2018-2019.
- Validation: 2020-2021.
- Locked walk-forward test: 2022H1-2026H1.
- Frozen pair rule: bidirectional Engle-Granger plus Johansen confirmation.
- Frozen classifier: class-balanced Logistic Regression.
- Frozen target: positive after-cost trade return.
- Frozen probability threshold: 0.60.

These choices were recorded in `src/pairs_trading/research_config.py` before the
locked test was run. In each locked-test half-year, the model is retrained only
on outcomes completed before that half-year starts.

## Locked-test result

The robust baseline contained 148 signals. The ML model approved 61. Mean trade
return changed from -0.207% for all baseline signals to +0.187% for approved
signals. A within-window random-filter permutation test gave p = 0.242, so this
event-level improvement is not statistically significant.

With ten fixed capital slots, no simultaneous reuse of a company, and 10 bps
cost per position change:

| Strategy | Trades | Total return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Baseline | 127 | -4.07% | -0.285 | -9.60% |
| ML filtered | 55 | -0.48% | -0.038 | -5.36% |

The defensible conclusion is that the frozen ML filter reduced losses and
drawdown in this sample, but did not create a profitable portfolio and did not
show statistically significant predictive improvement.

## Scope and remaining data limitations

- The study evaluates pairs trading in adjusted underlying equity prices for
  companies belonging to the contemporaneous Indian F&O universe. It does not
  claim to reproduce returns from individual futures contracts.
- The cost sensitivity uses 5, 10 and 20 bps per position change. Contract-level
  futures rolls, margins, lot sizes and expiry handling are outside the study's
  scope because the files contain underlying equity prices rather than futures
  contract histories.
- Annual membership identifies the F&O universe by year, not the exact date of
  an intra-year addition or removal. This should be disclosed as a remaining
  timing limitation if such changes occurred inside a calendar year.
