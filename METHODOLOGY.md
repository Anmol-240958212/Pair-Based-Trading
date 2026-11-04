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

## Experiment 2: post-hoc development study

After the locked result had been viewed, a separately labelled experiment was
implemented. It must not be presented as a second untouched holdout. The new
components are:

- no entry unless all 60 maximum holding days remain in the half-year;
- a 252-observation rolling OLS hedge ratio estimated using prior data only;
- formation hedge-ratio drift, spread-mean shift, volatility change and zero
  crossings;
- signal-date hedge drift, z-score volatility and recent zero crossings;
- a Logistic Regression estimate of reversion probability;
- a Ridge estimate of expected net trade return;
- approval only when both model thresholds and the structural-stability gate
  pass;
- ranking by expected return, reversion probability and signal-date risk;
- static inverse-volatility sizing with company-overlap and ten-position caps.

A fixed-versus-rolling hedge ablation was decided on the 2020-2021 validation
period. The rolling specification produced validation ROC-AUC 0.716 versus
0.422 for the fixed hedge and was therefore retained. This choice was not made
from the 2022+ comparison.

The rolling/boundary baseline produced 121 comparison trades; the two-stage
rule approved 13. At 10 bps, volatility-scaled portfolio results were:

| Strategy | Accepted trades | Total return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Dynamic/boundary baseline | 95 | -5.69% | -0.612 | -9.04% |
| Two-stage filtered | 13 | -1.19% | -0.227 | -2.79% |

The event-level permutation p-value was 0.787. Experiment 2 again reduced
exposure and drawdown but did not create a profitable or statistically reliable
strategy. Completed 2026H2 or later data is required for a genuinely fresh
test of any future specification.

## Experiment 3: daily pair-panel development study

Experiment 3 addresses the small effective ML sample without changing the
original locked record. Formation coverage is relaxed from 95% to 90%, while
the 500-observation minimum remains. This adds only zero to five companies per
window; the material sample increase comes from daily labelling.

Each robust bidirectional/Johansen pair-day with |z| >= 1 becomes one row. The
signal is executed on the next day and observed for at most 20 trading days.
The positive target is reached when a net +1% take-profit barrier occurs before
a -2% stop-loss barrier. If neither barrier occurs, the horizon return and date
complete the label. Every label becomes available only on its recorded exit
date, which prevents unfinished outcomes from entering later model training.

The resulting panel has 19,217 rows, compared with 230 completed trades in
Experiment 2. Initial training uses 3,747 rows from 2018-2019 and validation
uses 4,763 rows from 2020-2021. Logistic Regression and histogram Gradient
Boosting are refitted before every later half-year using only available labels.
Daily candidate signals are converted into executable portfolios by preventing
overlapping use of a company and limiting the portfolio to ten positions.

At 10 bps per position change, the post-2022 development comparison is:

| Strategy | Trades | Total return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Unfiltered daily signals | 1,698 | -19.33% | -0.999 | -23.87% |
| Logistic Regression | 140 | +0.89% | 0.155 | -2.85% |
| Histogram Gradient Boosting | 756 | -4.98% | -0.365 | -8.40% |

The Logistic result is positive at 5 and 10 bps but negative at 20 bps. A
20-day moving-block bootstrap gives p = 0.388 for positive mean daily return.
The result is therefore promising as a filtering improvement but is neither
cost-robust nor statistically significant and is not a fresh holdout.

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
