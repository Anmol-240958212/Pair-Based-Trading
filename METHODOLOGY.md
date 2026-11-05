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

## Experiment 4: mathematical and regime-model comparison

Experiment 4 leaves the Experiment 3 rows, targets, walk-forward periods,
portfolio constraints and costs unchanged. It compares ordinary Logistic
Regression against three alternative mathematical specifications:

- shrinkage Linear Discriminant Analysis, which estimates class-specific
  centres with a regularised shared covariance matrix;
- Gaussian Naive Bayes, which models each feature conditionally as an
  independent Gaussian within each outcome class;
- a two-state Gaussian Markov filter whose market and industry emissions are
  estimated on past dates, with state transitions estimated from the training
  sequence, followed by a one-sided forward filter and Logistic Regression.

The Markov probabilities measure the probability of the higher-volatility
state and the historically more favourable reversion state. They are computed
chronologically: future observations do not revise earlier state estimates.
Every classifier uses the same pre-declared 0.60 probability cutoff. Models are
refitted before each half-year using only labels completed before that window.

At 10 bps per position change, the post-2022 development results are:

| Strategy | Trades | Total return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Logistic Regression | 140 | +0.89% | 0.155 | -2.85% |
| Shrinkage LDA | 528 | -1.26% | -0.095 | -4.97% |
| Gaussian Naive Bayes | 668 | -4.90% | -0.364 | -10.83% |
| Markov regime + Logistic | 149 | +1.31% | 0.222 | -2.80% |

The Markov variant is the best development model, but it loses 1.63% at 20
bps and its 20-day moving-block bootstrap p-value is 0.328. It therefore does
not establish statistically reliable profitability and still requires a new
completed holdout period.

## Experiment 5: delayed common-factor confirmation

Experiment 5 tests whether the model should observe a divergence before
trading. Candidate observation periods are zero, one and three trading days.
For delayed candidates, the eventual signal and entry dates are shifted
forward; labels and returns begin only after the later entry, preventing the
observation interval from being counted as tradable profit.

The confirmation layer records same-direction movement, alignment with the
pair's industry return, one-day lead-lag sign agreement, the share of movement
attributable to the common industry factor, abnormal turnover, standardized
industry-residual shocks and the change in absolute spread z-score. A delayed
signal passes the economic gate only when at least half of observed stock
moves share a direction, at least one-third align with the industry factor, no
standardized idiosyncratic shock exceeds four and the spread has not
accelerated outward by more than 0.25 z-score units.

The 2020-2021 validation period selected zero delay: its approved mean return
was 0.310%, compared with 0.162% for one day and 0.247% for three days. Thus,
the originally defined selection procedure did not support waiting.

The later post-2022 diagnostic nevertheless produced:

| Observation | Trades | Return at 10 bps | Sharpe | Drawdown | Bootstrap p |
|---|---:|---:|---:|---:|---:|
| 0 days | 150 | +1.42% | 0.240 | -2.69% | 0.314 |
| 1 day | 80 | +0.08% | 0.024 | -2.76% | 0.483 |
| 3 days | 67 | +2.48% | 0.721 | -0.78% | 0.033 |

The three-day version also remained positive at 20 bps (+1.12%). This is the
strongest diagnostic result so far, but selecting it after viewing the
post-2022 comparison would be post-hoc. It should be frozen as a candidate and
tested on completed 2026H2 or later data before being adopted as the primary
strategy.

## Experiment 6: point-in-time NSE announcement veto

Experiment 6 treats official exchange announcements as an event-risk feed,
not as a directional trading model. Day zero is the confirmation/decision day;
the interval `[-3, 0]` begins at the initial divergence trigger and includes
only announcements broadcast by the decision date. Entry remains the next
trading day. The archive contains 118,480 timestamped announcements covering
111 of the 112 relevant companies. MCX returned no archive records, so rows
involving it are excluded from both sides of the news comparison.

Subjects and announcement text are classified with a frozen material-event
dictionary covering results, board outcomes, acquisitions, contracts, credit
ratings, management changes, defaults, litigation, regulatory events,
insolvency, buybacks, dividends and restructuring. A trade is vetoed when a
material event appears for exactly one pair member during the observation
window. This deliberately uses news as a structural-break warning rather than
trying to infer a buy or sell direction from text.

In 2020-2021 validation, the veto reduced approved mean return from 0.247% to
0.225%, so it was not validation-supported. In the later diagnostic it reduced
portfolio trades from 67 to 41, improved win rate from 65.7% to 68.3%, reduced
drawdown from -0.78% to -0.51% and returned +2.16% at 10 bps and +1.33% at 20
bps. This is a risk-control candidate, not a validated replacement rule.

## Experiment 7: payoff barrier comparison

Changing a stop changes which observations become positive labels and when
those labels become available. Therefore, Experiment 7 reconstructs each
future spread-return path, exit date, target and label-availability date before
retraining the three-day Markov/Logistic model. The compared net barriers are
+1%/-2%, +/-0.5%, +/-1% and +1.5%/-1%.

The 2020-2021 validation period retained +1%/-2%; its approved mean return was
0.247%. The corresponding values were -0.127% for +/-0.5%, 0.004% for +/-1%
and -0.074% for +1.5%/-1%.

Later portfolio results at 10 bps were:

| Net target / stop | Trades | Total return | Sharpe | Drawdown | Bootstrap p |
|---|---:|---:|---:|---:|---:|
| +1% / -2% | 67 | +2.48% | 0.721 | -0.78% | 0.033 |
| +0.5% / -0.5% | 30 | +0.70% | 0.340 | -0.72% | 0.183 |
| +1% / -1% | 44 | +0.97% | 0.410 | -0.77% | 0.175 |
| +1.5% / -1% | 53 | +0.85% | 0.277 | -1.45% | 0.265 |

Equal barriers are conceptually reasonable but did not improve this strategy.
The tighter stops frequently closed temporarily widening spreads before their
later reversion. The original asymmetric rule remains the selected candidate,
subject to confirmation on genuinely new data.

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
