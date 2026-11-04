"""Frozen research choices selected without examining the 2022+ test results."""

FORMATION_YEARS = 3
TRADING_MONTHS = 6
PAIR_SPECIFICATION = "bidirectional_johansen"
CLASSIFIER_NAME = "logistic_regression"
CLASSIFIER_TARGET = "profitable"
PROBABILITY_THRESHOLD = 0.60
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
MAXIMUM_HOLDING_DAYS = 60
TRANSACTION_COST_BPS = 10.0
DEVELOPMENT_END = "2019-12-31"
VALIDATION_END = "2021-12-31"
LOCKED_TEST_START = "2022-01-01"
LOCKED_TEST_END = "2026-06-30"

# FDR is reported as a robustness result but cannot be the primary filter: at
# 10% it leaves only 14 events overall and no usable development training set.
FDR_LEVEL = 0.10
