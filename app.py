from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.data import load_price_data, load_static_data, price_data_audit
from pairs_trading.backtest import backtest_pair
from pairs_trading.statistics import analyze_pair
from pairs_trading.universe import eligible_companies, same_industry_pairs


st.set_page_config(page_title="Indian Equity Pair Research", layout="wide")
st.title("Indian Equity Pair Research")
st.caption("Formation-window screening followed by a strictly later out-of-sample trading test.")


@st.cache_data(show_spinner="Loading annual price files...")
def cached_prices(years: tuple[int, ...]) -> pd.DataFrame:
    return load_price_data(ROOT, years)


@st.cache_data(show_spinner=False)
def cached_static():
    return load_static_data(ROOT)


available_years = list(range(2015, 2027))
with st.sidebar:
    st.header("Formation window")
    formation_year_range = st.select_slider(
        "Formation years", options=available_years[:-1], value=(2017, 2019)
    )
    formation_years = tuple(range(formation_year_range[0], formation_year_range[1] + 1))
    later_years = [year for year in available_years if year > formation_year_range[1]]
    trading_year = st.selectbox("Out-of-sample trading year", later_years, index=0)
    selected_years = tuple(sorted(set(formation_years + (trading_year,))))
    minimum_coverage = st.slider("Minimum data coverage", 0.80, 1.00, 0.95, 0.01)
    minimum_observations = st.number_input("Minimum observations", 60, 1000, 200, 20)
    minimum_turnover = st.number_input("Minimum median turnover (Rs. thousand)", 0.0, value=0.0, step=1000.0)

    st.header("Backtest rules")
    entry_z = st.number_input("Entry |z-score|", 0.5, 5.0, 2.0, 0.1)
    exit_z = st.number_input("Exit |z-score|", 0.0, 4.0, 0.5, 0.1)
    stop_z = st.number_input("Stop |z-score|", 1.0, 10.0, 4.0, 0.1)
    maximum_holding_days = st.number_input("Maximum holding days", 1, 252, 60, 1)
    transaction_cost_bps = st.number_input("Transaction cost (bps per turnover)", 0.0, 100.0, 10.0, 1.0)

prices = cached_prices(selected_years)
company_master, static_audit = cached_static()

formation_prices = prices.loc[prices["source_year"].isin(formation_years)]
trading_prices = prices.loc[prices["source_year"].eq(trading_year)]
minimum_date = formation_prices["date"].min().date()
maximum_date = formation_prices["date"].max().date()
trading_minimum_date = trading_prices["date"].min().date()
trading_maximum_date = trading_prices["date"].max().date()
default_trading_end = min(date(trading_year, 6, 30), trading_maximum_date)
with st.sidebar:
    formation_dates = st.date_input(
        "Exact formation dates",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
        key="formation_dates",
    )
    trading_dates = st.date_input(
        "Exact out-of-sample dates",
        value=(trading_minimum_date, default_trading_end),
        min_value=trading_minimum_date,
        max_value=trading_maximum_date,
        key="trading_dates",
    )

if not isinstance(formation_dates, (tuple, list)) or len(formation_dates) != 2:
    st.info("Select both a formation start date and end date.")
    st.stop()

formation_start, formation_end = formation_dates
if not isinstance(trading_dates, (tuple, list)) or len(trading_dates) != 2:
    st.info("Select both an out-of-sample start date and end date.")
    st.stop()
trading_start, trading_end = trading_dates
eligible = eligible_companies(
    prices,
    company_master,
    formation_start,
    formation_end,
    minimum_coverage=minimum_coverage,
    minimum_observations=int(minimum_observations),
    minimum_median_turnover=float(minimum_turnover),
)

audit = price_data_audit(prices)
metric_columns = st.columns(4)
metric_columns[0].metric("Loaded rows", f"{audit['rows']:,}")
metric_columns[1].metric("Loaded companies", f"{audit['companies']:,}")
metric_columns[2].metric("Price-eligible companies", f"{len(eligible):,}")
metric_columns[3].metric("Eligible with industry data", f"{eligible['industry'].notna().sum():,}")

if eligible.empty:
    st.warning("No companies meet the selected coverage, observation, and liquidity rules.")
    st.stop()

industry_counts = eligible.dropna(subset=["industry"]).groupby("industry").size().sort_index()
pairable_industries = industry_counts.loc[industry_counts >= 2]
if pairable_industries.empty:
    st.warning("No industry has at least two companies under the selected eligibility rules.")
    st.stop()

industry = st.selectbox(
    "Industry",
    pairable_industries.index.tolist(),
    format_func=lambda value: f"{value} ({pairable_industries[value]} companies)",
)
industry_companies = eligible.loc[eligible["industry"].eq(industry)].copy()

st.subheader("Eligible companies in the selected industry")
st.dataframe(
    industry_companies[
        ["company_name", "nse_symbol", "sector", "industry", "observations", "coverage", "median_turnover_rs_thousand"]
    ],
    width="stretch",
    hide_index=True,
    column_config={"coverage": st.column_config.NumberColumn(format="%.1%%")},
)

pairs = same_industry_pairs(eligible, industry)
st.caption(f"{len(industry_companies)} eligible companies generate {len(pairs)} same-industry candidate pairs.")
if pairs.empty:
    st.info("At least two eligible companies are required in an industry.")
    st.stop()

left_lookup = dict(zip(industry_companies["company_name"], industry_companies["capitaline_code"]))
left_name = st.selectbox("First company", sorted(left_lookup))
right_options = [name for name in sorted(left_lookup) if name != left_name]
right_name = st.selectbox("Second company", right_options)

try:
    metrics, detail = analyze_pair(
        prices,
        int(left_lookup[left_name]),
        int(left_lookup[right_name]),
        formation_start,
        formation_end,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

st.subheader("Formation-period relationship")
metrics_row = st.columns(5)
metrics_row[0].metric("Cointegration p-value", f"{metrics['cointegration_p_value']:.4f}")
metrics_row[1].metric("Return correlation", f"{metrics['return_correlation']:.3f}")
metrics_row[2].metric("Hedge ratio", f"{metrics['hedge_ratio']:.3f}")
half_life = metrics["spread_half_life_days"]
metrics_row[3].metric("Spread half-life", "Not mean reverting" if pd.isna(half_life) else f"{half_life:.1f} days")
metrics_row[4].metric("Latest z-score", f"{metrics['latest_z_score']:.2f}")

normalised = detail[["date", "left_close", "right_close"]].copy()
normalised["left_close"] /= normalised["left_close"].iloc[0]
normalised["right_close"] /= normalised["right_close"].iloc[0]
normalised = normalised.rename(columns={"left_close": left_name, "right_close": right_name})
price_chart = px.line(normalised, x="date", y=[left_name, right_name], title="Normalised prices")
st.plotly_chart(price_chart, width="stretch")

spread_chart = go.Figure()
spread_chart.add_trace(go.Scatter(x=detail["date"], y=detail["z_score"], name="Spread z-score"))
for level, colour in [(2, "#c0392b"), (0, "#7f8c8d"), (-2, "#2980b9")]:
    spread_chart.add_hline(y=level, line_dash="dash", line_color=colour)
spread_chart.update_layout(title="Formation-period spread z-score", xaxis_title="Date", yaxis_title="Z-score")
st.plotly_chart(spread_chart, width="stretch")

st.subheader("Out-of-sample baseline backtest")
st.caption(
    f"Formation estimates are frozen on {formation_start} to {formation_end}; "
    f"the strategy is evaluated on {trading_start} to {trading_end}."
)
try:
    backtest_metrics, daily_backtest, trade_log = backtest_pair(
        prices,
        int(left_lookup[left_name]),
        int(left_lookup[right_name]),
        formation_start,
        formation_end,
        trading_start,
        trading_end,
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        stop_z=float(stop_z),
        maximum_holding_days=int(maximum_holding_days),
        transaction_cost_bps=float(transaction_cost_bps),
    )
except ValueError as error:
    st.warning(f"Backtest unavailable: {error}")
else:
    result_columns = st.columns(6)
    result_columns[0].metric("Trades", f"{backtest_metrics['number_of_trades']}")
    win_rate = backtest_metrics["win_rate"]
    result_columns[1].metric("Win rate", "n.a." if pd.isna(win_rate) else f"{win_rate:.1%}")
    result_columns[2].metric("Net return", f"{backtest_metrics['net_total_return']:.2%}")
    sharpe = backtest_metrics["annualized_sharpe"]
    result_columns[3].metric("Annualized Sharpe", "n.a." if pd.isna(sharpe) else f"{sharpe:.2f}")
    result_columns[4].metric("Maximum drawdown", f"{backtest_metrics['maximum_drawdown']:.2%}")
    convergence = backtest_metrics["convergence_rate"]
    result_columns[5].metric(
        "Convergence rate", "n.a." if pd.isna(convergence) else f"{convergence:.1%}"
    )

    equity_chart = px.line(
        daily_backtest,
        x="date",
        y=["gross_equity", "net_equity"],
        title="Out-of-sample portfolio value",
        labels={"value": "Growth of 1.00", "variable": "Series"},
    )
    st.plotly_chart(equity_chart, width="stretch")

    test_spread_chart = go.Figure()
    test_spread_chart.add_trace(
        go.Scatter(
            x=daily_backtest["date"],
            y=daily_backtest["z_score"],
            name="Out-of-sample z-score",
        )
    )
    for level, colour in [
        (float(entry_z), "#c0392b"),
        (float(exit_z), "#7f8c8d"),
        (-float(exit_z), "#7f8c8d"),
        (-float(entry_z), "#2980b9"),
    ]:
        test_spread_chart.add_hline(y=level, line_dash="dash", line_color=colour)
    test_spread_chart.update_layout(
        title="Out-of-sample spread z-score",
        xaxis_title="Date",
        yaxis_title="Z-score",
    )
    st.plotly_chart(test_spread_chart, width="stretch")

    st.write("Trade log")
    if trade_log.empty:
        st.info("No entry signal occurred under the selected thresholds.")
    else:
        st.dataframe(
            trade_log,
            width="stretch",
            hide_index=True,
            column_config={
                "net_return": st.column_config.NumberColumn(format="%.2%%"),
                "entry_z_score": st.column_config.NumberColumn(format="%.2f"),
                "exit_z_score": st.column_config.NumberColumn(format="%.2f"),
            },
        )

with st.expander("Source-data notes"):
    duplicate_static = int((static_audit["static_record_count"] > 1).sum())
    conflicts = int(static_audit["classification_conflict"].sum())
    price_codes = set(prices["capitaline_code"].unique())
    static_codes = set(company_master["capitaline_code"].unique())
    st.write(f"Static company codes with multiple records: {duplicate_static}")
    st.write(f"Static company codes with conflicting sector or industry classifications: {conflicts}")
    st.write(f"Loaded price companies without a matching static record: {len(price_codes - static_codes)}")
    st.write("Duplicate or contradictory source records will be resolved explicitly during the formal data-audit phase.")

