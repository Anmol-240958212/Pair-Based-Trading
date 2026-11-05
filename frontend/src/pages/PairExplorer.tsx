import { useEffect, useMemo, useState } from "react";
import { getJson, query } from "../api";
import { MetricCard } from "../components/MetricCard";
import { Plot } from "../components/Plot";
import type { FinalModelSignal, IndustryRow, PairAnalysis, PairRow, WindowRow } from "../types";
import "../pair-explorer.css";

const specifications = [
  ["forward", "Forward Engle–Granger"],
  ["bidirectional", "Bidirectional Engle–Granger"],
  ["bidirectional_johansen", "Bidirectional + Johansen"],
  ["bidirectional_fdr", "Bidirectional + FDR"],
  ["bidirectional_fdr_johansen", "Strict: FDR + Johansen"],
];

const percent = (value: number | null, digits = 2) => value === null ? "n.a." : `${(value * 100).toFixed(digits)}%`;
const date = (value: string) => String(value).slice(0, 10);
const decisionLabel = (value: string) => ({
  rejected_confirmation: "Confirmation rejected",
  rejected_probability: "Probability below 60%",
  approved_not_selected: "Approved; portfolio full",
  executed: "Executed",
  not_approved: "Not approved",
}[value] ?? value.replaceAll("_", " "));

function signalTone(signal: FinalModelSignal) {
  if (signal.executed) return "positive";
  if (signal.approved) return "";
  return "negative";
}

export function PairExplorer() {
  const [windows, setWindows] = useState<WindowRow[]>([]);
  const [windowId, setWindowId] = useState("2025H1");
  const [industries, setIndustries] = useState<IndustryRow[]>([]);
  const [industry, setIndustry] = useState("");
  const [specification, setSpecification] = useState("bidirectional_johansen");
  const [pairs, setPairs] = useState<PairRow[]>([]);
  const [pairKey, setPairKey] = useState("");
  const [analysis, setAnalysis] = useState<PairAnalysis>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { getJson<WindowRow[]>("/api/windows").then(setWindows).catch((e: Error) => setError(e.message)); }, []);
  useEffect(() => {
    if (!windowId) return;
    getJson<IndustryRow[]>(`/api/industries?${query({ window_id: windowId })}`).then((rows) => {
      setIndustries(rows); setIndustry(rows[0]?.industry ?? "");
    }).catch((e: Error) => setError(e.message));
  }, [windowId]);
  useEffect(() => {
    if (!windowId || !industry) return;
    getJson<PairRow[]>(`/api/pairs?${query({ window_id: windowId, industry, specification })}`).then((rows) => {
      setPairs(rows); setPairKey(rows[0] ? `${rows[0].left_code}:${rows[0].right_code}` : ""); setAnalysis(undefined);
    }).catch((e: Error) => setError(e.message));
  }, [windowId, industry, specification]);

  const selectedPair = useMemo(() => pairs.find((row) => `${row.left_code}:${row.right_code}` === pairKey), [pairs, pairKey]);
  const runAnalysis = () => {
    if (!selectedPair) return;
    setLoading(true); setError("");
    getJson<PairAnalysis>(`/api/pairs/${windowId}/${selectedPair.left_code}/${selectedPair.right_code}`)
      .then(setAnalysis).catch((e: Error) => setError(e.message)).finally(() => setLoading(false));
  };

  const formation = analysis?.formation_series ?? [];
  const leftBase = formation[0]?.left_close ?? 1;
  const rightBase = formation[0]?.right_close ?? 1;
  const metrics = analysis?.formation_metrics;
  const backtest = analysis?.backtest_metrics;
  const model = analysis?.final_model;
  const latestSignals = model?.signals.slice(-30) ?? [];

  return (
    <main>
      <header className="page-header"><div><p className="eyebrow">Selected research strategy</p><h1>Pair explorer</h1><p>Inspect cointegration, then follow each divergence through the frozen Markov-regime Logistic model, three-day confirmation rule and final trading decision.</p></div><span className="status-pill">Final model · no news veto</span></header>
      <section className="reader-guide">
        <div><span>1</span><p><strong>Select comparable stocks</strong>Choose a time period, industry and pair. Only companies that pass the selected statistical relationship test are listed.</p></div>
        <div><span>2</span><p><strong>Inspect the relationship</strong>The formation charts show what the system learned using earlier data, before the later trading period began.</p></div>
        <div><span>3</span><p><strong>Read the model decision</strong>The ML section explains which divergences were rejected, approved or actually traded—and why.</p></div>
      </section>
      <section className="control-panel">
        <label>Walk-forward window<span className="field-help">Earlier years train the relationship; this six-month period tests it.</span><select value={windowId} onChange={(e) => setWindowId(e.target.value)}>{windows.map((row) => <option key={row.window_id}>{row.window_id}</option>)}</select></label>
        <label>Industry<span className="field-help">Restricts comparison to companies with similar businesses.</span><select value={industry} onChange={(e) => setIndustry(e.target.value)}>{industries.map((row) => <option key={row.industry} value={row.industry}>{row.industry} · {row.companies} companies</option>)}</select></label>
        <label>Cointegration rule<span className="field-help">Controls how much statistical evidence a pair must provide.</span><select value={specification} onChange={(e) => setSpecification(e.target.value)}>{specifications.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>Eligible pair<span className="field-help">Two related stocks assessed as one combined position.</span><select value={pairKey} onChange={(e) => setPairKey(e.target.value)}>{pairs.map((row) => <option key={`${row.left_code}:${row.right_code}`} value={`${row.left_code}:${row.right_code}`}>{row.left_company} / {row.right_company}{row.model_executed_trades ? ` · ${row.model_executed_trades} ML trades` : row.model_candidate_signals ? ` · ${row.model_candidate_signals} signals` : ""}</option>)}</select></label>
        <button disabled={!selectedPair || loading} onClick={runAnalysis}>{loading ? "Running model…" : "Analyse final model"}</button>
      </section>
      <p className="muted">{pairs.length} pairs satisfy the selected statistical rule. Pairs with final-model trades or signals are shown first.</p>
      {error && <div className="error-panel">{error}</div>}

      {selectedPair && !analysis && <section className="metric-grid compact">
        <MetricCard label="Forward p-value" value={selectedPair.cointegration_p_value.toFixed(4)} note="Smaller values mean stronger evidence of a stable long-run relationship." />
        <MetricCard label="Reverse p-value" value={selectedPair.cointegration_p_value_reverse.toFixed(4)} note="The same test with the stocks' regression roles reversed." />
        <MetricCard label="Johansen" value={selectedPair.johansen_passes_5pct ? "Confirmed" : "Not confirmed"} note="An independent robustness check for the long-run relationship." />
        <MetricCard label="Final-model history" value={selectedPair.model_available ? `${selectedPair.model_candidate_signals} signals` : "No signal"} note={selectedPair.model_executed_trades ? `${selectedPair.model_executed_trades} executed trades` : "Select Analyse final model for details"} />
      </section>}

      {analysis && metrics && backtest && model && <>
        <section className="metric-grid compact">
          <MetricCard label="Forward p-value" value={Number(metrics.cointegration_p_value).toFixed(4)} note="Lower than 0.05 passes the initial cointegration test." />
          <MetricCard label="Reverse p-value" value={Number(metrics.cointegration_p_value_reverse).toFixed(4)} note="Checks that the result is not dependent on regression direction." />
          <MetricCard label="Johansen" value={metrics.johansen_passes_5pct ? "Confirmed" : "Not confirmed"} note="A second method confirming the relationship." />
          <MetricCard label="Hedge ratio" value={Number(metrics.hedge_ratio).toFixed(3)} note="How much of the second stock balances one unit of the first." />
          <MetricCard label="Spread half-life" value={`${Number(metrics.spread_half_life_days).toFixed(1)} days`} note="Estimated time for half of a price divergence to disappear." />
          <MetricCard label="Latest formation z-score" value={Number(metrics.latest_z_score).toFixed(2)} note="Distance from the historical spread average, measured in standard deviations." />
        </section>
        <div className="chart-grid">
          <Plot title="Formation-period normalised prices" data={[
            { x: formation.map((p) => p.date), y: formation.map((p) => Number(p.left_close ?? 0) / leftBase), type: "scatter", mode: "lines", name: selectedPair?.left_company, line: { color: "#71b7ff" } },
            { x: formation.map((p) => p.date), y: formation.map((p) => Number(p.right_close ?? 0) / rightBase), type: "scatter", mode: "lines", name: selectedPair?.right_company, line: { color: "#48d6b2" } },
          ]} description="Both prices start at 1, making their relative growth paths comparable even when the actual share prices differ." />
          <Plot title="Formation spread z-score" description="The spread is the hedge-adjusted price difference. Zero is its historical average; ±2 indicates an unusually large divergence." data={[{ x: formation.map((p) => p.date), y: formation.map((p) => Number(p.z_score ?? 0)), type: "scatter", mode: "lines", name: "z-score", line: { color: "#d4a6ff" } }]} layout={{ shapes: [-2, 0, 2].map((level) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: level, y1: level, line: { color: level === 0 ? "#63758b" : "#ef7b6c", dash: "dot" } })) }} />
        </div>

        <section className="section-heading"><div><p className="eyebrow">Later six-month period</p><h2>Final ML strategy</h2></div><p>The displayed decisions are the saved walk-forward outputs used in Experiment 5, not a model refitted using later outcomes.</p></section>
        <section className="strategy-strip">
          <div><span>Model</span><strong>Markov + Logistic</strong></div>
          <div><span>Confirmation</span><strong>{model.config.observation_days} trading days</strong></div>
          <div><span>Approval</span><strong>≥ {percent(model.config.probability_threshold, 0)}</strong></div>
          <div><span>Payoff</span><strong>+{percent(model.config.take_profit_return, 0)} / {percent(model.config.stop_loss_return, 0)}</strong></div>
          <div><span>Horizon</span><strong>{model.config.maximum_holding_days} days</strong></div>
          <div><span>Cost</span><strong>{model.config.transaction_cost_bps_per_side} bps/side</strong></div>
        </section>

        {!model.available ? <div className="callout"><strong>No final-model signal for this pair and window</strong><p>{model.reason === "outside_post_2022_evaluation_period" ? "The selected final-model comparison begins in 2022; earlier windows remain available for formation and baseline inspection." : "The pair passed the selected formation test but did not create an eligible three-day divergence row in this trading window."}</p></div> : <>
          <section className="metric-grid compact">
            <MetricCard label="Candidate divergences" value={String(model.summary.candidate_signals)} note="Dates when the pair moved far enough apart to be evaluated." />
            <MetricCard label="Passed confirmation" value={String(model.summary.confirmation_passed)} note="Signals still supported after observing the next three trading days." />
            <MetricCard label="Approved signals" value={String(model.summary.approved_signals)} note="Confirmation + probability" />
            <MetricCard label="Executed trades" value={String(model.summary.executed_trades)} note="After portfolio-capacity control" />
            <MetricCard label="Pair trade return" value={percent(model.summary.compounded_pair_return)} note="Compounded result of this pair's executed trades after assumed costs." tone={model.summary.compounded_pair_return >= 0 ? "good" : "warn"} />
            <MetricCard label="Trade win rate" value={percent(model.summary.win_rate, 1)} note={model.summary.executed_trades ? `${model.summary.profitable_trades} of ${model.summary.executed_trades}` : "No executed trade"} />
          </section>
          <Plot title="Model probability at each three-day decision" data={[
            { x: model.signals.map((row) => row.signal_date), y: model.signals.map((row) => row.predicted_probability * 100), type: "scatter", mode: "markers", name: "All decisions", marker: { color: model.signals.map((row) => row.executed ? "#48d6b2" : row.approved ? "#71b7ff" : "#63758b"), size: model.signals.map((row) => row.executed ? 11 : 7) } },
          ]} description="Each dot is the model's estimated chance of a profitable reversion after the observation period. The dotted line is the 60% approval threshold; confirmation must also pass." layout={{ yaxis: { title: { text: "Predicted probability (%)" }, range: [0, 100] }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: model.config.probability_threshold * 100, y1: model.config.probability_threshold * 100, line: { color: "#ef7b6c", dash: "dot" } }] }} />
          {model.equity_series.length > 0 && <Plot title="Realised equity from this pair's executed ML trades" description="Starts at 1.00 and compounds only trades that passed every filter and were selected within the portfolio-position limit." data={[{ x: model.equity_series.map((row) => row.date), y: model.equity_series.map((row) => row.net_equity), type: "scatter", mode: "lines+markers", name: "Pair equity", line: { color: "#48d6b2", width: 3 } }]} />}
          <div className="callout table-guide"><strong>How to read the decision table</strong><p><b>Initial trigger</b> is when divergence was first detected. <b>3-day decision</b> is when the model was allowed to act. “Approved; portfolio full” means the signal passed, but another active position prevented execution. Observed outcomes are shown for historical evaluation and were not available to the model at decision time.</p></div>
          <div className="table-card"><table><thead><tr><th>Initial trigger</th><th>3-day decision</th><th>Initial z</th><th>Confirmed</th><th>ML probability</th><th>Decision</th><th>Observed outcome</th><th>Net return</th></tr></thead><tbody>{latestSignals.map((signal) => <tr key={signal.event_id}><td>{date(signal.trigger_date)}</td><td>{date(signal.signal_date)}</td><td>{signal.initial_z_score.toFixed(2)}</td><td>{signal.passes_confirmation ? "Pass" : "Reject"}</td><td>{percent(signal.predicted_probability, 1)}</td><td className={signalTone(signal)}>{decisionLabel(signal.decision)}</td><td>{signal.barrier_outcome.replaceAll("_", " ")}</td><td className={signal.trade_net_return >= 0 ? "positive" : "negative"}>{percent(signal.trade_net_return)}</td></tr>)}</tbody></table></div>
          {model.signals.length > latestSignals.length && <p className="muted">Showing the latest {latestSignals.length} of {model.signals.length} model decisions.</p>}
        </>}

        <section className="section-heading"><div><p className="eyebrow">Reference strategy</p><h2>Classical z-score baseline</h2></div><p>This remains visible as a comparison. It is not the selected final model.</p></section>
        <section className="metric-grid compact">
          <MetricCard label="Trades" value={String(backtest.number_of_trades)} />
          <MetricCard label="Net return" value={percent(Number(backtest.net_total_return))} note="Cumulative result after the baseline's assumed trading costs." />
          <MetricCard label="Sharpe ratio" value={backtest.annualized_sharpe === null ? "n.a." : Number(backtest.annualized_sharpe).toFixed(2)} note="Return relative to variability; higher is better." />
          <MetricCard label="Maximum drawdown" value={percent(Number(backtest.maximum_drawdown))} note="Largest decline from a previous equity peak." />
          <MetricCard label="Win rate" value={backtest.win_rate === null ? "n.a." : percent(Number(backtest.win_rate), 1)} note="Percentage of completed trades with positive net returns." />
        </section>
        <Plot title="Baseline out-of-sample net equity" description="A reference strategy using only the classical z-score entry and exit rules, without the ML and three-day confirmation filters." data={[{ x: analysis.trading_series.map((p) => p.date), y: analysis.trading_series.map((p) => Number(p.net_equity ?? 1)), type: "scatter", mode: "lines", name: "Baseline equity", line: { color: "#71b7ff", width: 3 } }]} />
        <div className="table-card"><table><thead><tr><th>Entry</th><th>Exit</th><th>Direction</th><th>Reason</th><th>Days</th><th>Net return</th></tr></thead><tbody>{analysis.trades.map((trade, index) => <tr key={index}><td>{date(String(trade.entry_date))}</td><td>{date(String(trade.exit_date))}</td><td>{trade.direction}</td><td>{String(trade.exit_reason).replaceAll("_", " ")}</td><td>{trade.holding_days}</td><td>{percent(Number(trade.net_return))}</td></tr>)}</tbody></table></div>
      </>}
    </main>
  );
}
