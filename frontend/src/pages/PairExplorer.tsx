import { useEffect, useMemo, useState } from "react";
import { getJson, query } from "../api";
import { MetricCard } from "../components/MetricCard";
import { Plot } from "../components/Plot";
import type { IndustryRow, PairAnalysis, PairRow, WindowRow } from "../types";

const specifications = [
  ["forward", "Forward Engle–Granger"],
  ["bidirectional", "Bidirectional Engle–Granger"],
  ["bidirectional_johansen", "Bidirectional + Johansen"],
  ["bidirectional_fdr", "Bidirectional + FDR"],
  ["bidirectional_fdr_johansen", "Strict: FDR + Johansen"],
];

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

  return (
    <main>
      <header className="page-header"><div><p className="eyebrow">Point-in-time research</p><h1>Pair explorer</h1><p>Screen economically similar companies, inspect robust cointegration evidence, then test the frozen relationship out of sample.</p></div></header>
      <section className="control-panel">
        <label>Walk-forward window<select value={windowId} onChange={(e) => setWindowId(e.target.value)}>{windows.map((row) => <option key={row.window_id}>{row.window_id}</option>)}</select></label>
        <label>Industry<select value={industry} onChange={(e) => setIndustry(e.target.value)}>{industries.map((row) => <option key={row.industry} value={row.industry}>{row.industry} · {row.companies} companies</option>)}</select></label>
        <label>Robustness rule<select value={specification} onChange={(e) => setSpecification(e.target.value)}>{specifications.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>Eligible pair<select value={pairKey} onChange={(e) => setPairKey(e.target.value)}>{pairs.map((row) => <option key={`${row.left_code}:${row.right_code}`} value={`${row.left_code}:${row.right_code}`}>{row.left_company} / {row.right_company}</option>)}</select></label>
        <button disabled={!selectedPair || loading} onClick={runAnalysis}>{loading ? "Analysing…" : "Analyse pair"}</button>
      </section>
      <p className="muted">{pairs.length} pairs satisfy the selected statistical rule. FDR may return no pairs because it is intentionally strict.</p>
      {error && <div className="error-panel">{error}</div>}

      {selectedPair && !analysis && <section className="metric-grid compact">
        <MetricCard label="Forward p-value" value={selectedPair.cointegration_p_value.toFixed(4)} />
        <MetricCard label="Reverse p-value" value={selectedPair.cointegration_p_value_reverse.toFixed(4)} />
        <MetricCard label="Johansen" value={selectedPair.johansen_passes_5pct ? "Confirmed" : "Not confirmed"} />
        <MetricCard label="Half-life" value={selectedPair.spread_half_life_days ? `${selectedPair.spread_half_life_days.toFixed(1)} days` : "Unavailable"} />
      </section>}

      {analysis && metrics && backtest && <>
        <section className="metric-grid compact">
          <MetricCard label="Forward p-value" value={Number(metrics.cointegration_p_value).toFixed(4)} />
          <MetricCard label="Reverse p-value" value={Number(metrics.cointegration_p_value_reverse).toFixed(4)} />
          <MetricCard label="Johansen" value={metrics.johansen_passes_5pct ? "Confirmed" : "Not confirmed"} />
          <MetricCard label="Hedge ratio" value={Number(metrics.hedge_ratio).toFixed(3)} />
          <MetricCard label="Half-life" value={`${Number(metrics.spread_half_life_days).toFixed(1)} days`} />
          <MetricCard label="Latest z-score" value={Number(metrics.latest_z_score).toFixed(2)} />
        </section>
        <div className="chart-grid">
          <Plot title="Formation-period normalised prices" data={[
            { x: formation.map((p) => p.date), y: formation.map((p) => Number(p.left_close ?? 0) / leftBase), type: "scatter", mode: "lines", name: selectedPair?.left_company, line: { color: "#71b7ff" } },
            { x: formation.map((p) => p.date), y: formation.map((p) => Number(p.right_close ?? 0) / rightBase), type: "scatter", mode: "lines", name: selectedPair?.right_company, line: { color: "#48d6b2" } },
          ]} />
          <Plot title="Formation spread z-score" data={[{ x: formation.map((p) => p.date), y: formation.map((p) => Number(p.z_score ?? 0)), type: "scatter", mode: "lines", name: "z-score", line: { color: "#d4a6ff" } }]} layout={{ shapes: [-2, 0, 2].map((level) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: level, y1: level, line: { color: level === 0 ? "#63758b" : "#ef7b6c", dash: "dot" } })) }} />
        </div>
        <section className="section-heading"><div><p className="eyebrow">Later six-month period</p><h2>Out-of-sample baseline</h2></div></section>
        <section className="metric-grid compact">
          <MetricCard label="Trades" value={String(backtest.number_of_trades)} />
          <MetricCard label="Net return" value={`${(Number(backtest.net_total_return) * 100).toFixed(2)}%`} />
          <MetricCard label="Sharpe" value={backtest.annualized_sharpe === null ? "n.a." : Number(backtest.annualized_sharpe).toFixed(2)} />
          <MetricCard label="Maximum drawdown" value={`${(Number(backtest.maximum_drawdown) * 100).toFixed(2)}%`} />
          <MetricCard label="Win rate" value={backtest.win_rate === null ? "n.a." : `${(Number(backtest.win_rate) * 100).toFixed(1)}%`} />
        </section>
        <Plot title="Out-of-sample net equity" data={[{ x: analysis.trading_series.map((p) => p.date), y: analysis.trading_series.map((p) => Number(p.net_equity ?? 1)), type: "scatter", mode: "lines", name: "Net equity", line: { color: "#48d6b2", width: 3 } }]} />
        <div className="table-card"><table><thead><tr><th>Entry</th><th>Exit</th><th>Direction</th><th>Reason</th><th>Days</th><th>Net return</th></tr></thead><tbody>{analysis.trades.map((trade, index) => <tr key={index}><td>{String(trade.entry_date).slice(0,10)}</td><td>{String(trade.exit_date).slice(0,10)}</td><td>{trade.direction}</td><td>{String(trade.exit_reason).replaceAll("_", " ")}</td><td>{trade.holding_days}</td><td>{(Number(trade.net_return) * 100).toFixed(2)}%</td></tr>)}</tbody></table></div>
      </>}
    </main>
  );
}
