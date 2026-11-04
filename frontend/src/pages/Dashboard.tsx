import { useEffect, useState } from "react";
import { getJson } from "../api";
import { MetricCard } from "../components/MetricCard";
import { Plot } from "../components/Plot";

type Overview = {
  rows: number;
  companies: number;
  industries: number;
  first_date: string;
  last_date: string;
  walkforward_windows: number;
  labelled_events: number;
  successful_events: number;
  filter_p_value: number;
  portfolio_at_10bps: Array<Record<string, number | string>>;
};

type Regime = { market_regime: string; events: number; success_rate: number; mean_trade_return: number };
type Daily = { date: string; equity: number };

export function Dashboard() {
  const [overview, setOverview] = useState<Overview>();
  const [regimes, setRegimes] = useState<Regime[]>([]);
  const [baseline, setBaseline] = useState<Daily[]>([]);
  const [ml, setMl] = useState<Daily[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      getJson<Overview>("/api/overview"),
      getJson<Regime[]>("/api/results/regimes"),
      getJson<Daily[]>("/api/results/portfolio-daily/baseline"),
      getJson<Daily[]>("/api/results/portfolio-daily/ml_filtered"),
    ])
      .then(([summary, regimeRows, baselineRows, mlRows]) => {
        setOverview(summary);
        setRegimes(regimeRows);
        setBaseline(baselineRows);
        setMl(mlRows);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) return <div className="error-panel">{error}</div>;
  if (!overview) return <div className="loading">Loading research results…</div>;
  const successRate = overview.successful_events / overview.labelled_events;

  return (
    <main>
      <header className="page-header">
        <div>
          <p className="eyebrow">Frozen out-of-sample research</p>
          <h1>Indian equity pair research</h1>
          <p>Cointegration screening, leakage-safe ML filtering and portfolio evaluation across the contemporaneous F&amp;O universe.</p>
        </div>
        <span className="status-pill">Adjusted prices · annual point-in-time universe</span>
      </header>

      <section className="metric-grid">
        <MetricCard label="Daily observations" value={overview.rows.toLocaleString()} note={`${overview.first_date} — ${overview.last_date}`} />
        <MetricCard label="Companies" value={String(overview.companies)} note={`${overview.industries} industries`} />
        <MetricCard label="Walk-forward windows" value={String(overview.walkforward_windows)} note="2018H1 — 2026H1" />
        <MetricCard label="Labelled signals" value={String(overview.labelled_events)} note={`${(successRate * 100).toFixed(1)}% strict success rate`} />
        <MetricCard label="Filter significance" value={`p = ${overview.filter_p_value.toFixed(3)}`} note="Not statistically significant" tone="warn" />
      </section>

      <section className="section-heading">
        <div><p className="eyebrow">Locked test</p><h2>Portfolio equity after 10 bps costs</h2></div>
        <p>The ML filter reduced losses and drawdown, but did not produce a profitable portfolio.</p>
      </section>
      <Plot
        title="Growth of one unit of capital"
        data={[
          { x: baseline.map((row) => row.date), y: baseline.map((row) => row.equity), type: "scatter", mode: "lines", name: "Baseline", line: { color: "#ef7b6c" } },
          { x: ml.map((row) => row.date), y: ml.map((row) => row.equity), type: "scatter", mode: "lines", name: "ML filtered", line: { color: "#48d6b2", width: 3 } },
        ]}
      />

      <section className="section-heading"><div><p className="eyebrow">Regime comparison</p><h2>The relationship changed after 2021</h2></div></section>
      <div className="table-card">
        <table><thead><tr><th>Regime</th><th>Signals</th><th>Success rate</th><th>Mean trade return</th></tr></thead>
          <tbody>{regimes.map((row) => <tr key={row.market_regime}><td>{row.market_regime.replaceAll("_", " ")}</td><td>{row.events}</td><td>{(row.success_rate * 100).toFixed(1)}%</td><td className={row.mean_trade_return >= 0 ? "positive" : "negative"}>{(row.mean_trade_return * 100).toFixed(2)}%</td></tr>)}</tbody>
        </table>
      </div>
    </main>
  );
}
