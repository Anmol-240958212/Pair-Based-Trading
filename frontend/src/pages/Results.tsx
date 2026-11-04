import { useEffect, useState } from "react";
import { getJson } from "../api";
import { MetricCard } from "../components/MetricCard";
import { Plot } from "../components/Plot";

type Locked = { window_id: string; baseline_trades: number; ml_approved_trades: number; baseline_mean_trade_return: number; ml_mean_trade_return: number; baseline_success_rate: number; ml_success_rate: number };
type Portfolio = { strategy: string; transaction_cost_bps: number; total_return: number; annualized_sharpe: number; maximum_drawdown: number; accepted_trades: number };
type Significance = { observed_ml_mean_return: number; random_filter_mean_return: number; random_filter_95pct_lower: number; random_filter_95pct_upper: number; one_sided_p_value: number; repetitions: number };

export function Results() {
  const [locked, setLocked] = useState<Locked[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio[]>([]);
  const [significance, setSignificance] = useState<Significance>();
  useEffect(() => { Promise.all([getJson<Locked[]>("/api/results/locked-test"), getJson<Portfolio[]>("/api/results/portfolio"), getJson<Significance>("/api/results/significance")]).then(([a,b,c]) => { setLocked(a); setPortfolio(b); setSignificance(c); }); }, []);
  const ten = portfolio.filter((row) => row.transaction_cost_bps === 10);
  const baseline = ten.find((row) => row.strategy === "baseline");
  const ml = ten.find((row) => row.strategy === "ml_filtered");
  return <main>
    <header className="page-header"><div><p className="eyebrow">2022H1 — 2026H1</p><h1>Locked-test results</h1><p>The specification was frozen before these outcomes were opened. The result is informative, not a profitability claim.</p></div></header>
    {baseline && ml && significance && <section className="metric-grid">
      <MetricCard label="Baseline return" value={`${(baseline.total_return*100).toFixed(2)}%`} tone="warn" />
      <MetricCard label="ML-filtered return" value={`${(ml.total_return*100).toFixed(2)}%`} tone="warn" />
      <MetricCard label="Drawdown reduction" value={`${((Math.abs(baseline.maximum_drawdown)-Math.abs(ml.maximum_drawdown))*100).toFixed(2)} pp`} />
      <MetricCard label="Permutation p-value" value={significance.one_sided_p_value.toFixed(3)} note={`${significance.repetitions.toLocaleString()} random filters`} tone="warn" />
    </section>}
    <Plot title="Mean trade return by test window" data={[
      { x: locked.map(r=>r.window_id), y: locked.map(r=>r.baseline_mean_trade_return*100), type:"bar", name:"Baseline", marker:{color:"#ef7b6c"}},
      { x: locked.map(r=>r.window_id), y: locked.map(r=>(r.ml_mean_trade_return ?? 0)*100), type:"bar", name:"ML approved", marker:{color:"#48d6b2"}},
    ]} layout={{barmode:"group", yaxis:{title:{text:"Mean return (%)"}, gridcolor:"#1f3045"}}} />
    <Plot title="Transaction-cost sensitivity" data={[
      { x: portfolio.filter(r=>r.strategy==="baseline").map(r=>r.transaction_cost_bps), y: portfolio.filter(r=>r.strategy==="baseline").map(r=>r.total_return*100), type:"scatter", mode:"lines+markers", name:"Baseline", line:{color:"#ef7b6c"}},
      { x: portfolio.filter(r=>r.strategy==="ml_filtered").map(r=>r.transaction_cost_bps), y: portfolio.filter(r=>r.strategy==="ml_filtered").map(r=>r.total_return*100), type:"scatter", mode:"lines+markers", name:"ML filtered", line:{color:"#48d6b2"}},
    ]} layout={{xaxis:{title:{text:"Cost per position change (bps)"}, gridcolor:"#1f3045"}, yaxis:{title:{text:"Total return (%)"}, gridcolor:"#1f3045"}}} />
    <div className="callout warn"><strong>Correct interpretation</strong><p>The filter reduced losses and maximum drawdown, but the 10 bps portfolio remained negative and the improvement was not statistically significant. The final test must not be reused for further tuning.</p></div>
  </main>;
}
