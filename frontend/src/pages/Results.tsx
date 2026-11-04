import { useEffect, useState } from "react";
import { getJson } from "../api";
import { MetricCard } from "../components/MetricCard";
import { Plot } from "../components/Plot";

type Locked = { window_id: string; baseline_trades: number; ml_approved_trades: number; baseline_mean_trade_return: number; ml_mean_trade_return: number; baseline_success_rate: number; ml_success_rate: number };
type Portfolio = { strategy: string; transaction_cost_bps: number; total_return: number; annualized_sharpe: number; maximum_drawdown: number; accepted_trades: number };
type Significance = { observed_ml_mean_return: number; random_filter_mean_return: number; random_filter_95pct_lower: number; random_filter_95pct_upper: number; one_sided_p_value: number; repetitions: number };
type Experiment2 = {
  interpretation: string;
  validation: { reversion_roc_auc: number; reversion_precision: number; approved_trades: number; approved_mean_trade_return: number };
  significance: Significance;
  portfolio: Portfolio[];
  windows: Array<{ window_id: string; all_mean_trade_return: number; approved_mean_trade_return: number | null }>;
};
type Experiment3 = {
  interpretation: string;
  panel: Array<{ daily_signal_rows: number }>;
  validation: Array<{ model: string; training_rows: number; validation_rows: number; roc_auc: number; approved_mean_return: number }>;
  portfolio: Portfolio[];
  significance: Array<{ strategy: string; one_sided_p_value: number }>;
  windows: Array<{ strategy: string; window_id: string; mean_trade_return: number; accepted_trades: number }>;
};

export function Results() {
  const [locked, setLocked] = useState<Locked[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio[]>([]);
  const [significance, setSignificance] = useState<Significance>();
  const [experiment2, setExperiment2] = useState<Experiment2>();
  const [experiment3, setExperiment3] = useState<Experiment3>();
  useEffect(() => { Promise.all([getJson<Locked[]>("/api/results/locked-test"), getJson<Portfolio[]>("/api/results/portfolio"), getJson<Significance>("/api/results/significance"), getJson<Experiment2>("/api/results/experiment2"), getJson<Experiment3>("/api/results/experiment3")]).then(([a,b,c,d,e]) => { setLocked(a); setPortfolio(b); setSignificance(c); setExperiment2(d); setExperiment3(e); }); }, []);
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
    {experiment2 && (() => {
      const rows = experiment2.portfolio.filter((row) => row.transaction_cost_bps === 10);
      const baseline2 = rows.find((row) => row.strategy === "dynamic_boundary_baseline");
      const filtered2 = rows.find((row) => row.strategy === "two_stage_filtered");
      if (!baseline2 || !filtered2) return null;
      return <section>
        <header className="page-header"><div><p className="eyebrow">Post-hoc development study</p><h1>Experiment 2</h1><p>Rolling hedge ratios, boundary-safe entries, structural-stability filtering, expected after-cost return and volatility-aware sizing.</p></div></header>
        <div className="metric-grid">
          <MetricCard label="Validation ROC-AUC" value={experiment2.validation.reversion_roc_auc.toFixed(3)} note="2020–2021 only" />
          <MetricCard label="Dynamic baseline" value={`${(baseline2.total_return*100).toFixed(2)}%`} tone="warn" />
          <MetricCard label="Two-stage filter" value={`${(filtered2.total_return*100).toFixed(2)}%`} tone="warn" />
          <MetricCard label="Filtered drawdown" value={`${(filtered2.maximum_drawdown*100).toFixed(2)}%`} />
          <MetricCard label="Permutation p-value" value={experiment2.significance.one_sided_p_value.toFixed(3)} tone="warn" />
        </div>
        <Plot title="Experiment 2 mean trade return by window" data={[
          { x: experiment2.windows.map(r=>r.window_id), y: experiment2.windows.map(r=>r.all_mean_trade_return*100), type:"bar", name:"All dynamic signals", marker:{color:"#ef7b6c"}},
          { x: experiment2.windows.map(r=>r.window_id), y: experiment2.windows.map(r=>(r.approved_mean_trade_return ?? 0)*100), type:"bar", name:"Two-stage approved", marker:{color:"#48d6b2"}},
        ]} layout={{barmode:"group", yaxis:{title:{text:"Mean return (%)"}, gridcolor:"#1f3045"}}} />
        <div className="callout warn"><strong>Not a new holdout</strong><p>These rules were developed after the original 2022–2026H1 result had been viewed. The filter reduced loss and drawdown but remained unprofitable. A final claim requires completed 2026H2 or later unseen data.</p></div>
      </section>;
    })()}
    {experiment3 && (() => {
      const rows = experiment3.portfolio.filter((row) => row.transaction_cost_bps === 10);
      const baseline3 = rows.find((row) => row.strategy === "daily_signal_baseline");
      const logistic3 = rows.find((row) => row.strategy === "logistic_regression");
      const validation3 = experiment3.validation.find((row) => row.model === "logistic_regression");
      const significance3 = experiment3.significance.find((row) => row.strategy === "logistic_regression");
      const windows3 = experiment3.windows.filter((row) => row.strategy === "logistic_regression");
      const panelRows = experiment3.panel.reduce((sum, row) => sum + row.daily_signal_rows, 0);
      if (!baseline3 || !logistic3 || !validation3 || !significance3) return null;
      return <section>
        <header className="page-header"><div><p className="eyebrow">Daily pair-panel development study</p><h1>Experiment 3</h1><p>90% coverage, 20-day barrier labels, purged chronological learning and non-overlapping portfolio execution.</p></div></header>
        <div className="metric-grid">
          <MetricCard label="Daily ML rows" value={panelRows.toLocaleString()} note={`${validation3.training_rows.toLocaleString()} initial training rows`} />
          <MetricCard label="Baseline return" value={`${(baseline3.total_return*100).toFixed(2)}%`} tone="warn" />
          <MetricCard label="Logistic return" value={`${(logistic3.total_return*100).toFixed(2)}%`} />
          <MetricCard label="Logistic Sharpe" value={logistic3.annualized_sharpe.toFixed(2)} />
          <MetricCard label="Maximum drawdown" value={`${(logistic3.maximum_drawdown*100).toFixed(2)}%`} />
          <MetricCard label="Bootstrap p-value" value={significance3.one_sided_p_value.toFixed(3)} tone="warn" />
        </div>
        <Plot title="Experiment 3 Logistic mean trade return by window" data={[
          { x: windows3.map(r=>r.window_id), y: windows3.map(r=>r.mean_trade_return*100), type:"bar", marker:{color:windows3.map(r=>r.mean_trade_return >= 0 ? "#48d6b2" : "#ef7b6c")} },
        ]} layout={{yaxis:{title:{text:"Mean return (%)"}, gridcolor:"#1f3045"}}} />
        <div className="callout warn"><strong>Promising, not proven</strong><p>The Logistic portfolio earned 0.89% at 10 bps with a 0.16 Sharpe, but becomes negative at 20 bps and the block-bootstrap p-value is 0.388. This is a development result on already-viewed years, not a new profitability claim.</p></div>
      </section>;
    })()}
  </main>;
}
