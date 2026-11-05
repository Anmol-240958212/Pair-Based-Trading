import { useEffect, useState } from "react";
import { getJson } from "../api";
import { MetricCard } from "../components/MetricCard";
import { Plot } from "../components/Plot";

type Locked = { window_id: string; baseline_trades: number; ml_approved_trades: number; baseline_mean_trade_return: number; ml_mean_trade_return: number; baseline_success_rate: number; ml_success_rate: number };
type Portfolio = { strategy: string; transaction_cost_bps: number; total_return: number; annualized_sharpe: number; maximum_drawdown: number; accepted_trades: number };
type Significance = { observed_ml_mean_return: number; random_filter_mean_return: number; random_filter_95pct_lower: number; random_filter_95pct_upper: number; one_sided_p_value: number; repetitions: number };
type TradeStatistics = { trades: number; profitable_trades: number; win_rate: number; target_rate: number; take_profit_hits: number; stop_loss_hits: number; horizon_exits: number; mean_trade_return: number; median_trade_return: number; average_winner: number; average_loser: number; profit_factor: number | null; average_holding_days: number; median_holding_days: number };
type ModelStatistics = { sample: string; rows: number; roc_auc: number; brier_score: number; accuracy_at_0_5: number; approved_rows: number; approved_target_rate: number; approved_profit_rate: number };
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
type Experiment4 = {
  interpretation: string;
  validation: Array<{ model: string; selected_threshold: number; roc_auc: number; approved_mean_return: number }>;
  portfolio: Portfolio[];
  significance: Array<{ strategy: string; one_sided_p_value: number }>;
  windows: Array<{ strategy: string; window_id: string; mean_trade_return: number; accepted_trades: number }>;
};
type Experiment5 = {
  interpretation: string;
  validation: Array<{ observation_days: number; approved_mean_return: number; approved_rows: number; selected_on_validation: boolean }>;
  portfolio: Array<Portfolio & { observation_days: number; selected_on_validation: boolean }>;
  significance: Array<{ observation_days: number; one_sided_p_value: number; selected_on_validation: boolean }>;
  windows: Array<{ observation_days: number; window_id: string; mean_trade_return: number; accepted_trades: number }>;
  three_day_trade_statistics: TradeStatistics;
  three_day_model_statistics: ModelStatistics[];
};
type Experiment6 = {
  interpretation: string;
  announcement_rows: number;
  covered_companies: number;
  comparison: Array<{ sample: string; strategy: string; approved_rows: number; approved_profit_rate: number }>;
  portfolio: Array<Portfolio & { win_rate: number }>;
  significance: Array<{ strategy: string; one_sided_p_value: number }>;
  without_news_trade_statistics: TradeStatistics;
  with_news_trade_statistics: TradeStatistics;
};
type Experiment7 = {
  interpretation: string;
  validation: Array<{ configuration: string; approved_mean_return: number; selected_on_validation: boolean }>;
  portfolio: Array<Portfolio & { configuration: string; win_rate: number; selected_on_validation: boolean }>;
  significance: Array<{ configuration: string; one_sided_p_value: number }>;
  trade_statistics: Array<TradeStatistics & { configuration: string }>;
};

function TradeStatsTable({ rows }: { rows: Array<{ label: string; stats: TradeStatistics }> }) {
  return <div className="table-card"><table>
    <thead><tr><th>Strategy</th><th>Trades</th><th>Win rate</th><th>Target hits</th><th>Avg. trade</th><th>Avg. winner</th><th>Avg. loser</th><th>Profit factor</th><th>Avg. holding</th></tr></thead>
    <tbody>{rows.map(({label,stats})=><tr key={label}>
      <td>{label}</td><td>{stats.trades}</td><td>{(stats.win_rate*100).toFixed(1)}%</td><td>{stats.take_profit_hits}</td>
      <td className={stats.mean_trade_return >= 0 ? "positive" : "negative"}>{(stats.mean_trade_return*100).toFixed(3)}%</td>
      <td className="positive">{(stats.average_winner*100).toFixed(3)}%</td><td className="negative">{(stats.average_loser*100).toFixed(3)}%</td>
      <td>{stats.profit_factor?.toFixed(2) ?? "—"}</td><td>{stats.average_holding_days.toFixed(2)} days</td>
    </tr>)}</tbody>
  </table></div>;
}

export function Results() {
  const [locked, setLocked] = useState<Locked[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio[]>([]);
  const [significance, setSignificance] = useState<Significance>();
  const [experiment2, setExperiment2] = useState<Experiment2>();
  const [experiment3, setExperiment3] = useState<Experiment3>();
  const [experiment4, setExperiment4] = useState<Experiment4>();
  const [experiment5, setExperiment5] = useState<Experiment5>();
  const [experiment6, setExperiment6] = useState<Experiment6>();
  const [experiment7, setExperiment7] = useState<Experiment7>();
  useEffect(() => { Promise.all([getJson<Locked[]>("/api/results/locked-test"), getJson<Portfolio[]>("/api/results/portfolio"), getJson<Significance>("/api/results/significance"), getJson<Experiment2>("/api/results/experiment2"), getJson<Experiment3>("/api/results/experiment3"), getJson<Experiment4>("/api/results/experiment4"), getJson<Experiment5>("/api/results/experiment5"), getJson<Experiment6>("/api/results/experiment6"), getJson<Experiment7>("/api/results/experiment7")]).then(([a,b,c,d,e,f,g,h,i]) => { setLocked(a); setPortfolio(b); setSignificance(c); setExperiment2(d); setExperiment3(e); setExperiment4(f); setExperiment5(g); setExperiment6(h); setExperiment7(i); }); }, []);
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
    {experiment4 && (() => {
      const rows = experiment4.portfolio.filter((row) => row.transaction_cost_bps === 10);
      const logistic4 = rows.find((row) => row.strategy === "logistic_regression");
      const markov4 = rows.find((row) => row.strategy === "markov_regime_logistic");
      const lda4 = rows.find((row) => row.strategy === "shrinkage_lda");
      const bayes4 = rows.find((row) => row.strategy === "gaussian_naive_bayes");
      const significance4 = experiment4.significance.find((row) => row.strategy === "markov_regime_logistic");
      const windows4 = experiment4.windows.filter((row) => row.strategy === "markov_regime_logistic");
      if (!logistic4 || !markov4 || !lda4 || !bayes4 || !significance4) return null;
      return <section>
        <header className="page-header"><div><p className="eyebrow">Mathematical-model development study</p><h1>Experiment 4</h1><p>Shrinkage discriminant analysis, Gaussian Naive Bayes and a two-state Gaussian Markov regime filter, evaluated on the same daily panel and execution rules.</p></div></header>
        <div className="metric-grid">
          <MetricCard label="Ordinary Logistic" value={`${(logistic4.total_return*100).toFixed(2)}%`} />
          <MetricCard label="Markov + Logistic" value={`${(markov4.total_return*100).toFixed(2)}%`} />
          <MetricCard label="Markov Sharpe" value={markov4.annualized_sharpe.toFixed(2)} />
          <MetricCard label="Markov drawdown" value={`${(markov4.maximum_drawdown*100).toFixed(2)}%`} />
          <MetricCard label="Bootstrap p-value" value={significance4.one_sided_p_value.toFixed(3)} tone="warn" />
        </div>
        <Plot title="Experiment 4 return at 10 bps" data={[
          { x:["Logistic", "Shrinkage LDA", "Gaussian Bayes", "Markov + Logistic"], y:[logistic4.total_return, lda4.total_return, bayes4.total_return, markov4.total_return].map(value=>value*100), type:"bar", marker:{color:[logistic4,lda4,bayes4,markov4].map(row=>row.total_return >= 0 ? "#48d6b2" : "#ef7b6c")} },
        ]} layout={{yaxis:{title:{text:"Total return (%)"}, gridcolor:"#1f3045"}}} />
        <Plot title="Markov-regime mean trade return by window" data={[
          { x:windows4.map(row=>row.window_id), y:windows4.map(row=>row.mean_trade_return*100), type:"bar", marker:{color:windows4.map(row=>row.mean_trade_return >= 0 ? "#48d6b2" : "#ef7b6c")} },
        ]} layout={{yaxis:{title:{text:"Mean return (%)"}, gridcolor:"#1f3045"}}} />
        <div className="callout warn"><strong>Best development model, still unproven</strong><p>The Markov filter improves the 10 bps result from 0.89% to 1.31% and the Sharpe from 0.16 to 0.22. It remains negative at 20 bps and its bootstrap p-value is 0.328, so fresh data is still required.</p></div>
      </section>;
    })()}
    {experiment5 && (() => {
      const tenBps = experiment5.portfolio.filter((row) => row.transaction_cost_bps === 10);
      const immediate = tenBps.find((row) => row.observation_days === 0);
      const oneDay = tenBps.find((row) => row.observation_days === 1);
      const threeDay = tenBps.find((row) => row.observation_days === 3);
      const threeDaySignificance = experiment5.significance.find((row) => row.observation_days === 3);
      const threeDayWindows = experiment5.windows.filter((row) => row.observation_days === 3);
      if (!immediate || !oneDay || !threeDay || !threeDaySignificance) return null;
      return <section>
        <header className="page-header"><div><p className="eyebrow">Delayed-confirmation development study</p><h1>Experiment 5</h1><p>Tests whether observing common-factor movement, abnormal company-specific shocks and spread direction for up to three trading days removes false-positive reversions.</p></div></header>
        <div className="metric-grid">
          <MetricCard label="Immediate entry" value={`${(immediate.total_return*100).toFixed(2)}%`} note="Selected by 2020–2021 validation" />
          <MetricCard label="Three-day diagnostic" value={`${(threeDay.total_return*100).toFixed(2)}%`} />
          <MetricCard label="Three-day Sharpe" value={threeDay.annualized_sharpe.toFixed(2)} />
          <MetricCard label="Three-day drawdown" value={`${(threeDay.maximum_drawdown*100).toFixed(2)}%`} />
          <MetricCard label="Bootstrap p-value" value={threeDaySignificance.one_sided_p_value.toFixed(3)} />
        </div>
        <div className="section-heading"><div><p className="eyebrow">Primary candidate</p><h2>Three-day trade statistics</h2></div><p>Net trade outcomes use the standard 10 bps assumption; unused portfolio slots remain cash.</p></div>
        <TradeStatsTable rows={[{label:"Three-day confirmation",stats:experiment5.three_day_trade_statistics}]} />
        <div className="metric-grid compact">
          <MetricCard label="Approved signals" value={experiment5.three_day_model_statistics[1].approved_rows.toLocaleString()} note={`${experiment5.three_day_model_statistics[1].rows.toLocaleString()} candidate rows`} />
          <MetricCard label="Approved profit rate" value={`${(experiment5.three_day_model_statistics[1].approved_profit_rate*100).toFixed(1)}%`} />
          <MetricCard label="Profit factor" value={experiment5.three_day_trade_statistics.profit_factor?.toFixed(2) ?? "—"} />
          <MetricCard label="Average winner" value={`${(experiment5.three_day_trade_statistics.average_winner*100).toFixed(2)}%`} />
          <MetricCard label="Average loser" value={`${(experiment5.three_day_trade_statistics.average_loser*100).toFixed(2)}%`} tone="warn" />
          <MetricCard label="Average holding" value={`${experiment5.three_day_trade_statistics.average_holding_days.toFixed(2)} days`} />
        </div>
        <Plot title="Observation-period comparison at 10 bps" data={[
          { x:["Immediate", "1 day", "3 days"], y:[immediate.total_return, oneDay.total_return, threeDay.total_return].map(value=>value*100), type:"bar", marker:{color:[immediate,oneDay,threeDay].map(row=>row.total_return >= 0 ? "#48d6b2" : "#ef7b6c")} },
        ]} layout={{yaxis:{title:{text:"Total return (%)"}, gridcolor:"#1f3045"}}} />
        <Plot title="Three-day transaction-cost sensitivity" data={[
          { x:experiment5.portfolio.filter(row=>row.observation_days===3).map(row=>row.transaction_cost_bps), y:experiment5.portfolio.filter(row=>row.observation_days===3).map(row=>row.total_return*100), type:"scatter", mode:"lines+markers", name:"Three-day confirmation", line:{color:"#48d6b2"} },
        ]} layout={{xaxis:{title:{text:"Cost per position change (bps)"},gridcolor:"#1f3045"},yaxis:{title:{text:"Total return (%)"},gridcolor:"#1f3045"}}} />
        <div className="section-heading"><div><p className="eyebrow">Classifier diagnostics</p><h2>Prediction quality</h2></div><p>Overall accuracy remains near random; the economic gates and portfolio constraints create the higher-quality subset.</p></div>
        <div className="table-card"><table><thead><tr><th>Sample</th><th>Rows</th><th>ROC-AUC</th><th>Accuracy</th><th>Approved</th><th>Approved target rate</th><th>Approved profit rate</th></tr></thead><tbody>
          {experiment5.three_day_model_statistics.map(row=><tr key={row.sample}><td>{row.sample}</td><td>{row.rows.toLocaleString()}</td><td>{row.roc_auc.toFixed(3)}</td><td>{(row.accuracy_at_0_5*100).toFixed(1)}%</td><td>{row.approved_rows}</td><td>{(row.approved_target_rate*100).toFixed(1)}%</td><td>{(row.approved_profit_rate*100).toFixed(1)}%</td></tr>)}
        </tbody></table></div>
        <Plot title="Three-day confirmation mean trade return by window" data={[
          { x:threeDayWindows.map(row=>row.window_id), y:threeDayWindows.map(row=>row.mean_trade_return*100), type:"bar", marker:{color:threeDayWindows.map(row=>row.mean_trade_return >= 0 ? "#48d6b2" : "#ef7b6c")} },
        ]} layout={{yaxis:{title:{text:"Mean return (%)"}, gridcolor:"#1f3045"}}} />
        <div className="callout warn"><strong>Promising post-hoc candidate</strong><p>The three-day rule returned 2.48% at 10 bps, remained positive at 20 bps and produced p = 0.033. But 2020–2021 validation selected immediate entry. Choosing three days now would use the later results, so it must be frozen and tested on new data before adoption.</p></div>
      </section>;
    })()}
    {experiment6 && (() => {
      const rows = experiment6.portfolio.filter((row) => row.transaction_cost_bps === 10);
      const withoutNews = rows.find((row) => row.strategy === "approved_without_news");
      const withNews = rows.find((row) => row.strategy === "approved_with_news_veto");
      if (!withoutNews || !withNews) return null;
      return <section>
        <header className="page-header"><div><p className="eyebrow">Point-in-time event-risk study</p><h1>Experiment 6</h1><p>{experiment6.announcement_rows.toLocaleString()} timestamped NSE announcements across {experiment6.covered_companies} companies, restricted to the trigger-to-confirmation window.</p></div></header>
        <div className="metric-grid">
          <MetricCard label="Without news veto" value={`${(withoutNews.total_return*100).toFixed(2)}%`} />
          <MetricCard label="With news veto" value={`${(withNews.total_return*100).toFixed(2)}%`} />
          <MetricCard label="Veto win rate" value={`${(withNews.win_rate*100).toFixed(1)}%`} />
          <MetricCard label="Veto drawdown" value={`${(withNews.maximum_drawdown*100).toFixed(2)}%`} />
        </div>
        <TradeStatsTable rows={[
          {label:"Three-day without news veto",stats:experiment6.without_news_trade_statistics},
          {label:"Three-day with news veto",stats:experiment6.with_news_trade_statistics},
        ]} />
        <Plot title="News-veto transaction-cost sensitivity" data={[
          { x:experiment6.portfolio.filter(row=>row.strategy==="approved_without_news").map(row=>row.transaction_cost_bps), y:experiment6.portfolio.filter(row=>row.strategy==="approved_without_news").map(row=>row.total_return*100), type:"scatter", mode:"lines+markers", name:"Without news veto", line:{color:"#71b7ff"} },
          { x:experiment6.portfolio.filter(row=>row.strategy==="approved_with_news_veto").map(row=>row.transaction_cost_bps), y:experiment6.portfolio.filter(row=>row.strategy==="approved_with_news_veto").map(row=>row.total_return*100), type:"scatter", mode:"lines+markers", name:"With news veto", line:{color:"#48d6b2"} },
        ]} layout={{xaxis:{title:{text:"Cost per position change (bps)"},gridcolor:"#1f3045"},yaxis:{title:{text:"Total return (%)"},gridcolor:"#1f3045"}}} />
        <div className="callout warn"><strong>Useful risk filter, not validation-selected</strong><p>The veto improved win rate and drawdown but slightly reduced total return. It also worsened the 2020–2021 validation mean return, so it should not replace the price-based rule without a fresh test.</p></div>
      </section>;
    })()}
    {experiment7 && (() => {
      const rows = experiment7.portfolio.filter((row) => row.transaction_cost_bps === 10);
      const labels: Record<string,string> = {target_1_stop_2:"+1% / −2%", symmetric_0_5:"±0.5%", symmetric_1_0:"±1%", target_1_5_stop_1:"+1.5% / −1%"};
      if (!rows.length) return null;
      return <section>
        <header className="page-header"><div><p className="eyebrow">Payoff-barrier development study</p><h1>Experiment 7</h1><p>Every target/stop rule rebuilds exits and labels before chronological retraining.</p></div></header>
        <Plot title="Payoff rules at 10 bps" data={[
          { x:rows.map(row=>labels[row.configuration] ?? row.configuration), y:rows.map(row=>row.total_return*100), type:"bar", marker:{color:rows.map(row=>row.selected_on_validation ? "#48d6b2" : "#8294aa")} },
        ]} layout={{yaxis:{title:{text:"Total return (%)"}, gridcolor:"#1f3045"}}} />
        <TradeStatsTable rows={experiment7.trade_statistics.map(stats=>({label:labels[stats.configuration] ?? stats.configuration,stats}))} />
        <div className="table-card"><table><thead><tr><th>Target / stop</th><th>Return at 5 bps</th><th>Return at 10 bps</th><th>Return at 20 bps</th><th>10-bps Sharpe</th><th>10-bps drawdown</th><th>Bootstrap p</th></tr></thead><tbody>
          {experiment7.trade_statistics.map(stats=>{
            const costRows=experiment7.portfolio.filter(row=>row.configuration===stats.configuration);
            const row10=costRows.find(row=>row.transaction_cost_bps===10);
            const p=experiment7.significance.find(row=>row.configuration===stats.configuration);
            return <tr key={stats.configuration}><td>{labels[stats.configuration] ?? stats.configuration}</td>{[5,10,20].map(cost=><td key={cost} className={(costRows.find(row=>row.transaction_cost_bps===cost)?.total_return ?? 0)>=0?"positive":"negative"}>{((costRows.find(row=>row.transaction_cost_bps===cost)?.total_return ?? 0)*100).toFixed(2)}%</td>)}<td>{row10?.annualized_sharpe.toFixed(3)}</td><td>{((row10?.maximum_drawdown ?? 0)*100).toFixed(2)}%</td><td>{p?.one_sided_p_value.toFixed(3)}</td></tr>;
          })}
        </tbody></table></div>
        <div className="callout"><strong>Existing asymmetric barrier retained</strong><p>The +1%/−2% rule remained best in validation and later diagnostics. Symmetric stops were frequently hit before the spread completed its reversion, reducing trade count and total return.</p></div>
      </section>;
    })()}
  </main>;
}
