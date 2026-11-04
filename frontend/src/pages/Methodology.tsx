import { useEffect, useState } from "react";
import { getJson } from "../api";

type Config = Record<string, string | number>;

export function Methodology() {
  const [config, setConfig] = useState<Config>({});
  useEffect(() => { getJson<Config>("/api/config").then(setConfig); }, []);
  return <main>
    <header className="page-header"><div><p className="eyebrow">Reproducible by design</p><h1>Methodology</h1><p>Every pair relationship is estimated before its later trading period, and every ML label becomes available only after its trade has finished.</p></div></header>
    <div className="method-grid">
      <article><span>01</span><h3>Point-in-time universe</h3><p>Each annual workbook contains that year’s F&amp;O companies. Formation eligibility requires 95% coverage and at least 500 observations.</p></article>
      <article><span>02</span><h3>Economic pre-selection</h3><p>Only companies in the same Capitaline industry are paired. Industry similarity narrows the search; it does not replace statistical testing.</p></article>
      <article><span>03</span><h3>Robust cointegration</h3><p>The frozen rule requires Engle–Granger significance in both regression directions and Johansen confirmation. FDR is reported separately because it leaves too little training data.</p></article>
      <article><span>04</span><h3>Strict chronology</h3><p>Three-year formation estimates are frozen before each six-month trading window. Signals execute the following day.</p></article>
      <article><span>05</span><h3>ML as a filter</h3><p>Logistic Regression estimates after-cost profitability. It does not decide historical cointegration and it cannot use unfinished trade outcomes.</p></article>
      <article><span>06</span><h3>Experiment 2</h3><p>A separately labelled development study combines rolling hedge ratios, entry-window protection, structural-stability features, a reversion classifier, a net-return regressor and volatility-aware sizing.</p></article>
      <article><span>07</span><h3>Experiment 3</h3><p>Coverage is relaxed to 90%, and every eligible pair-day above |z|=1 becomes a sample. A 20-day take-profit-before-stop target is learned with Logistic Regression and Gradient Boosting using label-available chronological splits.</p></article>
      <article><span>06</span><h3>Locked test</h3><p>Model choice used 2018–2021 only. The 2022–2026H1 period was opened after the specification was recorded.</p></article>
    </div>
    <section className="config-card"><p className="eyebrow">Frozen configuration</p>{Object.entries(config).map(([key,value]) => <div key={key}><span>{key.replaceAll("_"," ")}</span><strong>{String(value)}</strong></div>)}</section>
    <div className="callout"><strong>Study scope</strong><p>Adjusted underlying equity prices for companies in the contemporaneous Indian F&amp;O universe. The study does not reproduce individual futures contracts, rolls, margins or lot sizes.</p></div>
  </main>;
}
