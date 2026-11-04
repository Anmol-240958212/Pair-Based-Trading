import { useState } from "react";
import { Dashboard } from "./pages/Dashboard";
import { Methodology } from "./pages/Methodology";
import { PairExplorer } from "./pages/PairExplorer";
import { Results } from "./pages/Results";

const pages = [
  ["dashboard", "Overview"],
  ["pairs", "Pair explorer"],
  ["results", "Research results"],
  ["methodology", "Methodology"],
] as const;

export default function App() {
  const [page, setPage] = useState<(typeof pages)[number][0]>("dashboard");
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">P</div><div><strong>PairLab</strong><span>India research</span></div></div>
      <nav>{pages.map(([key,label]) => <button key={key} className={page===key?"active":""} onClick={()=>setPage(key)}>{label}</button>)}</nav>
      <div className="sidebar-note"><span className="pulse" />Research snapshot<br/><small>Locked through 2026H1</small></div>
    </aside>
    <div className="content">
      {page === "dashboard" && <Dashboard />}
      {page === "pairs" && <PairExplorer />}
      {page === "results" && <Results />}
      {page === "methodology" && <Methodology />}
    </div>
  </div>;
}
