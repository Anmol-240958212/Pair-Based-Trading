export type Numeric = number | null;

export interface WindowRow {
  window_id: string;
  formation_start: string;
  formation_end: string;
  trading_start: string;
  trading_end: string;
  eligible_companies: number;
  same_industry_pairs_tested: number;
  statistically_selected_pairs: number;
  labelled_events: number;
}

export interface IndustryRow {
  industry: string;
  candidate_pairs: number;
  companies: number;
}

export interface PairRow {
  window_id: string;
  industry: string;
  left_code: number;
  left_company: string;
  right_code: number;
  right_company: string;
  cointegration_p_value: number;
  cointegration_p_value_reverse: number;
  cointegration_fdr_q_value: number;
  passes_fdr: boolean;
  johansen_passes_5pct: boolean;
  return_correlation: number;
  hedge_ratio: number;
  spread_half_life_days: Numeric;
  latest_z_score: number;
}

export interface SeriesPoint {
  date: string;
  left_close?: number;
  right_close?: number;
  z_score?: number;
  gross_equity?: number;
  net_equity?: number;
}

export interface PairAnalysis {
  window: WindowRow;
  pair: PairRow;
  formation_metrics: Record<string, Numeric | boolean>;
  formation_series: SeriesPoint[];
  backtest_metrics: Record<string, Numeric>;
  trading_series: SeriesPoint[];
  trades: Array<Record<string, string | number>>;
}
