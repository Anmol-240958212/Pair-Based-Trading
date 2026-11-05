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
  model_available: boolean;
  model_candidate_signals: number;
  model_approved_signals: number;
  model_executed_trades: number;
}

export interface SeriesPoint {
  date: string;
  left_close?: number;
  right_close?: number;
  z_score?: number;
  gross_equity?: number;
  net_equity?: number;
}

export interface FinalModelConfig {
  model: string;
  observation_days: number;
  probability_threshold: number;
  take_profit_return: number;
  stop_loss_return: number;
  maximum_holding_days: number;
  transaction_cost_bps_per_side: number;
  mandatory_news_veto: boolean;
}

export interface FinalModelSummary {
  candidate_signals: number;
  confirmation_passed: number;
  probability_passed: number;
  approved_signals: number;
  executed_trades: number;
  profitable_trades: number;
  win_rate: Numeric;
  compounded_pair_return: number;
  average_trade_return: Numeric;
  average_holding_days: Numeric;
}

export interface FinalModelSignal {
  event_id: string;
  trigger_date: string;
  signal_date: string;
  entry_date: string;
  exit_date: string;
  direction: number;
  initial_z_score: number;
  entry_z_score: number;
  absolute_z_convergence: number;
  same_direction_fraction: number;
  factor_alignment_fraction: number;
  lead_lag_agreement_fraction: number;
  maximum_standardized_idiosyncratic_shock: number;
  markov_high_volatility_probability: number;
  markov_favorable_probability: number;
  predicted_probability: number;
  passes_confirmation: boolean;
  probability_passed: boolean;
  approved: boolean;
  executed: boolean;
  decision: string;
  barrier_outcome: string;
  trade_net_return: number;
  holding_days: number;
}

export interface FinalModelTrade {
  event_id: string;
  trigger_date: string;
  signal_date: string;
  entry_date: string;
  exit_date: string;
  direction: number;
  initial_z_score: number;
  entry_z_score: number;
  predicted_probability: number;
  barrier_outcome: string;
  trade_net_return: number;
  profitable: boolean;
  holding_days: number;
}

export interface FinalModelAnalysis {
  available: boolean;
  reason: string | null;
  config: FinalModelConfig;
  summary: FinalModelSummary;
  signals: FinalModelSignal[];
  trades: FinalModelTrade[];
  equity_series: Array<{ date: string; net_equity: number; event_id: string }>;
}

export interface PairAnalysis {
  window: WindowRow;
  pair: PairRow;
  formation_metrics: Record<string, Numeric | boolean>;
  formation_series: SeriesPoint[];
  backtest_metrics: Record<string, Numeric>;
  trading_series: SeriesPoint[];
  trades: Array<Record<string, string | number>>;
  final_model: FinalModelAnalysis;
}
