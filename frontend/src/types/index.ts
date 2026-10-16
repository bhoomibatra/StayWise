export type RiskCategory = 'High Risk' | 'Medium Risk' | 'Low Risk';
export type PredictedChurn = 'Churn' | 'Stay';
export type TaskStatus = 'Open' | 'In Progress' | 'Completed' | 'Cancelled';
export type TaskPriority = 'High' | 'Medium' | 'Low';

export interface OverviewKPIs {
  total_customers: number;
  predicted_churners: number;
  predicted_non_churners: number;
  predicted_churn_rate: number;
  high_risk_count: number;
  high_risk_pct: number;
  medium_risk_count: number;
  medium_risk_pct: number;
  low_risk_count: number;
  low_risk_pct: number;
  actual_churn_count: number;
  actual_churn_rate: number;
  model_accuracy: number;
  model_roc_auc: number;
  model_f1: number;
  model_recall: number;
  active_tasks_count: number;
  completed_tasks_count: number;
}

export interface RiskDistributionItem {
  category: RiskCategory;
  count: number;
  pct: number;
  color: string;
}

export interface PredictionDistributionItem {
  label: string;
  count: number;
  pct: number;
  color: string;
}

export interface ContractBreakdownItem {
  contract: string;
  total: number;
  pred_churn: number;
  pred_churn_rate: number;
  actual_churn: number;
  actual_churn_rate: number;
}

export interface TenureBreakdownItem {
  band: string;
  total: number;
  pred_churn: number;
  pred_churn_rate: number;
  actual_churn: number;
  actual_churn_rate: number;
}

export interface PaymentBreakdownItem {
  payment: string;
  total: number;
  pred_churn: number;
  pred_churn_rate: number;
}

export interface OverviewData {
  kpis: OverviewKPIs;
  risk_distribution: RiskDistributionItem[];
  prediction_distribution: PredictionDistributionItem[];
  contract_breakdown: ContractBreakdownItem[];
  tenure_breakdown: TenureBreakdownItem[];
  payment_breakdown: PaymentBreakdownItem[];
  system_status: {
    model_name: string;
    pipeline_status: string;
    last_scored_at: string;
    data_source: string;
    calibration_status: string;
  };
}

export interface CustomerSummaryItem {
  customer_id: string;
  predicted_churn: PredictedChurn;
  churn_probability: number;
  churn_probability_pct: number;
  risk_category: RiskCategory;
  contract: string;
  tenure: number;
  monthly_charges: number;
  total_charges: number | null;
  payment_method: string;
  internet_service: string;
  historical_churn?: string;
  has_active_retention_task: boolean;
}

export interface PaginationMeta {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CustomerQueryResponse {
  items: CustomerSummaryItem[];
  pagination: PaginationMeta;
}

export interface RiskDriver {
  title: string;
  severity: 'High' | 'Medium' | 'Low';
  detail: string;
}

export interface ProtectiveFactor {
  title: string;
  detail: string;
}

export interface RecommendedAction {
  type: string;
  title: string;
  description: string;
  priority: 'High' | 'Medium' | 'Low';
}

export interface RetentionTask {
  id: number;
  customer_id: string;
  task_type: string;
  priority: TaskPriority;
  status: TaskStatus;
  assigned_to: string;
  notes: string;
  due_date: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface RetentionNote {
  id: number;
  customer_id: string;
  author: string;
  note: string;
  created_at: string;
}

export interface CustomerProfile {
  customer_id: string;
  model_output: {
    predicted_churn: PredictedChurn;
    churn_probability: number;
    churn_probability_pct: number;
    risk_category: RiskCategory;
    historical_actual_churn?: string;
  };
  account_details: {
    tenure: number;
    contract: string;
    paperless_billing: string;
    payment_method: string;
    monthly_charges: number;
    total_charges: number | null;
  };
  demographics: {
    gender: string;
    senior_citizen: string;
    partner: string;
    dependents: string;
  };
  services: {
    phone_service: string;
    multiple_lines: string;
    internet_service: string;
    online_security: string;
    online_backup: string;
    device_protection: string;
    tech_support: string;
    streaming_tv: string;
    streaming_movies: string;
  };
  risk_drivers: RiskDriver[];
  protective_factors: ProtectiveFactor[];
  recommended_actions: RecommendedAction[];
  retention_history: {
    tasks: RetentionTask[];
    notes: RetentionNote[];
  };
}

export interface SegmentStat {
  segment: string;
  total: number;
  historical_churn: number;
  historical_churn_rate: number;
  predicted_churn: number;
  predicted_churn_rate: number;
}

export interface AnalyticsData {
  by_contract: SegmentStat[];
  by_tenure: SegmentStat[];
  by_payment: SegmentStat[];
  by_internet: SegmentStat[];
  by_tech_support: SegmentStat[];
  by_online_security: SegmentStat[];
  by_paperless: SegmentStat[];
  financial_comparison: {
    churners: {
      count: number;
      avg_monthly_charges: number;
      median_monthly_charges: number;
      avg_tenure: number;
      median_tenure: number;
      avg_total_charges: number;
    };
    retainers: {
      count: number;
      avg_monthly_charges: number;
      median_monthly_charges: number;
      avg_tenure: number;
      median_tenure: number;
      avg_total_charges: number;
    };
  };
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface DetailedEvaluation {
  confusion_matrix: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    total: number;
  };
  roc_curve: Array<{ fpr: number; tpr: number }>;
  metrics: ModelMetrics;
}

export interface ModelPerformanceData {
  summary: Record<string, ModelMetrics>;
  best_model: string;
  selection_rationale: string;
  feature_importances: Array<{ feature: string; importance: number }>;
  detailed_evaluations: Record<string, DetailedEvaluation>;
  holdout_test_size: number;
  methodology: {
    validation_scheme: string;
    benchmark_dataset: string;
    leakage_prevention: string;
    calibration_note: string;
  };
}

export interface ThresholdConfig {
  high_threshold: number;
  medium_threshold: number;
}
