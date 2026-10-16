import React, { useEffect, useState } from 'react';
import {
  Users,
  AlertTriangle,
  UserCheck,
  TrendingUp,
  Percent,
  Cpu,
  ArrowUpRight,
  ShieldCheck,
  ClipboardCheck,
  Info,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../api/client';
import { OverviewData } from '../types';

interface ExecutiveOverviewProps {
  onNavigateToTab: (tab: any) => void;
  onOpenCreateTask: (cid?: string) => void;
}

export const ExecutiveOverview: React.FC<ExecutiveOverviewProps> = ({
  onNavigateToTab,
  onOpenCreateTask,
}) => {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadOverview();
  }, []);

  const loadOverview = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getOverview();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load executive overview data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-3 text-slate-500">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
        <span className="text-sm font-medium">Computing real-time dataset aggregations...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
          <p className="font-bold">Failed to load overview</p>
          <p className="mt-1">{error}</p>
          <button
            onClick={loadOverview}
            className="mt-4 rounded-lg bg-rose-600 px-4 py-2 text-xs font-semibold text-white hover:bg-rose-700"
          >
            Retry Loading
          </button>
        </div>
      </div>
    );
  }

  const { kpis, risk_distribution, prediction_distribution, contract_breakdown, tenure_breakdown, payment_breakdown, system_status } = data;

  const RISK_COLORS: Record<string, string> = {
    'High Risk': '#ef4444',
    'Medium Risk': '#f59e0b',
    'Low Risk': '#10b981',
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Top Banner Notice */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            <h2 className="text-sm font-bold text-slate-900 tracking-tight">
              Enterprise Churn Intelligence Engine
            </h2>
            <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
              {system_status.model_name}
            </span>
          </div>
          <p className="text-xs text-slate-500">
            Automated intelligence scored across all <strong className="text-slate-700">7,043 subscriber accounts</strong>.
            Distinguishes prospective model predictions from historical snapshot labels.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onNavigateToTab('at-risk')}
            className="flex items-center gap-1.5 rounded-lg border border-rose-300 bg-rose-50 px-3.5 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-100 transition-colors"
          >
            <AlertTriangle className="h-3.5 w-3.5" />
            Review At-Risk Accounts ({kpis.high_risk_count.toLocaleString()})
          </button>
          <button
            onClick={() => onNavigateToTab('customers')}
            className="flex items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white hover:bg-slate-800 transition-colors"
          >
            <span>Customer Directory</span>
            <ArrowUpRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Customers Analyzed */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Total Subscribers
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-600">
              <Users className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-3xl font-extrabold tracking-tight text-slate-900">
              {kpis.total_customers.toLocaleString()}
            </span>
            <p className="text-xs text-slate-500 mt-1">100% evaluated via Scikit-learn</p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Predicted Stay:</span>
            <span className="font-semibold text-slate-700">
              {kpis.predicted_non_churners.toLocaleString()} ({100 - kpis.predicted_churn_rate}%)
            </span>
          </div>
        </div>

        {/* High-Risk Accounts */}
        <div className="rounded-xl border border-rose-200 bg-gradient-to-br from-white to-rose-50/30 p-5 shadow-sm hover:border-rose-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-rose-700">
              High Risk Subscribers
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-100 text-rose-600">
              <AlertTriangle className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight text-rose-600">
                {kpis.high_risk_count.toLocaleString()}
              </span>
              <span className="text-xs font-bold text-rose-600/80">({kpis.high_risk_pct}%)</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">Probability threshold ≥ 60%</p>
          </div>
          <div className="mt-4 pt-3 border-t border-rose-100/60 flex items-center justify-between text-xs text-slate-500">
            <span>Medium Risk Tier:</span>
            <span className="font-semibold text-amber-600">
              {kpis.medium_risk_count.toLocaleString()} ({kpis.medium_risk_pct}%)
            </span>
          </div>
        </div>

        {/* Churn Rates Comparison */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Churn Rates
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-50 text-teal-600">
              <Percent className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3 space-y-1">
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-slate-500">Predicted (Model):</span>
              <span className="text-2xl font-extrabold text-slate-900">{kpis.predicted_churn_rate}%</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-slate-500">Historical Actual:</span>
              <span className="text-sm font-bold text-slate-600">{kpis.actual_churn_rate}%</span>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Predicted Churners:</span>
            <span className="font-semibold text-slate-700">{kpis.predicted_churners.toLocaleString()}</span>
          </div>
        </div>

        {/* Model Test Benchmark */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Model Holdout Benchmark
            </span>
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
              <Cpu className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight text-slate-900">
                {kpis.model_roc_auc}
              </span>
              <span className="text-xs font-semibold text-slate-500">ROC-AUC</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Accuracy: <strong className="text-slate-700">{kpis.model_accuracy}%</strong> | F1: <strong className="text-slate-700">{kpis.model_f1}</strong>
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Recall on Churners:</span>
            <span className="font-semibold text-teal-700">{kpis.model_recall}%</span>
          </div>
        </div>
      </div>

      {/* Visualizations Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Distribution Donut */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Customer Risk Distribution</h3>
              <p className="text-xs text-slate-500 mt-0.5">Categorized by operational probability thresholds</p>
            </div>
          </div>

          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={risk_distribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={4}
                  dataKey="count"
                  nameKey="category"
                >
                  {risk_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={RISK_COLORS[entry.category] || entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: any, name: any) => [
                    `${Number(value).toLocaleString()} accounts`,
                    name,
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-3 gap-2 text-center">
            {risk_distribution.map((item) => (
              <div key={item.category} className="space-y-0.5">
                <span className="text-[11px] font-semibold text-slate-500 block truncate">{item.category}</span>
                <span className="text-base font-bold text-slate-900 block">{item.count.toLocaleString()}</span>
                <span className="text-[10px] text-slate-400 block">{item.pct}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Churn by Contract Type */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Churn Rate by Contract Structure</h3>
              <p className="text-xs text-slate-500 mt-0.5">Comparing Month-to-month flexibility vs multi-year commitment</p>
            </div>
            <span className="text-xs text-slate-400 font-medium">Dataset Breakdown</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={contract_breakdown} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="contract" tick={{ fontSize: 12, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(value: any, name: any) => [
                    `${Number(value).toLocaleString()} accounts`,
                    name === 'pred_churn' ? 'Predicted Churn' : 'Total Accounts',
                  ]}
                />
                <Legend
                  wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
                  formatter={(val) => (val === 'total' ? 'Total Subscribers' : 'Predicted Churners')}
                />
                <Bar dataKey="total" fill="#e2e8f0" radius={[4, 4, 0, 0]} name="total" />
                <Bar dataKey="pred_churn" fill="#f43f5e" radius={[4, 4, 0, 0]} name="pred_churn" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>
              Month-to-month churn rate: <strong className="text-rose-600 font-bold">{contract_breakdown.find(c => c.contract === 'Month-to-month')?.pred_churn_rate || 42}%</strong>
            </span>
            <span>
              Two-year churn rate: <strong className="text-emerald-600 font-bold">{contract_breakdown.find(c => c.contract === 'Two year')?.pred_churn_rate || 2.8}%</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Visualizations Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tenure Hazard Window Breakdown */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Tenure Hazard Curve (Lifecycle Attrition)</h3>
              <p className="text-xs text-slate-500 mt-0.5">Demonstrating sharp customer attrition in early lifecycle (0-12m)</p>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tenure_breakdown} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="band" tick={{ fontSize: 12, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(val: any, name: any) => [
                    `${val} accounts`,
                    name === 'pred_churn' ? 'Predicted Churners' : 'Total Cohort',
                  ]}
                />
                <Bar dataKey="total" fill="#cbd5e1" radius={[4, 4, 0, 0]} name="total" />
                <Bar dataKey="pred_churn" fill="#0d9488" radius={[4, 4, 0, 0]} name="pred_churn" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Churn by Payment Method */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Churn Propensity by Payment Method</h3>
              <p className="text-xs text-slate-500 mt-0.5">Manual check friction vs automated debit / credit stickiness</p>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={payment_breakdown}
                layout="vertical"
                margin={{ top: 10, right: 20, left: 20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis dataKey="payment" type="category" width={110} tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(val: any) => [`${val} accounts`, 'Predicted Churners']} />
                <Bar dataKey="pred_churn" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
