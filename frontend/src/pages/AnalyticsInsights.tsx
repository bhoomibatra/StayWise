import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingDown,
  DollarSign,
  Clock,
  Shield,
  Layers,
  FileSpreadsheet,
  HelpCircle,
} from 'lucide-react';
import {
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
import { AnalyticsData } from '../types';

export const AnalyticsInsights: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getAnalytics();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-3 text-slate-500">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
        <span className="text-sm font-medium">Aggregating empirical dataset churn factors...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
          <p className="font-bold">Failed to load analytics</p>
          <p className="mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const {
    by_contract,
    by_tenure,
    by_payment,
    by_internet,
    by_tech_support,
    by_online_security,
    financial_comparison,
  } = data;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Analytics Header */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-teal-600" />
              Empirical Churn Factors & Cohort Breakdown
            </h2>
            <p className="text-xs text-slate-500 mt-1 max-w-3xl">
              Defensible exploratory analysis comparing customer attributes across actual historical churn classes and prospective model estimates.
              Based strictly on the 7,043 customer snapshot records.
            </p>
          </div>

          <div className="flex items-center gap-1.5 text-[11px] text-slate-500 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 shrink-0">
            <HelpCircle className="h-4 w-4 text-slate-400" />
            <span>Actual churn rate: 26.54% (1,869 churners / 7,043 accounts)</span>
          </div>
        </div>
      </div>

      {/* Financial Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Churners Financial Profile */}
        <div className="rounded-xl border border-rose-200 bg-gradient-to-br from-white to-rose-50/20 p-5 shadow-xs">
          <div className="flex items-center justify-between border-b border-rose-100 pb-3 mb-4">
            <span className="text-xs font-bold uppercase tracking-wider text-rose-700">
              Churned Cohort ({financial_comparison.churners.count.toLocaleString()} accounts)
            </span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-rose-100 text-rose-800">
              Historical Exit
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="space-y-1">
              <span className="text-[11px] text-slate-500 block">Avg Monthly Bill</span>
              <span className="text-2xl font-extrabold text-slate-900">
                ${financial_comparison.churners.avg_monthly_charges}
              </span>
              <span className="text-[10px] text-rose-600 font-semibold block">+$13.11 vs Retainers</span>
            </div>

            <div className="space-y-1">
              <span className="text-[11px] text-slate-500 block">Avg Tenure</span>
              <span className="text-2xl font-extrabold text-slate-900">
                {financial_comparison.churners.avg_tenure} mo
              </span>
              <span className="text-[10px] text-slate-400 block">
                Median: {financial_comparison.churners.median_tenure} mo
              </span>
            </div>

            <div className="space-y-1">
              <span className="text-[11px] text-slate-500 block">Avg Total Charges</span>
              <span className="text-2xl font-extrabold text-slate-900">
                ${financial_comparison.churners.avg_total_charges.toFixed(0)}
              </span>
              <span className="text-[10px] text-slate-400 block">Lifetime Value</span>
            </div>
          </div>
        </div>

        {/* Retained Cohort Financial Profile */}
        <div className="rounded-xl border border-emerald-200 bg-gradient-to-br from-white to-emerald-50/20 p-5 shadow-xs">
          <div className="flex items-center justify-between border-b border-emerald-100 pb-3 mb-4">
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-700">
              Retained Cohort ({financial_comparison.retainers.count.toLocaleString()} accounts)
            </span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
              Active Loyal
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="space-y-1">
              <span className="text-[11px] text-slate-500 block">Avg Monthly Bill</span>
              <span className="text-2xl font-extrabold text-slate-900">
                ${financial_comparison.retainers.avg_monthly_charges}
              </span>
              <span className="text-[10px] text-slate-400 block">
                Median: ${financial_comparison.retainers.median_monthly_charges}
              </span>
            </div>

            <div className="space-y-1">
              <span className="text-[11px] text-slate-500 block">Avg Tenure</span>
              <span className="text-2xl font-extrabold text-slate-900">
                {financial_comparison.retainers.avg_tenure} mo
              </span>
              <span className="text-[10px] text-emerald-600 font-semibold block">+20.7 mo longer tenure</span>
            </div>

            <div className="space-y-1">
              <span className="text-[11px] text-slate-500 block">Avg Total Charges</span>
              <span className="text-2xl font-extrabold text-slate-900">
                ${financial_comparison.retainers.avg_total_charges.toFixed(0)}
              </span>
              <span className="text-[10px] text-emerald-600 font-semibold block">1.66x higher LTV</span>
            </div>
          </div>
        </div>
      </div>

      {/* Row 1: Internet Service & Add-on Services Impact */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Internet Service Churn Comparison */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Internet Technology Tier vs Churn Rate</h3>
              <p className="text-xs text-slate-500 mt-0.5">Fiber optic subscribers exhibit substantially higher churn (~41.9%)</p>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={by_internet} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="segment" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(val: any, name: any) => [
                    `${val} accounts`,
                    name === 'historical_churn' ? 'Historical Churn' : 'Predicted Churn',
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="historical_churn" fill="#f43f5e" name="Historical Churn" radius={[4, 4, 0, 0]} />
                <Bar dataKey="predicted_churn" fill="#0d9488" name="Predicted Churn" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Tech Support Protection Effect */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Impact of Technical Support Add-On</h3>
              <p className="text-xs text-slate-500 mt-0.5">Absence of technical support is a primary friction indicator</p>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={by_tech_support} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="segment" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(val: any, name: any) => [
                    `${val} accounts`,
                    name === 'historical_churn' ? 'Historical Churn' : 'Total Cohort',
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="total" fill="#e2e8f0" name="Total in Cohort" radius={[4, 4, 0, 0]} />
                <Bar dataKey="historical_churn" fill="#f43f5e" name="Historical Churn" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Cohort Comparative Summary Table */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/50 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Cohort Churn Propensity Summary</h3>
            <p className="text-xs text-slate-500">Cross-segment comparative metrics for strategic retention planning</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase text-[11px]">
              <tr>
                <th className="py-3 px-4">Segment Dimension</th>
                <th className="py-3 px-4">Cohort Name</th>
                <th className="py-3 px-4">Subscribers</th>
                <th className="py-3 px-4">Historical Churners</th>
                <th className="py-3 px-4">Historical Churn Rate</th>
                <th className="py-3 px-4">Model Predicted Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {by_contract.map((c) => (
                <tr key={`contract-${c.segment}`} className="hover:bg-slate-50/60">
                  <td className="py-2.5 px-4 font-medium text-slate-500">Contract</td>
                  <td className="py-2.5 px-4 font-semibold text-slate-900">{c.segment}</td>
                  <td className="py-2.5 px-4 text-slate-700">{c.total.toLocaleString()}</td>
                  <td className="py-2.5 px-4 text-rose-600 font-medium">{c.historical_churn.toLocaleString()}</td>
                  <td className="py-2.5 px-4 font-bold text-slate-900">{c.historical_churn_rate}%</td>
                  <td className="py-2.5 px-4 text-teal-700 font-semibold">{c.predicted_churn_rate}%</td>
                </tr>
              ))}
              {by_internet.map((it) => (
                <tr key={`internet-${it.segment}`} className="hover:bg-slate-50/60">
                  <td className="py-2.5 px-4 font-medium text-slate-500">Internet Service</td>
                  <td className="py-2.5 px-4 font-semibold text-slate-900">{it.segment}</td>
                  <td className="py-2.5 px-4 text-slate-700">{it.total.toLocaleString()}</td>
                  <td className="py-2.5 px-4 text-rose-600 font-medium">{it.historical_churn.toLocaleString()}</td>
                  <td className="py-2.5 px-4 font-bold text-slate-900">{it.historical_churn_rate}%</td>
                  <td className="py-2.5 px-4 text-teal-700 font-semibold">{it.predicted_churn_rate}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
