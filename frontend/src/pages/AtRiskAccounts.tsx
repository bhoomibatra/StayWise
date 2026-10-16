import React, { useState, useEffect } from 'react';
import {
  AlertTriangle,
  Flame,
  ArrowRight,
  PlusCircle,
  Eye,
  Filter,
  CheckCircle,
  HelpCircle,
  ShieldAlert,
} from 'lucide-react';
import { api } from '../api/client';
import { CustomerSummaryItem, PaginationMeta } from '../types';
import { RiskBadge, ChurnBadge } from '../components/Badges';

interface AtRiskAccountsProps {
  onSelectCustomer: (customerId: string) => void;
  onOpenCreateTask: (customerId: string, taskType?: string) => void;
}

export const AtRiskAccounts: React.FC<AtRiskAccountsProps> = ({
  onSelectCustomer,
  onOpenCreateTask,
}) => {
  const [accounts, setAccounts] = useState<CustomerSummaryItem[]>([]);
  const [pagination, setPagination] = useState<PaginationMeta>({
    total: 0,
    page: 1,
    page_size: 25,
    total_pages: 1,
  });
  const [minRisk, setMinRisk] = useState<'High Risk' | 'Medium Risk'>('High Risk');
  const [contract, setContract] = useState('All');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAtRisk = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getAtRiskAccounts({
        min_risk: minRisk,
        contract,
        page: pagination.page,
        page_size: pagination.page_size,
      });
      setAccounts(res.items);
      setPagination(res.pagination);
    } catch (err: any) {
      setError(err.message || 'Failed to load at-risk queue');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAtRisk();
  }, [minRisk, contract, pagination.page]);

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Priority Triage Header */}
      <div className="rounded-xl border border-rose-200 bg-gradient-to-r from-rose-50 via-white to-amber-50/20 p-6 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-600 text-white">
              <Flame className="h-4 w-4" />
            </div>
            <h2 className="text-base font-bold text-slate-900 tracking-tight">
              Prioritized Account Triage Queue
            </h2>
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-100 text-rose-800">
              Immediate Intervention
            </span>
          </div>
          <p className="text-xs text-slate-600 max-w-3xl">
            Ranked queue of subscribers exhibiting highest estimated churn propensity.
            Enables retention specialists to proactively engage accounts prior to cancellation notice.
          </p>
        </div>

        {/* Methodology note */}
        <div className="flex items-center gap-1.5 text-[11px] text-slate-500 bg-white/80 border border-slate-200 rounded-lg px-3 py-2 shrink-0">
          <HelpCircle className="h-4 w-4 text-slate-400" />
          <span>Probabilities represent statistical model risk estimates, not certain churn.</span>
        </div>
      </div>

      {/* Filter and Queue Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold text-slate-600 flex items-center gap-1">
            <Filter className="h-3.5 w-3.5" />
            Queue Scope:
          </span>

          {/* Risk Scope toggle */}
          <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-0.5">
            <button
              onClick={() => {
                setMinRisk('High Risk');
                setPagination((p) => ({ ...p, page: 1 }));
              }}
              className={`rounded-md px-3 py-1 font-semibold transition-all ${
                minRisk === 'High Risk'
                  ? 'bg-rose-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🚨 High Risk (≥ 60%)
            </button>
            <button
              onClick={() => {
                setMinRisk('Medium Risk');
                setPagination((p) => ({ ...p, page: 1 }));
              }}
              className={`rounded-md px-3 py-1 font-semibold transition-all ${
                minRisk === 'Medium Risk'
                  ? 'bg-slate-800 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              ⚠️ Medium & High (≥ 35%)
            </button>
          </div>

          {/* Contract Filter */}
          <select
            value={contract}
            onChange={(e) => {
              setContract(e.target.value);
              setPagination((p) => ({ ...p, page: 1 }));
            }}
            className="rounded-lg border border-slate-300 px-3 py-1 text-xs text-slate-700 bg-white focus:outline-none focus:border-teal-500"
          >
            <option value="All">All Contract Types</option>
            <option value="Month-to-month">Month-to-month Only</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>

        <div className="text-slate-500">
          Showing <strong className="text-slate-900">{pagination.total.toLocaleString()}</strong> at-risk accounts in queue
        </div>
      </div>

      {/* Account Cards Grid */}
      <div className="space-y-3">
        {loading ? (
          <div className="py-20 text-center text-slate-400 flex flex-col items-center justify-center gap-2">
            <div className="h-7 w-7 animate-spin rounded-full border-2 border-rose-600 border-t-transparent" />
            <span className="text-xs font-medium">Prioritizing at-risk accounts...</span>
          </div>
        ) : accounts.length === 0 ? (
          <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-500">
            <CheckCircle className="h-10 w-10 text-emerald-500 mx-auto mb-2" />
            <h3 className="font-bold text-slate-800">No matching at-risk accounts found</h3>
            <p className="text-xs text-slate-400 mt-1">Great job! All accounts within this filter scope are retained.</p>
          </div>
        ) : (
          accounts.map((acc, idx) => (
            <div
              key={acc.customer_id}
              className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs hover:border-slate-300 hover:shadow-md transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-4 group"
            >
              {/* Left Column: ID & Model Probability */}
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-slate-900 text-teal-400 font-mono font-bold text-sm">
                  #{idx + 1 + (pagination.page - 1) * pagination.page_size}
                </div>

                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span
                      onClick={() => onSelectCustomer(acc.customer_id)}
                      className="font-mono text-sm font-bold text-slate-900 hover:text-teal-600 cursor-pointer underline-offset-2 hover:underline"
                    >
                      {acc.customer_id}
                    </span>
                    <RiskBadge category={acc.risk_category} size="sm" />
                    {acc.has_active_retention_task && (
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 border border-teal-200">
                        Action Active
                      </span>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
                    <span>
                      Tenure: <strong className="text-slate-800">{acc.tenure} mo</strong>
                    </span>
                    <span>•</span>
                    <span>
                      Contract: <strong className="text-slate-800">{acc.contract}</strong>
                    </span>
                    <span>•</span>
                    <span>
                      Monthly: <strong className="text-slate-800">${acc.monthly_charges.toFixed(2)}</strong>
                    </span>
                    <span>•</span>
                    <span className="truncate max-w-[150px]">{acc.payment_method}</span>
                  </div>
                </div>
              </div>

              {/* Middle Column: Probability Meter */}
              <div className="w-full md:w-56 shrink-0 space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500 font-medium">Estimated Likelihood:</span>
                  <span className="font-extrabold text-slate-900 text-sm">
                    {acc.churn_probability_pct}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200/50">
                  <div
                    className={`h-2 rounded-full ${
                      acc.churn_probability >= 0.6 ? 'bg-rose-500' : 'bg-amber-500'
                    }`}
                    style={{ width: `${acc.churn_probability_pct}%` }}
                  />
                </div>
              </div>

              {/* Right Column: Actions */}
              <div className="flex items-center gap-2 shrink-0 w-full md:w-auto justify-end pt-3 md:pt-0 border-t md:border-t-0 border-slate-100">
                <button
                  onClick={() => onSelectCustomer(acc.customer_id)}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors"
                >
                  <Eye className="h-3.5 w-3.5 text-slate-500" />
                  360° Profile
                </button>

                <button
                  onClick={() =>
                    onOpenCreateTask(
                      acc.customer_id,
                      acc.contract === 'Month-to-month'
                        ? 'Contract Upgrade Offer'
                        : 'CSM Early Lifecycle Health Check'
                    )
                  }
                  className="flex items-center gap-1.5 rounded-lg bg-teal-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-teal-700 transition-all"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  Assign Follow-up
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Pagination */}
      {pagination.total_pages > 1 && (
        <div className="flex items-center justify-between border-t border-slate-200 pt-4 text-xs text-slate-500">
          <span>
            Page {pagination.page} of {pagination.total_pages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPagination((p) => ({ ...p, page: Math.max(1, p.page - 1) }))}
              disabled={pagination.page <= 1 || loading}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={() => setPagination((p) => ({ ...p, page: Math.min(p.total_pages, p.page + 1) }))}
              disabled={pagination.page >= pagination.total_pages || loading}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
