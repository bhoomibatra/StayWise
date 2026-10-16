import React, { useState, useEffect } from 'react';
import {
  Search,
  Filter,
  Download,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Eye,
  PlusCircle,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { api } from '../api/client';
import { CustomerSummaryItem, PaginationMeta } from '../types';
import { RiskBadge, ChurnBadge } from '../components/Badges';

interface CustomerDirectoryProps {
  onSelectCustomer: (customerId: string) => void;
  onOpenCreateTask: (customerId: string) => void;
}

export const CustomerDirectory: React.FC<CustomerDirectoryProps> = ({
  onSelectCustomer,
  onOpenCreateTask,
}) => {
  const [customers, setCustomers] = useState<CustomerSummaryItem[]>([]);
  const [pagination, setPagination] = useState<PaginationMeta>({
    total: 0,
    page: 1,
    page_size: 25,
    total_pages: 1,
  });
  const [search, setSearch] = useState('');
  const [predictedChurn, setPredictedChurn] = useState('All');
  const [riskCategory, setRiskCategory] = useState('All');
  const [contract, setContract] = useState('All');
  const [sortBy, setSortBy] = useState('churn_probability');
  const [sortOrder, setSortOrder] = useState('desc');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCustomers = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getCustomers({
        search,
        predicted_churn: predictedChurn,
        risk_category: riskCategory,
        contract,
        sort_by: sortBy,
        sort_order: sortOrder,
        page: pagination.page,
        page_size: pagination.page_size,
      });
      setCustomers(res.items);
      setPagination(res.pagination);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch customers');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomers();
  }, [search, predictedChurn, riskCategory, contract, sortBy, sortOrder, pagination.page]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value);
    setPagination((prev) => ({ ...prev, page: 1 }));
  };

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
    setPagination((prev) => ({ ...prev, page: 1 }));
  };

  const handleExport = () => {
    const url = api.getExportUrl({
      search,
      predicted_churn: predictedChurn,
      risk_category: riskCategory,
      contract,
    });
    window.open(url, '_blank');
  };

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Search and Filters Bar */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
          {/* Search Input */}
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={search}
              onChange={handleSearchChange}
              placeholder="Search by Customer ID (e.g. 9237-HQITU)..."
              className="w-full rounded-lg border border-slate-300 pl-10 pr-4 py-2 text-xs text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none"
            />
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors shadow-xs"
            >
              <Download className="h-3.5 w-3.5 text-slate-500" />
              Export Filtered CSV
            </button>
          </div>
        </div>

        {/* Filters Grid */}
        <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center gap-3 text-xs">
          <span className="font-semibold text-slate-500 flex items-center gap-1">
            <Filter className="h-3.5 w-3.5" />
            Filters:
          </span>

          {/* Risk Tier Filter */}
          <select
            value={riskCategory}
            onChange={(e) => {
              setRiskCategory(e.target.value);
              setPagination((p) => ({ ...p, page: 1 }));
            }}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 bg-white focus:border-teal-500 focus:outline-none"
          >
            <option value="All">All Risk Tiers</option>
            <option value="High Risk">🚨 High Risk Only</option>
            <option value="Medium Risk">⚠️ Medium Risk Only</option>
            <option value="Low Risk">🛡️ Low Risk Only</option>
          </select>

          {/* Churn Prediction Filter */}
          <select
            value={predictedChurn}
            onChange={(e) => {
              setPredictedChurn(e.target.value);
              setPagination((p) => ({ ...p, page: 1 }));
            }}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 bg-white focus:border-teal-500 focus:outline-none"
          >
            <option value="All">All Predictions</option>
            <option value="Churn">Predicted Churn</option>
            <option value="Stay">Predicted Stay</option>
          </select>

          {/* Contract Filter */}
          <select
            value={contract}
            onChange={(e) => {
              setContract(e.target.value);
              setPagination((p) => ({ ...p, page: 1 }));
            }}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 bg-white focus:border-teal-500 focus:outline-none"
          >
            <option value="All">All Contracts</option>
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>

          <span className="ml-auto text-slate-500 text-xs">
            Showing <strong className="text-slate-800">{pagination.total.toLocaleString()}</strong> matching accounts
          </span>
        </div>
      </div>

      {/* Main Table */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3.5 px-4">Customer ID</th>
                <th
                  onClick={() => handleSort('churn_probability')}
                  className="py-3.5 px-4 cursor-pointer hover:text-slate-900 select-none"
                >
                  <div className="flex items-center gap-1">
                    <span>Likelihood</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-400" />
                  </div>
                </th>
                <th className="py-3.5 px-4">Risk Tier</th>
                <th className="py-3.5 px-4">Predicted Outcome</th>
                <th className="py-3.5 px-4">Contract</th>
                <th
                  onClick={() => handleSort('tenure')}
                  className="py-3.5 px-4 cursor-pointer hover:text-slate-900 select-none"
                >
                  <div className="flex items-center gap-1">
                    <span>Tenure</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-400" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('MonthlyCharges')}
                  className="py-3.5 px-4 cursor-pointer hover:text-slate-900 select-none"
                >
                  <div className="flex items-center gap-1">
                    <span>Monthly</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-400" />
                  </div>
                </th>
                <th className="py-3.5 px-4">Payment Method</th>
                <th className="py-3.5 px-4">Historical Actual</th>
                <th className="py-3.5 px-4 text-right">Retention Actions</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={10} className="py-16 text-center text-slate-400">
                    <div className="flex items-center justify-center gap-2">
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
                      <span>Loading customer batch...</span>
                    </div>
                  </td>
                </tr>
              ) : customers.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-16 text-center text-slate-400">
                    <AlertCircle className="h-8 w-8 text-slate-300 mx-auto mb-2" />
                    <p className="font-semibold text-slate-600">No customers match the active filters</p>
                    <p className="text-xs text-slate-400 mt-1">Try broadening your search or resetting filters.</p>
                  </td>
                </tr>
              ) : (
                customers.map((c) => (
                  <tr
                    key={c.customer_id}
                    onClick={() => onSelectCustomer(c.customer_id)}
                    className="hover:bg-slate-50/80 cursor-pointer transition-colors group"
                  >
                    <td className="py-3 px-4 font-mono font-semibold text-slate-900 flex items-center gap-2">
                      <span>{c.customer_id}</span>
                      {c.has_active_retention_task && (
                        <span
                          className="h-2 w-2 rounded-full bg-teal-500"
                          title="Has active retention task"
                        />
                      )}
                    </td>

                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-900">{c.churn_probability_pct}%</span>
                        <div className="w-12 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-1.5 rounded-full ${
                              c.churn_probability >= 0.6
                                ? 'bg-rose-500'
                                : c.churn_probability >= 0.35
                                ? 'bg-amber-500'
                                : 'bg-emerald-500'
                            }`}
                            style={{ width: `${c.churn_probability_pct}%` }}
                          />
                        </div>
                      </div>
                    </td>

                    <td className="py-3 px-4">
                      <RiskBadge category={c.risk_category} size="sm" />
                    </td>

                    <td className="py-3 px-4">
                      <ChurnBadge prediction={c.predicted_churn} />
                    </td>

                    <td className="py-3 px-4 text-slate-700">{c.contract}</td>

                    <td className="py-3 px-4 text-slate-700">{c.tenure} mo</td>

                    <td className="py-3 px-4 font-semibold text-slate-900">
                      ${c.monthly_charges.toFixed(2)}
                    </td>

                    <td className="py-3 px-4 text-slate-600 truncate max-w-[140px]" title={c.payment_method}>
                      {c.payment_method}
                    </td>

                    <td className="py-3 px-4 text-slate-500">
                      {c.historical_churn ? (
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                            c.historical_churn === 'Yes'
                              ? 'bg-rose-50 text-rose-700 border border-rose-200'
                              : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {c.historical_churn === 'Yes' ? 'Actual Churn' : 'Actual Retain'}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>

                    <td className="py-3 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => onSelectCustomer(c.customer_id)}
                          className="p-1.5 rounded-md text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
                          title="View 360 Profile"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => onOpenCreateTask(c.customer_id)}
                          className="p-1.5 rounded-md text-teal-600 hover:text-teal-800 hover:bg-teal-50 transition-colors"
                          title="Assign Retention Follow-up"
                        >
                          <PlusCircle className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div className="flex items-center justify-between border-t border-slate-200 bg-slate-50/50 px-6 py-3.5 text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <span>Page Size:</span>
            <select
              value={pagination.page_size}
              onChange={(e) =>
                setPagination((prev) => ({
                  ...prev,
                  page_size: Number(e.target.value),
                  page: 1,
                }))
              }
              className="rounded border border-slate-300 px-2 py-1 text-xs bg-white"
            >
              <option value={15}>15</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
            <span className="text-slate-400 ml-2">
              Page {pagination.page} of {pagination.total_pages}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPagination((prev) => ({ ...prev, page: Math.max(1, prev.page - 1) }))}
              disabled={pagination.page <= 1 || loading}
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition-colors"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </button>
            <button
              onClick={() =>
                setPagination((prev) => ({
                  ...prev,
                  page: Math.min(prev.total_pages, prev.page + 1),
                }))
              }
              disabled={pagination.page >= pagination.total_pages || loading}
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition-colors"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
