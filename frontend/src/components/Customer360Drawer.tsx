import React, { useState, useEffect } from 'react';
import {
  X,
  ShieldAlert,
  ShieldCheck,
  CreditCard,
  Wifi,
  Calendar,
  Clock,
  Send,
  PlusCircle,
  FileText,
  UserCheck,
  CheckCircle,
  ChevronRight,
} from 'lucide-react';
import { api } from '../api/client';
import { CustomerProfile, RetentionTask, RetentionNote } from '../types';
import { RiskBadge, ChurnBadge, TaskStatusBadge, PriorityBadge } from './Badges';

interface Customer360DrawerProps {
  customerId: string | null;
  onClose: () => void;
  onOpenCreateTask: (cid: string, taskType?: string, notes?: string) => void;
}

export const Customer360Drawer: React.FC<Customer360DrawerProps> = ({
  customerId,
  onClose,
  onOpenCreateTask,
}) => {
  const [profile, setProfile] = useState<CustomerProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [noteAuthor, setNoteAuthor] = useState('Sarah Chen (CSM)');
  const [newNote, setNewNote] = useState('');
  const [isAddingNote, setIsAddingNote] = useState(false);

  const fetchProfile = async (id: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getCustomerProfile(id);
      setProfile(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load customer profile');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (customerId) {
      fetchProfile(customerId);
    } else {
      setProfile(null);
    }
  }, [customerId]);

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!profile || !newNote.trim()) return;

    try {
      setIsAddingNote(true);
      await api.addCustomerNote({
        customer_id: profile.customer_id,
        author: noteAuthor,
        note: newNote.trim(),
      });
      setNewNote('');
      // Reload profile to refresh note timeline
      await fetchProfile(profile.customer_id);
    } catch (err: any) {
      alert(`Failed to save note: ${err.message}`);
    } finally {
      setIsAddingNote(false);
    }
  };

  if (!customerId) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/50 backdrop-blur-sm transition-opacity">
      <div className="w-full max-w-2xl bg-white shadow-2xl border-l border-slate-200 flex flex-col h-full animate-in slide-in-from-right duration-200">
        {/* Drawer Header */}
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4 bg-slate-50/80">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-900 text-teal-400 font-mono font-bold shadow-sm">
              360°
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-slate-900 font-mono">
                  {profile ? profile.customer_id : customerId}
                </h2>
                {profile && <RiskBadge category={profile.model_output.risk_category} />}
                {profile && <ChurnBadge prediction={profile.model_output.predicted_churn} />}
              </div>
              <p className="text-xs text-slate-500">Customer 360° Deep-Dive & Retention Cockpit</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-2 text-slate-400 hover:bg-slate-200/60 hover:text-slate-700 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Drawer Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {loading && (
            <div className="flex flex-col items-center justify-center py-20 text-slate-400 gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
              <p className="text-sm">Retrieving customer account data and model inference...</p>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
              {error}
            </div>
          )}

          {profile && !loading && (
            <>
              {/* Top Model Triage Banner */}
              <div className="rounded-xl border border-slate-200 bg-gradient-to-r from-slate-50 via-white to-slate-50 p-4 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      Model Estimate (Holdout Calibrated)
                    </span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-3xl font-extrabold tracking-tight text-slate-900">
                        {profile.model_output.churn_probability_pct}%
                      </span>
                      <span className="text-xs text-slate-500 font-medium">Estimated Churn Likelihood</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <button
                      onClick={() =>
                        onOpenCreateTask(
                          profile.customer_id,
                          profile.recommended_actions[0]?.type || 'Contract Upgrade Offer',
                          profile.recommended_actions[0]?.description || ''
                        )
                      }
                      className="flex items-center gap-1.5 rounded-lg bg-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-teal-700 transition-all"
                    >
                      <PlusCircle className="h-4 w-4" />
                      Assign Retention Action
                    </button>
                  </div>
                </div>

                {/* Probability Bar */}
                <div className="mt-3 w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      profile.model_output.churn_probability >= 0.6
                        ? 'bg-rose-500'
                        : profile.model_output.churn_probability >= 0.35
                        ? 'bg-amber-500'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.min(100, profile.model_output.churn_probability_pct)}%` }}
                  />
                </div>

                <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                  <span>Thresholds: Med ≥ 35% | High ≥ 60%</span>
                  {profile.model_output.historical_actual_churn && (
                    <span className="font-medium text-slate-600">
                      Historical Record: <span className="underline">{profile.model_output.historical_actual_churn} Churned</span>
                    </span>
                  )}
                </div>
              </div>

              {/* Empirical Risk Drivers & Protective Factors */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Risk Drivers */}
                <div className="rounded-xl border border-rose-100 bg-rose-50/40 p-4">
                  <div className="flex items-center gap-2 mb-3 text-rose-800 font-semibold text-xs uppercase tracking-wider">
                    <ShieldAlert className="h-4 w-4 text-rose-600" />
                    Empirical Risk Indicators ({profile.risk_drivers.length})
                  </div>
                  {profile.risk_drivers.length === 0 ? (
                    <p className="text-xs text-slate-500 italic">No acute risk triggers detected.</p>
                  ) : (
                    <ul className="space-y-2.5">
                      {profile.risk_drivers.map((d, i) => (
                        <li key={i} className="text-xs">
                          <span className="font-semibold text-rose-900 flex items-center gap-1">
                            • {d.title}
                          </span>
                          <p className="text-slate-600 mt-0.5 pl-3 leading-relaxed">{d.detail}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                {/* Protective Factors */}
                <div className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-4">
                  <div className="flex items-center gap-2 mb-3 text-emerald-800 font-semibold text-xs uppercase tracking-wider">
                    <ShieldCheck className="h-4 w-4 text-emerald-600" />
                    Account Stabilizers ({profile.protective_factors.length})
                  </div>
                  {profile.protective_factors.length === 0 ? (
                    <p className="text-xs text-slate-500 italic">No strong protective factors identified.</p>
                  ) : (
                    <ul className="space-y-2.5">
                      {profile.protective_factors.map((p, i) => (
                        <li key={i} className="text-xs">
                          <span className="font-semibold text-emerald-900 flex items-center gap-1">
                            ✓ {p.title}
                          </span>
                          <p className="text-slate-600 mt-0.5 pl-3 leading-relaxed">{p.detail}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {/* Account, Demographics & Billing Details */}
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                  <CreditCard className="h-4 w-4 text-slate-400" />
                  Contract & Financial Profile
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-slate-500 block">Contract</span>
                    <span className="font-semibold text-slate-900 text-sm mt-0.5 block">
                      {profile.account_details.contract}
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-slate-500 block">Tenure</span>
                    <span className="font-semibold text-slate-900 text-sm mt-0.5 block">
                      {profile.account_details.tenure} months
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-slate-500 block">Monthly Rate</span>
                    <span className="font-semibold text-slate-900 text-sm mt-0.5 block">
                      ${profile.account_details.monthly_charges.toFixed(2)}
                    </span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-slate-500 block">Lifetime Total</span>
                    <span className="font-semibold text-slate-900 text-sm mt-0.5 block">
                      {profile.account_details.total_charges
                        ? `$${profile.account_details.total_charges.toFixed(2)}`
                        : 'N/A'}
                    </span>
                  </div>
                </div>

                <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-600">
                  <div>
                    <span className="text-slate-400">Payment Channel: </span>
                    <span className="font-medium text-slate-800">{profile.account_details.payment_method}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Paperless Billing: </span>
                    <span className="font-medium text-slate-800">{profile.account_details.paperless_billing}</span>
                  </div>
                </div>
              </div>

              {/* Services Matrix */}
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                  <Wifi className="h-4 w-4 text-slate-400" />
                  Subscribed Service Footprint
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                  {Object.entries(profile.services).map(([key, val]) => {
                    const formatted = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
                    const isPositive = val === 'Yes' || (val !== 'No' && val !== 'No internet service' && val !== 'No phone service');
                    return (
                      <div
                        key={key}
                        className={`p-2 rounded-lg border flex items-center justify-between ${
                          isPositive
                            ? 'bg-teal-50/50 border-teal-200 text-teal-900'
                            : 'bg-slate-50 border-slate-100 text-slate-500'
                        }`}
                      >
                        <span className="truncate">{formatted}</span>
                        <span className="font-medium shrink-0 ml-1.5">{val}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Recommended Retention Playbook */}
              <div className="rounded-xl border border-teal-200 bg-teal-50/40 p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-teal-900 mb-3 flex items-center gap-1.5">
                  <CheckCircle className="h-4 w-4 text-teal-600" />
                  Recommended Retention Interventions
                </h3>

                <div className="space-y-2.5">
                  {profile.recommended_actions.map((act, i) => (
                    <div
                      key={i}
                      className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-3 rounded-lg bg-white border border-teal-100 shadow-xs"
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold text-slate-900">{act.title}</span>
                          <PriorityBadge priority={act.priority} />
                        </div>
                        <p className="text-xs text-slate-600">{act.description}</p>
                      </div>

                      <button
                        onClick={() =>
                          onOpenCreateTask(profile.customer_id, act.type, act.description)
                        }
                        className="shrink-0 flex items-center gap-1 rounded-md bg-teal-600 px-3 py-1.5 text-[11px] font-semibold text-white hover:bg-teal-700 transition-colors"
                      >
                        Launch
                        <ChevronRight className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Retention History & Persistent Activity Log */}
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                    <FileText className="h-4 w-4 text-slate-400" />
                    Retention Timeline & Logged Notes ({profile.retention_history.notes.length})
                  </h3>
                </div>

                {/* Existing Tasks */}
                {profile.retention_history.tasks.length > 0 && (
                  <div className="space-y-2">
                    <span className="text-[11px] font-semibold text-slate-400 uppercase">Assigned Tasks</span>
                    {profile.retention_history.tasks.map((t) => (
                      <div
                        key={t.id}
                        className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex items-start justify-between gap-3 text-xs"
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-slate-900">{t.task_type}</span>
                            <TaskStatusBadge status={t.status} />
                            <PriorityBadge priority={t.priority} />
                          </div>
                          {t.notes && <p className="text-slate-600 mt-1">{t.notes}</p>}
                          <div className="flex items-center gap-3 text-[11px] text-slate-400 mt-1.5">
                            <span>Owner: {t.assigned_to}</span>
                            <span>•</span>
                            <span>Due: {t.due_date || 'No date set'}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Note Log List */}
                <div className="space-y-2.5">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase">Interaction Notes</span>
                  {profile.retention_history.notes.length === 0 ? (
                    <p className="text-xs text-slate-400 italic">No interaction notes recorded yet.</p>
                  ) : (
                    profile.retention_history.notes.map((n) => (
                      <div key={n.id} className="p-3 rounded-lg border border-slate-100 bg-slate-50 text-xs">
                        <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                          <span className="font-semibold text-slate-700">{n.author}</span>
                          <span>{new Date(n.created_at).toLocaleDateString()}</span>
                        </div>
                        <p className="text-slate-700">{n.note}</p>
                      </div>
                    ))
                  )}
                </div>

                {/* Add Note Form */}
                <form onSubmit={handleAddNote} className="pt-2 border-t border-slate-100 space-y-2">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={newNote}
                      onChange={(e) => setNewNote(e.target.value)}
                      placeholder="Add customer interaction log or call summary..."
                      className="flex-1 rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-900 focus:border-teal-500 focus:outline-none"
                    />
                    <button
                      type="submit"
                      disabled={isAddingNote || !newNote.trim()}
                      className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-900 disabled:opacity-50 flex items-center gap-1"
                    >
                      <Send className="h-3 w-3" />
                      Log
                    </button>
                  </div>
                </form>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
