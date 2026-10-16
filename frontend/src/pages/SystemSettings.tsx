import React, { useState, useEffect } from 'react';
import {
  Settings,
  Sliders,
  Database,
  ShieldAlert,
  Info,
  CheckCircle,
  Save,
  Server,
  FileCode,
} from 'lucide-react';
import { api } from '../api/client';
import { ThresholdConfig } from '../types';

interface SystemSettingsProps {
  onThresholdsUpdated: () => void;
}

export const SystemSettings: React.FC<SystemSettingsProps> = ({ onThresholdsUpdated }) => {
  const [highThreshold, setHighThreshold] = useState(0.60);
  const [medThreshold, setMedThreshold] = useState(0.35);
  const [isSaving, setIsSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadThresholds();
  }, []);

  const loadThresholds = async () => {
    try {
      const cfg = await api.getThresholds();
      setHighThreshold(cfg.high_threshold);
      setMedThreshold(cfg.medium_threshold);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (medThreshold >= highThreshold) {
      setError('Medium Risk threshold must be strictly lower than High Risk threshold.');
      return;
    }

    try {
      setIsSaving(true);
      setError(null);
      await api.updateThresholds({
        high_threshold: highThreshold,
        medium_threshold: medThreshold,
      });
      setSavedSuccess(true);
      onThresholdsUpdated();
      setTimeout(() => setSavedSuccess(false), 3000);
    } catch (err: any) {
      setError(err.message || 'Failed to update thresholds');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <Settings className="h-5 w-5 text-teal-600" />
          System Settings & Risk Boundary Configuration
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Adjust operational classification boundaries, inspect dataset health, and review prototype runtime status.
        </p>
      </div>

      {/* Threshold Configuration Form */}
      <form onSubmit={handleSave} className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Sliders className="h-4 w-4 text-teal-600" />
              Operational Triage Risk Thresholds
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Maps continuous logistic regression probability outputs into actionable customer triage tiers.
            </p>
          </div>

          <div className="text-[11px] text-slate-400 bg-slate-50 border border-slate-200 rounded px-2 py-1">
            Re-scores all 7,043 accounts upon save
          </div>
        </div>

        {error && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 p-3 text-xs text-rose-700 font-medium">
            {error}
          </div>
        )}

        {savedSuccess && (
          <div className="rounded-lg bg-emerald-50 border border-emerald-200 p-3 text-xs text-emerald-800 font-semibold flex items-center gap-2">
            <CheckCircle className="h-4 w-4 text-emerald-600" />
            Triage thresholds saved! Dataset re-scored successfully.
          </div>
        )}

        <div className="space-y-6">
          {/* High Risk Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-rose-700 flex items-center gap-1.5">
                🚨 High Risk Threshold:
              </span>
              <span className="font-mono text-sm bg-rose-50 border border-rose-200 text-rose-700 px-2 py-0.5 rounded">
                Prob ≥ {(highThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.90"
              step="0.05"
              value={highThreshold}
              onChange={(e) => setHighThreshold(parseFloat(e.target.value))}
              className="w-full accent-rose-600 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">
              Customers above this probability receive priority outreach and executive retention escalation.
            </p>
          </div>

          {/* Medium Risk Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-amber-700 flex items-center gap-1.5">
                ⚠️ Medium Risk Threshold:
              </span>
              <span className="font-mono text-sm bg-amber-50 border border-amber-200 text-amber-700 px-2 py-0.5 rounded">
                Prob ≥ {(medThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.20"
              max="0.49"
              step="0.05"
              value={medThreshold}
              onChange={(e) => setMedThreshold(parseFloat(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <p className="text-[11px] text-slate-500">
              Customers between Medium and High thresholds are enrolled in proactive nurture and auto-pay campaigns.
            </p>
          </div>
        </div>

        {/* Current Active Tiers Preview */}
        <div className="rounded-lg bg-slate-50 border border-slate-200 p-4 space-y-2 text-xs">
          <span className="font-bold text-slate-700 block">Active Triage Bands:</span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-2.5 rounded bg-white border border-rose-200 text-rose-800">
              <span className="font-bold block">🚨 High Risk</span>
              <span className="font-mono text-[11px]">≥ {(highThreshold * 100).toFixed(0)}% Likelihood</span>
            </div>
            <div className="p-2.5 rounded bg-white border border-amber-200 text-amber-800">
              <span className="font-bold block">⚠️ Medium Risk</span>
              <span className="font-mono text-[11px]">
                {(medThreshold * 100).toFixed(0)}% - {(highThreshold * 100 - 1).toFixed(0)}% Likelihood
              </span>
            </div>
            <div className="p-2.5 rounded bg-white border border-emerald-200 text-emerald-800">
              <span className="font-bold block">🛡️ Low Risk</span>
              <span className="font-mono text-[11px]">&lt; {(medThreshold * 100).toFixed(0)}% Likelihood</span>
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={isSaving}
            className="flex items-center gap-1.5 rounded-lg bg-teal-600 px-5 py-2 text-xs font-semibold text-white shadow-sm hover:bg-teal-700 disabled:opacity-50 transition-all"
          >
            <Save className="h-4 w-4" />
            {isSaving ? 'Re-scoring...' : 'Save & Update Engine'}
          </button>
        </div>
      </form>

      {/* Dataset & Engine Health */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Database className="h-4 w-4 text-slate-500" />
          Data Pipeline & Storage Architecture
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
            <span className="text-slate-400 block">Dataset Source</span>
            <span className="font-semibold text-slate-900 block">IBM Telco Customer Churn</span>
            <span className="text-[11px] text-slate-500 block">7,043 rows • 21 feature columns</span>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
            <span className="text-slate-400 block">Machine Learning Pipeline</span>
            <span className="font-semibold text-slate-900 block">Scikit-learn ColumnTransformer + LR</span>
            <span className="text-[11px] text-slate-500 block">models/best_model.joblib (C=10.0)</span>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
            <span className="text-slate-400 block">Local Task Persistence</span>
            <span className="font-semibold text-slate-900 block">SQLite Database (data/retention.db)</span>
            <span className="text-[11px] text-slate-500 block">Persistent across server restarts</span>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
            <span className="text-slate-400 block">Backend Server</span>
            <span className="font-semibold text-slate-900 block">Python FastAPI + Uvicorn</span>
            <span className="text-[11px] text-slate-500 block">REST endpoints with CORS support</span>
          </div>
        </div>

        {/* Local Prototype Context Alert */}
        <div className="rounded-lg border border-blue-200 bg-blue-50/70 p-4 text-xs text-blue-900 space-y-1">
          <span className="font-bold flex items-center gap-1.5">
            <Info className="h-4 w-4 text-blue-600" />
            Prototype Operating Notice
          </span>
          <p className="text-blue-800 leading-relaxed">
            StayWise is configured as a local enterprise prototype application. Machine learning inference is performed dynamically in Python using the saved Scikit-learn pipeline. Retention actions, team assignments, and interaction notes are stored locally in SQLite. Outbound communications (emails, phone calls) are simulated as workflow items without external dispatch.
          </p>
        </div>
      </div>
    </div>
  );
};
