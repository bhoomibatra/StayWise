import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Award,
  CheckCircle2,
  HelpCircle,
  BarChart2,
  Activity,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { api } from '../api/client';
import { ModelPerformanceData } from '../types';

export const ModelPerformance: React.FC = () => {
  const [data, setData] = useState<ModelPerformanceData | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('Logistic Regression');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPerformance();
  }, []);

  const loadPerformance = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getModelPerformance();
      setData(res);
      if (res.best_model) setSelectedModel(res.best_model);
    } catch (err: any) {
      setError(err.message || 'Failed to load model performance data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-3 text-slate-500">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
        <span className="text-sm font-medium">Loading holdout test benchmarks and evaluation curves...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
          <p className="font-bold">Failed to load performance metrics</p>
          <p className="mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const { summary, best_model, selection_rationale, feature_importances, detailed_evaluations, holdout_test_size, methodology } = data;

  // Prepare combined ROC data for Recharts
  const modelsList = Object.keys(detailed_evaluations);
  const rocPointsCount = detailed_evaluations['Logistic Regression']?.roc_curve?.length || 35;
  const combinedRocData = [];
  for (let i = 0; i < rocPointsCount; i++) {
    const pt: Record<string, any> = {
      fpr: detailed_evaluations['Logistic Regression']?.roc_curve[i]?.fpr ?? (i / (rocPointsCount - 1)),
    };
    modelsList.forEach((name) => {
      pt[name] = detailed_evaluations[name]?.roc_curve[i]?.tpr;
    });
    pt['Random Guess'] = pt.fpr;
    combinedRocData.push(pt);
  }

  // Active model confusion matrix
  const activeEval = detailed_evaluations[selectedModel] || detailed_evaluations['Logistic Regression'];
  const cm = activeEval?.confusion_matrix || { tn: 926, fp: 109, fn: 165, tp: 209, total: 1409 };

  const MODEL_COLORS: Record<string, string> = {
    'Logistic Regression': '#0d9488', // Teal
    'K-Nearest Neighbors': '#3b82f6', // Blue
    'Random Forest': '#f59e0b',        // Amber
    'Support Vector Machine': '#8b5cf6', // Violet
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Top Banner: Algorithm Selection Rationale */}
      <div className="rounded-xl border border-teal-200 bg-gradient-to-r from-teal-50/70 via-white to-emerald-50/40 p-6 shadow-sm">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-600 text-white font-bold text-xs">
                <Award className="h-4 w-4" />
              </span>
              <h2 className="text-base font-bold text-slate-900 tracking-tight">
                Selected Production Architecture: {best_model}
              </h2>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                Optimal Trade-Off
              </span>
            </div>
            <p className="text-xs text-slate-600 max-w-4xl leading-relaxed mt-1">
              {selection_rationale}
            </p>
          </div>

          <div className="rounded-lg bg-white border border-teal-200 p-3 text-xs text-slate-600 shrink-0 space-y-1">
            <div className="font-semibold text-slate-900 flex items-center gap-1.5">
              <ShieldCheck className="h-4 w-4 text-teal-600" />
              Holdout Evaluation Set
            </div>
            <div className="text-[11px] text-slate-500">
              N = <strong>{holdout_test_size.toLocaleString()}</strong> untouched test rows (20% split)
            </div>
          </div>
        </div>
      </div>

      {/* Model Benchmark Table */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/50 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900">4-Algorithm Holdout Benchmark Comparison</h3>
            <p className="text-xs text-slate-500">Evaluated on the exact same stratified holdout test split (1,409 rows)</p>
          </div>
          <span className="text-[11px] font-medium text-slate-400">Zero Data Leakage • Pipeline Enforced</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase text-[11px]">
              <tr>
                <th className="py-3 px-4">Algorithm</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Recall (Churners Caught)</th>
                <th className="py-3 px-4">F1-Score</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(summary).map(([name, m]) => {
                const isSelected = name === best_model;
                return (
                  <tr
                    key={name}
                    onClick={() => setSelectedModel(name)}
                    className={`cursor-pointer transition-colors ${
                      selectedModel === name
                        ? 'bg-teal-50/60 font-semibold'
                        : 'hover:bg-slate-50/60'
                    }`}
                  >
                    <td className="py-3 px-4 font-bold text-slate-900 flex items-center gap-2">
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: MODEL_COLORS[name] || '#94a3b8' }}
                      />
                      <span>{name}</span>
                    </td>
                    <td className="py-3 px-4">
                      {isSelected ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-bold text-teal-700 bg-teal-100/80 px-2 py-0.5 rounded-full border border-teal-200">
                          <CheckCircle2 className="h-3 w-3" />
                          Selected
                        </span>
                      ) : (
                        <span className="text-[11px] text-slate-400">Benchmarked</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-sm font-extrabold text-slate-900">
                      {(m.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 font-bold text-slate-800">{m.f1.toFixed(4)}</td>
                    <td className="py-3 px-4 font-bold text-teal-700">{m.roc_auc.toFixed(4)}</td>
                    <td className="py-3 px-4 text-slate-700">{(m.precision * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 text-slate-700">{(m.accuracy * 100).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 2: Overlaid ROC Curves & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Overlaid ROC Curves */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Overlaid Receiver Operating Characteristic (ROC)</h3>
              <p className="text-xs text-slate-500 mt-0.5">True Positive Rate vs False Positive Rate across classification thresholds</p>
            </div>
            <span className="text-xs font-mono text-slate-400">Holdout N=1,409</span>
          </div>

          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedRocData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="fpr"
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  label={{ value: 'False Positive Rate (1 - Specificity)', position: 'insideBottom', offset: -5, fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  label={{ value: 'True Positive Rate (Recall)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11 }}
                />
                <Tooltip
                  formatter={(val: any, name: any) => [`${(Number(val) * 100).toFixed(1)}%`, name]}
                  labelFormatter={(lbl) => `FPR: ${(Number(lbl) * 100).toFixed(1)}%`}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }} />
                <Line
                  type="monotone"
                  dataKey="Logistic Regression"
                  stroke={MODEL_COLORS['Logistic Regression']}
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="Random Forest"
                  stroke={MODEL_COLORS['Random Forest']}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="Support Vector Machine"
                  stroke={MODEL_COLORS['Support Vector Machine']}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="K-Nearest Neighbors"
                  stroke={MODEL_COLORS['K-Nearest Neighbors']}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="Random Guess"
                  stroke="#cbd5e1"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Holdout Confusion Matrix */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-slate-900">Confusion Matrix</h3>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="rounded-lg border border-slate-300 px-2 py-1 text-xs text-slate-700 bg-white"
              >
                {modelsList.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Holdout classification counts for <strong className="text-slate-800">{selectedModel}</strong>
            </p>

            {/* Matrix 2x2 Grid */}
            <div className="grid grid-cols-2 gap-3 text-center">
              {/* True Negative */}
              <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/40">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-800 block">
                  True Negatives (TN)
                </span>
                <span className="text-2xl font-extrabold text-emerald-700 mt-1 block">
                  {cm.tn.toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-500 mt-0.5 block">Actual Stay → Pred Stay</span>
              </div>

              {/* False Positive */}
              <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/40">
                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800 block">
                  False Positives (FP)
                </span>
                <span className="text-2xl font-extrabold text-amber-700 mt-1 block">
                  {cm.fp.toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-500 mt-0.5 block">Actual Stay → Pred Churn</span>
              </div>

              {/* False Negative */}
              <div className="p-4 rounded-xl border border-rose-300 bg-rose-50/70">
                <span className="text-[10px] font-bold uppercase tracking-wider text-rose-800 block">
                  False Negatives (FN)
                </span>
                <span className="text-2xl font-extrabold text-rose-700 mt-1 block">
                  {cm.fn.toLocaleString()}
                </span>
                <span className="text-[10px] text-rose-600 font-semibold mt-0.5 block">
                  Missed Churners!
                </span>
              </div>

              {/* True Positive */}
              <div className="p-4 rounded-xl border border-teal-200 bg-teal-50/40">
                <span className="text-[10px] font-bold uppercase tracking-wider text-teal-800 block">
                  True Positives (TP)
                </span>
                <span className="text-2xl font-extrabold text-teal-700 mt-1 block">
                  {cm.tp.toLocaleString()}
                </span>
                <span className="text-[10px] text-slate-500 mt-0.5 block">Actual Churn → Pred Churn</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-500 space-y-1">
            <div className="flex justify-between">
              <span>Sensitivity / Recall:</span>
              <strong className="text-slate-800">{((cm.tp / (cm.tp + cm.fn)) * 100).toFixed(1)}%</strong>
            </div>
            <div className="flex justify-between">
              <span>Specificity:</span>
              <strong className="text-slate-800">{((cm.tn / (cm.tn + cm.fp)) * 100).toFixed(1)}%</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Row 3: Feature Importances */}
      {feature_importances.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Random Forest Top 10 Feature Importances</h3>
              <p className="text-xs text-slate-500 mt-0.5">Mean decrease in Gini impurity across 100 decision trees</p>
            </div>
            <span className="text-xs text-slate-400">Gini Impurity</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={feature_importances.map((f) => ({
                  feature: f.feature.replace('cat__', '').replace('num__', '').replace('_', ' '),
                  importance: Number((f.importance * 100).toFixed(1)),
                }))}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} unit="%" />
                <YAxis dataKey="feature" type="category" tick={{ fontSize: 11, fill: '#334155' }} />
                <Tooltip formatter={(val: any) => [`${val}%`, 'Relative Importance']} />
                <Bar dataKey="importance" fill="#0d9488" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};
