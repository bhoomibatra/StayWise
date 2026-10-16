import React from 'react';
import { RefreshCw, Database, Sparkles, PlusCircle } from 'lucide-react';

interface HeaderProps {
  title: string;
  subtitle: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  onOpenCreateTask?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  onRefresh,
  isRefreshing = false,
  onOpenCreateTask,
}) => {
  return (
    <header className="sticky top-0 z-20 flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 bg-white/95 px-8 py-4 backdrop-blur shadow-sm">
      <div className="flex flex-col">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">{title}</h1>
        <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        {/* Dataset Indicator */}
        <div className="hidden sm:flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-600">
          <Database className="h-3.5 w-3.5 text-slate-400" />
          <span>7,043 Records Scored</span>
        </div>

        {/* Model Pipeline Pill */}
        <div className="hidden md:flex items-center gap-1.5 rounded-lg border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-medium text-teal-800">
          <Sparkles className="h-3.5 w-3.5 text-teal-600" />
          <span>Scikit-learn Pipeline (LR C=10.0)</span>
        </div>

        {/* Refresh button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 shadow-sm transition-colors disabled:opacity-50"
            title="Refresh live metrics"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin text-teal-600' : 'text-slate-500'}`} />
            <span className="hidden sm:inline">Refresh</span>
          </button>
        )}

        {/* Quick Task Creation */}
        {onOpenCreateTask && (
          <button
            onClick={onOpenCreateTask}
            className="flex items-center gap-1.5 rounded-lg bg-teal-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 transition-all"
          >
            <PlusCircle className="h-4 w-4" />
            <span>New Retention Action</span>
          </button>
        )}
      </div>
    </header>
  );
};
