import React from 'react';
import { RiskCategory, PredictedChurn, TaskPriority, TaskStatus } from '../types';

export const RiskBadge: React.FC<{ category: RiskCategory; size?: 'sm' | 'md' }> = ({
  category,
  size = 'md',
}) => {
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs font-semibold';

  if (category === 'High Risk') {
    return (
      <span
        className={`inline-flex items-center gap-1 rounded-full border border-rose-300 bg-rose-50 text-rose-700 ${sizeClasses}`}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-rose-500 animate-pulse" />
        High Risk
      </span>
    );
  }
  if (category === 'Medium Risk') {
    return (
      <span
        className={`inline-flex items-center gap-1 rounded-full border border-amber-300 bg-amber-50 text-amber-700 ${sizeClasses}`}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
        Medium Risk
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border border-emerald-300 bg-emerald-50 text-emerald-700 ${sizeClasses}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
      Low Risk
    </span>
  );
};

export const ChurnBadge: React.FC<{ prediction: PredictedChurn }> = ({ prediction }) => {
  if (prediction === 'Churn') {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800 border border-red-200">
        Predicted Churn
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-sky-100 text-sky-800 border border-sky-200">
      Predicted Stay
    </span>
  );
};

export const TaskStatusBadge: React.FC<{ status: TaskStatus }> = ({ status }) => {
  switch (status) {
    case 'Completed':
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
          Completed
        </span>
      );
    case 'In Progress':
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          In Progress
        </span>
      );
    case 'Open':
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
          Open
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-700">
          {status}
        </span>
      );
  }
};

export const PriorityBadge: React.FC<{ priority: TaskPriority }> = ({ priority }) => {
  switch (priority) {
    case 'High':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold text-rose-600 bg-rose-50 border border-rose-200">
          High
        </span>
      );
    case 'Medium':
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium text-amber-700 bg-amber-50 border border-amber-200">
          Medium
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium text-slate-600 bg-slate-100 border border-slate-200">
          Low
        </span>
      );
  }
};
