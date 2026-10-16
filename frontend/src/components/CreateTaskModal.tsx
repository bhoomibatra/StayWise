import React, { useState, useEffect } from 'react';
import { X, Calendar, User, Tag, FileText, CheckCircle2 } from 'lucide-react';
import { api } from '../api/client';
import { TaskPriority } from '../types';

interface CreateTaskModalProps {
  isOpen: boolean;
  onClose: () => void;
  onTaskCreated: () => void;
  initialCustomerId?: string;
  initialTaskType?: string;
  initialNotes?: string;
}

export const CreateTaskModal: React.FC<CreateTaskModalProps> = ({
  isOpen,
  onClose,
  onTaskCreated,
  initialCustomerId = '',
  initialTaskType = 'Contract Upgrade Offer',
  initialNotes = '',
}) => {
  const [customerId, setCustomerId] = useState(initialCustomerId);
  const [taskType, setTaskType] = useState(initialTaskType);
  const [priority, setPriority] = useState<TaskPriority>('High');
  const [assignedTo, setAssignedTo] = useState('Sarah Chen (CSM)');
  const [notes, setNotes] = useState(initialNotes);
  const [dueDate, setDueDate] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setCustomerId(initialCustomerId);
      setTaskType(initialTaskType || 'Contract Upgrade Offer');
      setNotes(initialNotes || '');
      setError(null);
      // Default due date: 7 days from now
      const d = new Date();
      d.setDate(d.getDate() + 7);
      setDueDate(d.toISOString().split('T')[0]);
    }
  }, [isOpen, initialCustomerId, initialTaskType, initialNotes]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!customerId.trim()) {
      setError('Please provide a Customer ID.');
      return;
    }

    try {
      setIsSubmitting(true);
      setError(null);
      await api.createRetentionTask({
        customer_id: customerId.trim().toUpperCase(),
        task_type: taskType,
        priority,
        assigned_to: assignedTo,
        notes,
        due_date: dueDate,
      });
      onTaskCreated();
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to create retention task');
    } finally {
      setIsSubmitting(false);
    }
  };

  const taskTypeOptions = [
    'Contract Upgrade Offer',
    'Auto-Pay Migration Bonus',
    'Value Add-on Onboarding',
    'CSM Early Lifecycle Health Check',
    'Billing Review & Rate Negotiation',
    'Executive Escalation Call',
    'Loyalty Milestone Recognition',
  ];

  const assigneeOptions = [
    'Sarah Chen (CSM)',
    'Alex Rivera (Retention Specialist)',
    'Michael Scott (Account Executive)',
    'Priya Sharma (Customer Success Lead)',
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-2xl bg-white shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50/50 px-6 py-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-100 text-teal-700">
              <CheckCircle2 className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">Create Retention Action</h2>
              <p className="text-xs text-slate-500">Assign account intervention to retention team</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="rounded-lg bg-rose-50 border border-rose-200 p-3 text-xs text-rose-700 font-medium">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Customer ID */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Customer ID <span className="text-rose-500">*</span>
              </label>
              <input
                type="text"
                required
                value={customerId}
                onChange={(e) => setCustomerId(e.target.value)}
                placeholder="e.g. 9237-HQITU"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none uppercase font-mono"
              />
            </div>

            {/* Priority */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center gap-1">
                <Tag className="h-3.5 w-3.5 text-slate-400" />
                Priority Tier
              </label>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value as TaskPriority)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none"
              >
                <option value="High">🚨 High Priority</option>
                <option value="Medium">⚠️ Medium Priority</option>
                <option value="Low">🛡️ Standard / Low</option>
              </select>
            </div>
          </div>

          {/* Action Type */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">Action Type</label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none"
            >
              {taskTypeOptions.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Assigned To */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center gap-1">
                <User className="h-3.5 w-3.5 text-slate-400" />
                Owner
              </label>
              <select
                value={assignedTo}
                onChange={(e) => setAssignedTo(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none"
              >
                {assigneeOptions.map((owner) => (
                  <option key={owner} value={owner}>
                    {owner}
                  </option>
                ))}
              </select>
            </div>

            {/* Due Date */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center gap-1">
                <Calendar className="h-3.5 w-3.5 text-slate-400" />
                Target Due Date
              </label>
              <input
                type="date"
                value={dueDate}
                onChange={(e) => setDueDate(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none"
              />
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center gap-1">
              <FileText className="h-3.5 w-3.5 text-slate-400" />
              Follow-up Strategy / Context Notes
            </label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Outline specific customer conversation points, discount limits, or service check instructions..."
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none resize-none"
            />
          </div>

          {/* Footer Buttons */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-300 px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="rounded-lg bg-teal-600 px-5 py-2 text-xs font-semibold text-white shadow-sm hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 transition-all disabled:opacity-50"
            >
              {isSubmitting ? 'Creating...' : 'Create Action Task'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
