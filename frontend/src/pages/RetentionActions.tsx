import React, { useState, useEffect } from 'react';
import {
  ClipboardList,
  CheckCircle2,
  Clock,
  User,
  PlusCircle,
  Filter,
  Trash2,
  Calendar,
  AlertCircle,
  Check,
  Tag,
  FileText,
} from 'lucide-react';
import { api } from '../api/client';
import { RetentionTask, TaskStatus, TaskPriority } from '../types';
import { TaskStatusBadge, PriorityBadge } from '../components/Badges';

interface RetentionActionsProps {
  onSelectCustomer: (customerId: string) => void;
  onOpenCreateTask: (customerId?: string) => void;
  tasksVersion: number;
  onTaskMutated: () => void;
}

export const RetentionActions: React.FC<RetentionActionsProps> = ({
  onSelectCustomer,
  onOpenCreateTask,
  tasksVersion,
  onTaskMutated,
}) => {
  const [tasks, setTasks] = useState<RetentionTask[]>([]);
  const [statusFilter, setStatusFilter] = useState('All');
  const [priorityFilter, setPriorityFilter] = useState('All');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getRetentionTasks({
        status: statusFilter,
        priority: priorityFilter,
      });
      setTasks(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load retention tasks');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, [statusFilter, priorityFilter, tasksVersion]);

  const handleStatusChange = async (taskId: number, newStatus: TaskStatus) => {
    try {
      await api.updateRetentionTask(taskId, { status: newStatus });
      onTaskMutated();
      fetchTasks();
    } catch (err: any) {
      alert(`Failed to update status: ${err.message}`);
    }
  };

  const handleDeleteTask = async (taskId: number) => {
    if (!confirm('Are you sure you want to delete this retention task?')) return;
    try {
      await api.deleteRetentionTask(taskId);
      onTaskMutated();
      fetchTasks();
    } catch (err: any) {
      alert(`Failed to delete task: ${err.message}`);
    }
  };

  // Group counts
  const openCount = tasks.filter((t) => t.status === 'Open').length;
  const inProgressCount = tasks.filter((t) => t.status === 'In Progress').length;
  const completedCount = tasks.filter((t) => t.status === 'Completed').length;

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner and Quick Stats */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <ClipboardList className="h-5 w-5 text-teal-600" />
            Retention Workflow & Follow-Up Management
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Track customer interventions, log team assignments, and record status transitions.
            All tasks are locally persisted in SQLite and survive application restarts.
          </p>
        </div>

        <button
          onClick={() => onOpenCreateTask()}
          className="flex items-center gap-2 rounded-lg bg-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-teal-700 transition-all shrink-0"
        >
          <PlusCircle className="h-4 w-4" />
          Create New Task
        </button>
      </div>

      {/* Task Summary Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div
          onClick={() => setStatusFilter(statusFilter === 'Open' ? 'All' : 'Open')}
          className={`cursor-pointer rounded-xl border p-4 transition-all ${
            statusFilter === 'Open'
              ? 'border-amber-400 bg-amber-50/60 shadow-sm'
              : 'border-slate-200 bg-white hover:border-slate-300'
          }`}
        >
          <div className="flex items-center justify-between text-xs text-amber-700 font-semibold">
            <span>Open Tasks</span>
            <span className="h-2 w-2 rounded-full bg-amber-500" />
          </div>
          <span className="text-2xl font-extrabold text-slate-900 mt-2 block">{openCount}</span>
          <span className="text-[11px] text-slate-500">Awaiting customer outreach</span>
        </div>

        <div
          onClick={() => setStatusFilter(statusFilter === 'In Progress' ? 'All' : 'In Progress')}
          className={`cursor-pointer rounded-xl border p-4 transition-all ${
            statusFilter === 'In Progress'
              ? 'border-blue-400 bg-blue-50/60 shadow-sm'
              : 'border-slate-200 bg-white hover:border-slate-300'
          }`}
        >
          <div className="flex items-center justify-between text-xs text-blue-700 font-semibold">
            <span>In Progress</span>
            <span className="h-2 w-2 rounded-full bg-blue-500" />
          </div>
          <span className="text-2xl font-extrabold text-slate-900 mt-2 block">{inProgressCount}</span>
          <span className="text-[11px] text-slate-500">Actively in negotiation / review</span>
        </div>

        <div
          onClick={() => setStatusFilter(statusFilter === 'Completed' ? 'All' : 'Completed')}
          className={`cursor-pointer rounded-xl border p-4 transition-all ${
            statusFilter === 'Completed'
              ? 'border-emerald-400 bg-emerald-50/60 shadow-sm'
              : 'border-slate-200 bg-white hover:border-slate-300'
          }`}
        >
          <div className="flex items-center justify-between text-xs text-emerald-700 font-semibold">
            <span>Completed Actions</span>
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
          </div>
          <span className="text-2xl font-extrabold text-slate-900 mt-2 block">{completedCount}</span>
          <span className="text-[11px] text-slate-500">Successfully resolved or saved</span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold text-slate-600 flex items-center gap-1">
            <Filter className="h-3.5 w-3.5" />
            Filters:
          </span>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 bg-white focus:outline-none focus:border-teal-500"
          >
            <option value="All">All Statuses</option>
            <option value="Open">Open</option>
            <option value="In Progress">In Progress</option>
            <option value="Completed">Completed</option>
          </select>

          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-700 bg-white focus:outline-none focus:border-teal-500"
          >
            <option value="All">All Priorities</option>
            <option value="High">High Priority</option>
            <option value="Medium">Medium Priority</option>
            <option value="Low">Low Priority</option>
          </select>
        </div>

        <span className="text-slate-500">
          Showing <strong className="text-slate-900">{tasks.length}</strong> tasks
        </span>
      </div>

      {/* Task List */}
      <div className="space-y-3">
        {loading ? (
          <div className="py-20 text-center text-slate-400 flex flex-col items-center justify-center gap-2">
            <div className="h-7 w-7 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
            <span className="text-xs font-medium">Loading retention tasks from SQLite...</span>
          </div>
        ) : tasks.length === 0 ? (
          <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-500">
            <CheckCircle2 className="h-10 w-10 text-slate-300 mx-auto mb-2" />
            <h3 className="font-bold text-slate-800">No retention tasks found</h3>
            <p className="text-xs text-slate-400 mt-1">
              Create your first follow-up action to begin tracking customer retention work.
            </p>
          </div>
        ) : (
          tasks.map((task) => (
            <div
              key={task.id}
              className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs hover:border-slate-300 transition-all flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4"
            >
              {/* Task Details */}
              <div className="space-y-2 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    onClick={() => onSelectCustomer(task.customer_id)}
                    className="font-mono text-sm font-bold text-slate-900 hover:text-teal-600 cursor-pointer underline-offset-2 hover:underline"
                  >
                    {task.customer_id}
                  </span>
                  <span className="text-slate-300">•</span>
                  <span className="text-sm font-semibold text-slate-800">{task.task_type}</span>
                  <PriorityBadge priority={task.priority} />
                  <TaskStatusBadge status={task.status} />
                </div>

                {task.notes && (
                  <p className="text-xs text-slate-600 leading-relaxed max-w-3xl bg-slate-50 rounded-lg p-2.5 border border-slate-100">
                    {task.notes}
                  </p>
                )}

                <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                  <span className="flex items-center gap-1">
                    <User className="h-3.5 w-3.5 text-slate-400" />
                    Owner: <strong className="text-slate-700">{task.assigned_to}</strong>
                  </span>
                  {task.due_date && (
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3.5 w-3.5 text-slate-400" />
                      Due: <strong className="text-slate-700">{task.due_date}</strong>
                    </span>
                  )}
                  <span className="flex items-center gap-1 text-[11px] text-slate-400">
                    <Clock className="h-3 w-3" />
                    Created: {new Date(task.created_at).toLocaleDateString()}
                  </span>
                  {task.completed_at && (
                    <span className="flex items-center gap-1 text-[11px] text-emerald-600 font-medium">
                      <Check className="h-3 w-3" />
                      Completed: {new Date(task.completed_at).toLocaleDateString()}
                    </span>
                  )}
                </div>
              </div>

              {/* Status Update Dropdown & Delete */}
              <div className="flex items-center gap-3 shrink-0 w-full lg:w-auto justify-end pt-3 lg:pt-0 border-t lg:border-t-0 border-slate-100">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-slate-400">Status:</span>
                  <select
                    value={task.status}
                    onChange={(e) => handleStatusChange(task.id, e.target.value as TaskStatus)}
                    className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-800 bg-white focus:outline-none focus:border-teal-500"
                  >
                    <option value="Open">Open</option>
                    <option value="In Progress">In Progress</option>
                    <option value="Completed">Completed</option>
                    <option value="Cancelled">Cancelled</option>
                  </select>
                </div>

                <button
                  onClick={() => handleDeleteTask(task.id)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                  title="Delete Task"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
