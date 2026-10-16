import React from 'react';
import {
  LayoutDashboard,
  Users,
  AlertTriangle,
  ClipboardList,
  BarChart3,
  Cpu,
  Settings,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Radio,
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'customers'
  | 'at-risk'
  | 'retention'
  | 'analytics'
  | 'performance'
  | 'settings';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  openTasksCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  collapsed,
  onToggleCollapse,
  openTasksCount = 0,
}) => {
  const navItems = [
    { id: 'overview', label: 'Executive Overview', icon: LayoutDashboard },
    { id: 'customers', label: 'Customer Directory', icon: Users },
    { id: 'at-risk', label: 'At-Risk Accounts', icon: AlertTriangle, badge: 'Priority' },
    { id: 'retention', label: 'Retention Actions', icon: ClipboardList, count: openTasksCount },
    { id: 'analytics', label: 'Analytics & Insights', icon: BarChart3 },
    { id: 'performance', label: 'Model Performance', icon: Cpu },
    { id: 'settings', label: 'Settings & Status', icon: Settings },
  ];

  return (
    <aside
      className={`relative flex flex-col bg-slate-900 border-r border-slate-800 text-slate-300 transition-all duration-200 select-none ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between px-4 py-5 border-b border-slate-800">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-md shadow-teal-500/20">
            <Radio className="h-5 w-5" />
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="text-base font-bold text-white tracking-tight flex items-center gap-1.5">
                StayWise
                <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                  SaaS
                </span>
              </span>
              <span className="text-xs text-slate-400 font-normal">Retention Intelligence</span>
            </div>
          )}
        </div>

        {/* Collapse toggle */}
        <button
          onClick={onToggleCollapse}
          className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 space-y-1.5 px-3 py-4 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id as NavTab)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group relative ${
                isActive
                  ? 'bg-teal-600/15 text-teal-400 font-semibold border border-teal-500/30'
                  : 'text-slate-400 hover:bg-slate-800/80 hover:text-slate-200'
              }`}
              title={collapsed ? item.label : undefined}
            >
              <Icon
                className={`h-5 w-5 shrink-0 transition-colors ${
                  isActive ? 'text-teal-400' : 'text-slate-400 group-hover:text-slate-200'
                }`}
              />

              {!collapsed && <span className="flex-1 text-left truncate">{item.label}</span>}

              {!collapsed && item.badge && (
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">
                  {item.badge}
                </span>
              )}

              {!collapsed && item.count !== undefined && item.count > 0 && (
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-teal-400 border border-slate-700">
                  {item.count}
                </span>
              )}

              {/* Tooltip on collapse */}
              {collapsed && (
                <div className="absolute left-full ml-2 hidden rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-white shadow-lg group-hover:block z-50 whitespace-nowrap">
                  {item.label}
                </div>
              )}
            </button>
          );
        })}
      </nav>

      {/* System Status Footer */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40">
        {!collapsed ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Pipeline Active
              </span>
              <span className="font-mono text-[11px] text-slate-500">v2.0</span>
            </div>
            <div className="text-[11px] text-slate-400">
              <span className="text-slate-300 font-medium">Model:</span> Logistic Regression
            </div>
            <div className="text-[10px] text-slate-500 leading-tight">
              Enterprise Local Sandbox • SQLite Persistent
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title="Model Pipeline Active">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          </div>
        )}
      </div>
    </aside>
  );
};
