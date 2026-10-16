import React, { useState, useEffect } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Header } from './components/Header';
import { ExecutiveOverview } from './pages/ExecutiveOverview';
import { CustomerDirectory } from './pages/CustomerDirectory';
import { AtRiskAccounts } from './pages/AtRiskAccounts';
import { RetentionActions } from './pages/RetentionActions';
import { AnalyticsInsights } from './pages/AnalyticsInsights';
import { ModelPerformance } from './pages/ModelPerformance';
import { SystemSettings } from './pages/SystemSettings';
import { Customer360Drawer } from './components/Customer360Drawer';
import { CreateTaskModal } from './components/CreateTaskModal';
import { api } from './api/client';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);

  // Create Task Modal state
  const [isCreateTaskOpen, setIsCreateTaskOpen] = useState(false);
  const [createTaskCid, setCreateTaskCid] = useState('');
  const [createTaskType, setCreateTaskType] = useState('Contract Upgrade Offer');
  const [createTaskNotes, setCreateTaskNotes] = useState('');

  // Tasks tracker
  const [tasksVersion, setTasksVersion] = useState(0);
  const [openTasksCount, setOpenTasksCount] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchTaskStats = async () => {
    try {
      const tasks = await api.getRetentionTasks({ status: 'Open' });
      setOpenTasksCount(tasks.length);
    } catch (err) {
      console.error('Failed to update task badge:', err);
    }
  };

  useEffect(() => {
    fetchTaskStats();
  }, [tasksVersion]);

  const handleOpenCreateTask = (cid = '', taskType = 'Contract Upgrade Offer', notes = '') => {
    setCreateTaskCid(cid);
    setCreateTaskType(taskType);
    setCreateTaskNotes(notes);
    setIsCreateTaskOpen(true);
  };

  const handleTaskCreatedOrMutated = () => {
    setTasksVersion((v) => v + 1);
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    setTasksVersion((v) => v + 1);
    setTimeout(() => setIsRefreshing(false), 600);
  };

  const getPageMeta = () => {
    switch (currentTab) {
      case 'overview':
        return {
          title: 'Executive Overview',
          subtitle: 'Real-time retention KPIs & cohort health across 7,043 subscriber records',
        };
      case 'customers':
        return {
          title: 'Customer Directory',
          subtitle: 'Searchable subscriber portfolio scored with Scikit-learn ML pipeline',
        };
      case 'at-risk':
        return {
          title: 'Prioritized At-Risk Accounts',
          subtitle: 'Actionable triage queue of subscribers exhibiting highest churn hazard',
        };
      case 'retention':
        return {
          title: 'Retention Action Management',
          subtitle: 'Persistent task assignments, status tracking, and account intervention log',
        };
      case 'analytics':
        return {
          title: 'Analytics & Insights',
          subtitle: 'Empirical factor distributions, billing friction, and financial comparisons',
        };
      case 'performance':
        return {
          title: 'Model Performance & Benchmarks',
          subtitle: 'Holdout test set evaluation across Logistic Regression, KNN, Random Forest, and SVM',
        };
      case 'settings':
        return {
          title: 'Settings & Status',
          subtitle: 'Configurable probability triage thresholds and data pipeline health',
        };
    }
  };

  const meta = getPageMeta();

  return (
    <div className="flex h-screen bg-slate-50 font-sans overflow-hidden">
      {/* Left Sidebar */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        openTasksCount={openTasksCount}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <Header
          title={meta.title}
          subtitle={meta.subtitle}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          onOpenCreateTask={() => handleOpenCreateTask()}
        />

        <main className="flex-1 pb-16">
          {currentTab === 'overview' && (
            <ExecutiveOverview
              onNavigateToTab={setCurrentTab}
              onOpenCreateTask={handleOpenCreateTask}
            />
          )}

          {currentTab === 'customers' && (
            <CustomerDirectory
              onSelectCustomer={(cid) => setSelectedCustomerId(cid)}
              onOpenCreateTask={(cid) => handleOpenCreateTask(cid)}
            />
          )}

          {currentTab === 'at-risk' && (
            <AtRiskAccounts
              onSelectCustomer={(cid) => setSelectedCustomerId(cid)}
              onOpenCreateTask={(cid, tType) => handleOpenCreateTask(cid, tType)}
            />
          )}

          {currentTab === 'retention' && (
            <RetentionActions
              onSelectCustomer={(cid) => setSelectedCustomerId(cid)}
              onOpenCreateTask={(cid) => handleOpenCreateTask(cid)}
              tasksVersion={tasksVersion}
              onTaskMutated={handleTaskCreatedOrMutated}
            />
          )}

          {currentTab === 'analytics' && <AnalyticsInsights />}

          {currentTab === 'performance' && <ModelPerformance />}

          {currentTab === 'settings' && (
            <SystemSettings onThresholdsUpdated={handleTaskCreatedOrMutated} />
          )}
        </main>
      </div>

      {/* Customer 360° Drawer */}
      <Customer360Drawer
        customerId={selectedCustomerId}
        onClose={() => setSelectedCustomerId(null)}
        onOpenCreateTask={(cid, tType, notes) => {
          handleOpenCreateTask(cid, tType, notes);
        }}
      />

      {/* Create Task Modal */}
      <CreateTaskModal
        isOpen={isCreateTaskOpen}
        onClose={() => setIsCreateTaskOpen(false)}
        onTaskCreated={handleTaskCreatedOrMutated}
        initialCustomerId={createTaskCid}
        initialTaskType={createTaskType}
        initialNotes={createTaskNotes}
      />
    </div>
  );
};

export default App;
