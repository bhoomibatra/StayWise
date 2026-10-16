import {
  AnalyticsData,
  CustomerProfile,
  CustomerQueryResponse,
  ModelPerformanceData,
  OverviewData,
  RetentionNote,
  RetentionTask,
  ThresholdConfig,
} from '../types';

const API_BASE = '/api';

export const api = {
  async getOverview(): Promise<OverviewData> {
    const res = await fetch(`${API_BASE}/overview`);
    if (!res.ok) throw new Error(`Failed to load overview: ${res.statusText}`);
    return res.json();
  },

  async getCustomers(params: {
    search?: string;
    predicted_churn?: string;
    risk_category?: string;
    contract?: string;
    sort_by?: string;
    sort_order?: string;
    page?: number;
    page_size?: number;
  }): Promise<CustomerQueryResponse> {
    const q = new URLSearchParams();
    if (params.search) q.set('search', params.search);
    if (params.predicted_churn && params.predicted_churn !== 'All') q.set('predicted_churn', params.predicted_churn);
    if (params.risk_category && params.risk_category !== 'All') q.set('risk_category', params.risk_category);
    if (params.contract && params.contract !== 'All') q.set('contract', params.contract);
    if (params.sort_by) q.set('sort_by', params.sort_by);
    if (params.sort_order) q.set('sort_order', params.sort_order);
    if (params.page) q.set('page', params.page.toString());
    if (params.page_size) q.set('page_size', params.page_size.toString());

    const res = await fetch(`${API_BASE}/customers?${q.toString()}`);
    if (!res.ok) throw new Error(`Failed to load customers: ${res.statusText}`);
    return res.json();
  },

  getExportUrl(params: {
    search?: string;
    predicted_churn?: string;
    risk_category?: string;
    contract?: string;
  }): string {
    const q = new URLSearchParams();
    if (params.search) q.set('search', params.search);
    if (params.predicted_churn && params.predicted_churn !== 'All') q.set('predicted_churn', params.predicted_churn);
    if (params.risk_category && params.risk_category !== 'All') q.set('risk_category', params.risk_category);
    if (params.contract && params.contract !== 'All') q.set('contract', params.contract);
    return `${API_BASE}/customers/export?${q.toString()}`;
  },

  async getCustomerProfile(customerId: string): Promise<CustomerProfile> {
    const res = await fetch(`${API_BASE}/customers/${customerId}`);
    if (!res.ok) throw new Error(`Customer ${customerId} not found`);
    return res.json();
  },

  async getAtRiskAccounts(params: {
    min_risk?: string;
    contract?: string;
    page?: number;
    page_size?: number;
  }): Promise<CustomerQueryResponse> {
    const q = new URLSearchParams();
    if (params.min_risk) q.set('min_risk', params.min_risk);
    if (params.contract && params.contract !== 'All') q.set('contract', params.contract);
    if (params.page) q.set('page', params.page.toString());
    if (params.page_size) q.set('page_size', params.page_size.toString());

    const res = await fetch(`${API_BASE}/at-risk?${q.toString()}`);
    if (!res.ok) throw new Error(`Failed to load at-risk accounts: ${res.statusText}`);
    return res.json();
  },

  async getRetentionTasks(params?: {
    status?: string;
    priority?: string;
    customer_id?: string;
  }): Promise<RetentionTask[]> {
    const q = new URLSearchParams();
    if (params?.status && params.status !== 'All') q.set('status', params.status);
    if (params?.priority && params.priority !== 'All') q.set('priority', params.priority);
    if (params?.customer_id) q.set('customer_id', params.customer_id);

    const res = await fetch(`${API_BASE}/retention/tasks?${q.toString()}`);
    if (!res.ok) throw new Error(`Failed to load retention tasks: ${res.statusText}`);
    return res.json();
  },

  async createRetentionTask(task: {
    customer_id: string;
    task_type: string;
    priority: string;
    assigned_to: string;
    notes: string;
    due_date?: string;
  }): Promise<RetentionTask> {
    const res = await fetch(`${API_BASE}/retention/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(task),
    });
    if (!res.ok) throw new Error(`Failed to create task: ${res.statusText}`);
    return res.json();
  },

  async updateRetentionTask(taskId: number, updates: {
    status?: string;
    priority?: string;
    assigned_to?: string;
    notes?: string;
    due_date?: string;
  }): Promise<RetentionTask> {
    const res = await fetch(`${API_BASE}/retention/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!res.ok) throw new Error(`Failed to update task: ${res.statusText}`);
    return res.json();
  },

  async deleteRetentionTask(taskId: number): Promise<void> {
    const res = await fetch(`${API_BASE}/retention/tasks/${taskId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error(`Failed to delete task: ${res.statusText}`);
  },

  async addCustomerNote(note: {
    customer_id: string;
    author: string;
    note: string;
  }): Promise<RetentionNote> {
    const res = await fetch(`${API_BASE}/retention/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(note),
    });
    if (!res.ok) throw new Error(`Failed to add note: ${res.statusText}`);
    return res.json();
  },

  async getAnalytics(): Promise<AnalyticsData> {
    const res = await fetch(`${API_BASE}/analytics`);
    if (!res.ok) throw new Error(`Failed to load analytics: ${res.statusText}`);
    return res.json();
  },

  async getModelPerformance(): Promise<ModelPerformanceData> {
    const res = await fetch(`${API_BASE}/model-performance`);
    if (!res.ok) throw new Error(`Failed to load model performance: ${res.statusText}`);
    return res.json();
  },

  async getThresholds(): Promise<ThresholdConfig> {
    const res = await fetch(`${API_BASE}/settings/thresholds`);
    if (!res.ok) throw new Error(`Failed to load thresholds: ${res.statusText}`);
    return res.json();
  },

  async updateThresholds(config: ThresholdConfig): Promise<ThresholdConfig> {
    const res = await fetch(`${API_BASE}/settings/thresholds`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(`Failed to update thresholds: ${res.statusText}`);
    return res.json();
  },
};
