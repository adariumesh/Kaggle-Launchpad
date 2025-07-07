// API client for communicating with the backend service

export interface BackendConfig {
  baseUrl: string;
  apiKey?: string;
}

export interface CreateProjectRequest {
  competitionName: string;
  competitionUrl?: string;
  options: {
    includeEDA: boolean;
    includeBaseline: boolean;
    initializeGit: boolean;
    selectedModel: string;
    missingValueStrategy: 'median' | 'mean' | 'mode' | 'advanced';
    includeAdvancedFeatureEngineering: boolean;
    crossValidationFolds: number;
    hyperparameterTuning: boolean;
    ensembleMethod: 'none' | 'voting' | 'stacking' | 'blending';
    autoFeatureSelection: boolean;
    includeDeepLearning: boolean;
    dataAugmentation: boolean;
    outlierDetection: 'none' | 'isolation-forest' | 'local-outlier-factor' | 'one-class-svm';
    dimensionalityReduction: 'none' | 'pca' | 'tsne' | 'umap';
    advancedValidation: 'simple' | 'stratified' | 'time-series' | 'group' | 'adversarial';
    includeExplainability: boolean;
    optimizationObjective: 'accuracy' | 'speed' | 'memory' | 'interpretability';
    includeAutoML: boolean;
    generateDocumentation: boolean;
    includeUnitTests: boolean;
    codeOptimization: 'none' | 'basic' | 'advanced';
    useLatestPractices?: boolean;
    adaptToCompetitionType?: boolean;
    includeWinningTechniques?: boolean;
    optimizeForLeaderboard?: boolean;
  };
}

export class ApiClient {
  private config: BackendConfig;

  constructor(config: BackendConfig) {
    this.config = config;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} ${errorText}`);
    }

    return response.json();
  }

  // Project management endpoints
  async createProject(request: CreateProjectRequest): Promise<{ projectId: string }> {
    return this.makeRequest<{ projectId: string }>('/api/projects', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getProject(projectId: string): Promise<import('./client-storage').ProjectData> {
    return this.makeRequest<import('./client-storage').ProjectData>(`/api/projects/${projectId}`);
  }

  async cancelProject(projectId: string): Promise<void> {
    await this.makeRequest(`/api/projects/${projectId}/cancel`, {
      method: 'POST',
    });
  }

  // File download endpoints
  async downloadProjectFiles(projectId: string): Promise<Blob> {
    const response = await fetch(`${this.config.baseUrl}/api/projects/${projectId}/download`, {
      headers: this.config.apiKey ? {
        'Authorization': `Bearer ${this.config.apiKey}`
      } : {},
    });

    if (!response.ok) {
      throw new Error(`Failed to download project files: ${response.status}`);
    }

    return response.blob();
  }

  async downloadNotebook(projectId: string): Promise<Blob> {
    const response = await fetch(`${this.config.baseUrl}/api/projects/${projectId}/notebook`, {
      headers: this.config.apiKey ? {
        'Authorization': `Bearer ${this.config.apiKey}`
      } : {},
    });

    if (!response.ok) {
      throw new Error(`Failed to download notebook: ${response.status}`);
    }

    return response.blob();
  }
}

// Default API client instance
const defaultConfig: BackendConfig = {
  baseUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
  apiKey: process.env.NEXT_PUBLIC_API_KEY,
};

export const apiClient = new ApiClient(defaultConfig);