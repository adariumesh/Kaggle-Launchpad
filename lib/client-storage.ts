// Client-side storage utilities for project data
// Simplified for single project workflow

export enum WorkflowState {
  QUEUED = 'queued',
  INITIALIZING = 'initializing',
  ANALYZING_COMPETITION = 'analyzing_competition',
  GATHERING_PRACTICES = 'gathering_practices',
  GENERATING_CODE = 'generating_code',
  CREATING_STRUCTURE = 'creating_structure',
  FINALIZING = 'finalizing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface ProjectData {
  id: string;
  competitionName: string;
  status: WorkflowState;
  progress: number;
  currentStep: string;
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
    competitionUrl?: string;
    useLatestPractices?: boolean;
    adaptToCompetitionType?: boolean;
    includeWinningTechniques?: boolean;
    optimizeForLeaderboard?: boolean;
  };
  files?: GeneratedFile[];
  notebook?: KaggleNotebook;
  error?: string;
  createdAt: string;
  estimatedCompletion?: string;
}

export interface GeneratedFile {
  path: string;
  content: string;
}

export interface KaggleNotebook {
  path: string;
  content: string;
  expectedScore?: string;
  description: string;
  complexity: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  estimatedRuntime: string;
  memoryUsage: string;
  techniques: string[];
}

const STORAGE_KEY = 'kaggle_launchpad_current_project';

export class ClientStorage {
  static saveProject(project: ProjectData): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(project));
    } catch (error) {
      console.error('Failed to save project:', error);
    }
  }

  static getProject(id: string): ProjectData | null {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const project = JSON.parse(stored);
        return project.id === id ? project : null;
      }
      return null;
    } catch (error) {
      console.error('Failed to get project:', error);
      return null;
    }
  }

  static getCurrentProject(): ProjectData | null {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.error('Failed to get current project:', error);
      return null;
    }
  }

  static deleteProject(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Failed to delete project:', error);
    }
  }

  static clearProject(): void {
    this.deleteProject();
  }
}