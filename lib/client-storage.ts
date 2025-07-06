// Client-side storage utilities for project data
import { WorkflowState } from './project-workflow';

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
    // Ultra-advanced options
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
    // Workflow-specific options
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

const STORAGE_KEY = 'kaggle_launchpad_projects';

export class ClientStorage {
  static saveProject(project: ProjectData): void {
    try {
      const projects = this.getAllProjects();
      const existingIndex = projects.findIndex(p => p.id === project.id);
      
      if (existingIndex >= 0) {
        projects[existingIndex] = project;
      } else {
        projects.unshift(project);
      }
      
      // Keep only last 50 projects
      const trimmedProjects = projects.slice(0, 50);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmedProjects));
    } catch (error) {
      console.error('Failed to save project:', error);
    }
  }

  static getProject(id: string): ProjectData | null {
    try {
      const projects = this.getAllProjects();
      return projects.find(p => p.id === id) || null;
    } catch (error) {
      console.error('Failed to get project:', error);
      return null;
    }
  }

  static getAllProjects(): ProjectData[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to get projects:', error);
      return [];
    }
  }

  static getRecentProjects(limit: number = 10): ProjectData[] {
    return this.getAllProjects().slice(0, limit);
  }

  static deleteProject(id: string): void {
    try {
      const projects = this.getAllProjects();
      const filtered = projects.filter(p => p.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    } catch (error) {
      console.error('Failed to delete project:', error);
    }
  }

  static clearAllProjects(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Failed to clear projects:', error);
    }
  }

  static getProjectsByStatus(status: WorkflowState): ProjectData[] {
    return this.getAllProjects().filter(p => p.status === status);
  }

  static getRunningProjects(): ProjectData[] {
    const runningStates = [
      WorkflowState.QUEUED,
      WorkflowState.INITIALIZING,
      WorkflowState.ANALYZING_COMPETITION,
      WorkflowState.GATHERING_PRACTICES,
      WorkflowState.GENERATING_CODE,
      WorkflowState.CREATING_STRUCTURE,
      WorkflowState.FINALIZING
    ];
    
    return this.getAllProjects().filter(p => runningStates.includes(p.status));
  }
}