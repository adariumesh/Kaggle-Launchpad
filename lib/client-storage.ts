// Client-side storage utilities for project data
export interface ProjectData {
  id: string;
  competitionName: string;
  status: 'idle' | 'generating' | 'completed' | 'error';
  progress: number;
  currentStep: string;
  options: {
    includeEDA: boolean;
    includeBaseline: boolean;
    initializeGit: boolean;
  };
  files?: GeneratedFile[];
  error?: string;
  createdAt: string;
  estimatedCompletion?: string;
}

export interface GeneratedFile {
  path: string;
  content: string;
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
}