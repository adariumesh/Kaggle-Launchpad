import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types for our database tables
export interface Project {
  id: string;
  competition_name: string;
  competition_url?: string;
  status: 'queued' | 'running' | 'completed' | 'error';
  progress: number;
  current_step: string;
  options: {
    includeEDA: boolean;
    includeBaseline: boolean;
    initializeGit: boolean;
  };
  files?: any[];
  error_message?: string;
  created_at: string;
  updated_at: string;
  estimated_completion?: string;
}

export interface ProjectFile {
  id: string;
  project_id: string;
  file_path: string;
  file_content: string;
  file_size: number;
  created_at: string;
}

// Database operations
export class ProjectService {
  static async createProject(data: {
    competition_name: string;
    competition_url?: string;
    options: Project['options'];
  }): Promise<Project> {
    const projectId = `project_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const { data: project, error } = await supabase
      .from('projects')
      .insert({
        id: projectId,
        competition_name: data.competition_name,
        competition_url: data.competition_url,
        status: 'queued',
        progress: 0,
        current_step: 'Initializing project generation...',
        options: data.options,
        estimated_completion: new Date(Date.now() + 3 * 60 * 1000).toISOString(), // 3 minutes from now
      })
      .select()
      .single();

    if (error) {
      throw new Error(`Failed to create project: ${error.message}`);
    }

    return project;
  }

  static async updateProjectStatus(
    projectId: string, 
    updates: Partial<Pick<Project, 'status' | 'progress' | 'current_step' | 'error_message'>>
  ): Promise<void> {
    const { error } = await supabase
      .from('projects')
      .update({
        ...updates,
        updated_at: new Date().toISOString(),
      })
      .eq('id', projectId);

    if (error) {
      throw new Error(`Failed to update project: ${error.message}`);
    }
  }

  static async getProject(projectId: string): Promise<Project | null> {
    const { data, error } = await supabase
      .from('projects')
      .select('*')
      .eq('id', projectId)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return null; // Project not found
      }
      throw new Error(`Failed to get project: ${error.message}`);
    }

    return data;
  }

  static async saveProjectFiles(projectId: string, files: any[]): Promise<void> {
    // First, save files data to the project record
    const { error: projectError } = await supabase
      .from('projects')
      .update({
        files: files,
        updated_at: new Date().toISOString(),
      })
      .eq('id', projectId);

    if (projectError) {
      throw new Error(`Failed to save project files: ${projectError.message}`);
    }

    // Also save individual files to project_files table for better querying
    const fileRecords = files.map(file => ({
      project_id: projectId,
      file_path: file.path,
      file_content: file.content,
      file_size: file.content.length,
    }));

    const { error: filesError } = await supabase
      .from('project_files')
      .insert(fileRecords);

    if (filesError) {
      console.warn('Failed to save individual file records:', filesError.message);
      // Don't throw here as the main files are saved in the project record
    }
  }

  static async getProjectFiles(projectId: string): Promise<any[]> {
    const project = await this.getProject(projectId);
    return project?.files || [];
  }

  static async getRecentProjects(limit: number = 10): Promise<Project[]> {
    const { data, error } = await supabase
      .from('projects')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      throw new Error(`Failed to get recent projects: ${error.message}`);
    }

    return data || [];
  }

  static async deleteProject(projectId: string): Promise<void> {
    // Delete project files first
    await supabase
      .from('project_files')
      .delete()
      .eq('project_id', projectId);

    // Delete project
    const { error } = await supabase
      .from('projects')
      .delete()
      .eq('id', projectId);

    if (error) {
      throw new Error(`Failed to delete project: ${error.message}`);
    }
  }
}