import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type Database = {
  public: {
    Tables: {
      projects: {
        Row: {
          id: string;
          competition_name: string;
          competition_url: string | null;
          status: string;
          progress: number;
          current_step: string;
          options: any;
          files: any | null;
          error_message: string | null;
          estimated_completion: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          competition_name: string;
          competition_url?: string | null;
          status?: string;
          progress?: number;
          current_step?: string;
          options: any;
          files?: any | null;
          error_message?: string | null;
          estimated_completion?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          competition_name?: string;
          competition_url?: string | null;
          status?: string;
          progress?: number;
          current_step?: string;
          options?: any;
          files?: any | null;
          error_message?: string | null;
          estimated_completion?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      project_files: {
        Row: {
          id: string;
          project_id: string;
          file_path: string;
          file_content: string;
          file_size: number;
          created_at: string;
        };
        Insert: {
          id?: string;
          project_id: string;
          file_path: string;
          file_content: string;
          file_size?: number;
          created_at?: string;
        };
        Update: {
          id?: string;
          project_id?: string;
          file_path?: string;
          file_content?: string;
          file_size?: number;
          created_at?: string;
        };
      };
    };
  };
};