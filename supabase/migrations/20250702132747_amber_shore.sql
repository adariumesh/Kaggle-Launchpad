/*
  # Create projects and project files tables

  1. New Tables
    - `projects`
      - `id` (text, primary key) - Unique project identifier
      - `competition_name` (text) - Name of the Kaggle competition
      - `competition_url` (text, optional) - Full URL to the competition
      - `status` (text) - Current status: queued, running, completed, error
      - `progress` (integer) - Progress percentage (0-100)
      - `current_step` (text) - Current generation step description
      - `options` (jsonb) - Generation options (EDA, baseline, git)
      - `files` (jsonb, optional) - Generated project files
      - `error_message` (text, optional) - Error message if generation failed
      - `estimated_completion` (timestamptz, optional) - Estimated completion time
      - `created_at` (timestamptz) - When project was created
      - `updated_at` (timestamptz) - When project was last updated

    - `project_files`
      - `id` (uuid, primary key) - Unique file identifier
      - `project_id` (text) - Reference to projects table
      - `file_path` (text) - Path of the file within the project
      - `file_content` (text) - Content of the file
      - `file_size` (integer) - Size of the file in bytes
      - `created_at` (timestamptz) - When file was created

  2. Security
    - Enable RLS on both tables
    - Add policies for public access (since we don't have user auth yet)
*/

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
  id text PRIMARY KEY,
  competition_name text NOT NULL,
  competition_url text,
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'error')),
  progress integer NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  current_step text NOT NULL DEFAULT 'Initializing...',
  options jsonb NOT NULL DEFAULT '{}',
  files jsonb,
  error_message text,
  estimated_completion timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create project_files table
CREATE TABLE IF NOT EXISTS project_files (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  file_path text NOT NULL,
  file_content text NOT NULL,
  file_size integer NOT NULL DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_project_files_project_id ON project_files(project_id);

-- Enable Row Level Security
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_files ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (no authentication required for now)
CREATE POLICY "Allow public read access to projects"
  ON projects
  FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert access to projects"
  ON projects
  FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Allow public update access to projects"
  ON projects
  FOR UPDATE
  TO public
  USING (true);

CREATE POLICY "Allow public read access to project_files"
  ON project_files
  FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert access to project_files"
  ON project_files
  FOR INSERT
  TO public
  WITH CHECK (true);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON projects
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();