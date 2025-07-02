import { NextRequest, NextResponse } from 'next/server';
import { ProjectService } from '@/lib/supabase';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');
    
    if (!projectId) {
      return NextResponse.json(
        { error: 'Project ID is required' },
        { status: 400 }
      );
    }

    // Get project from Supabase
    const project = await ProjectService.getProject(projectId);
    
    if (!project) {
      return NextResponse.json(
        { error: 'Project not found' },
        { status: 404 }
      );
    }

    const response = {
      projectId: project.id,
      status: project.status,
      progress: project.progress,
      currentStep: project.current_step,
      error: project.error_message,
      createdAt: project.created_at,
      updatedAt: project.updated_at,
      estimatedCompletion: project.estimated_completion,
      competition: project.competition_name,
      options: project.options,
    };

    return NextResponse.json(response, { status: 200 });
    
  } catch (error) {
    console.error('Status check error:', error);
    return NextResponse.json(
      { error: 'Failed to get project status' },
      { status: 500 }
    );
  }
}