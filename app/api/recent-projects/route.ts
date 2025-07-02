import { NextRequest, NextResponse } from 'next/server';
import { ProjectService } from '@/lib/supabase';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '10');
    
    // Get recent projects from Supabase
    const projects = await ProjectService.getRecentProjects(limit);
    
    const response = projects.map(project => ({
      id: project.id,
      competitionName: project.competition_name,
      status: project.status,
      progress: project.progress,
      createdAt: project.created_at,
      options: project.options,
    }));

    return NextResponse.json(response, { status: 200 });
    
  } catch (error) {
    console.error('Recent projects error:', error);
    return NextResponse.json(
      { error: 'Failed to get recent projects' },
      { status: 500 }
    );
  }
}