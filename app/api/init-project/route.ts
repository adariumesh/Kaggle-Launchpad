import { NextRequest, NextResponse } from 'next/server';
import { ProjectGenerator } from '@/lib/project-generator';
import { ProjectService } from '@/lib/supabase';

interface ProjectGenerationRequest {
  competition: string;
  options: {
    includeEDA: boolean;
    includeBaseline: boolean;
    initializeGit: boolean;
  };
}

export async function POST(request: NextRequest) {
  try {
    const body: ProjectGenerationRequest = await request.json();
    
    // Validate input
    if (!body.competition || !body.competition.trim()) {
      return NextResponse.json(
        { error: 'Competition name or URL is required' },
        { status: 400 }
      );
    }

    // Validate competition format
    const competition = body.competition.trim();
    const urlPattern = /^https?:\/\/.*kaggle\.com\/competitions\/([^\/\?]+)/;
    const namePattern = /^[a-zA-Z0-9-_]+$/;
    
    if (!urlPattern.test(competition) && !namePattern.test(competition)) {
      return NextResponse.json(
        { error: 'Please enter a valid competition name (letters, numbers, hyphens) or Kaggle competition URL' },
        { status: 400 }
      );
    }

    // Extract competition URL if provided
    const competitionUrl = urlPattern.test(competition) ? competition : undefined;
    const competitionName = urlPattern.test(competition) 
      ? competition.match(urlPattern)?.[1] || competition
      : competition;

    // Create project in database
    const project = await ProjectService.createProject({
      competition_name: competitionName,
      competition_url: competitionUrl,
      options: body.options,
    });
    
    // Start project generation in background
    generateProjectAsync(project.id, competitionName, body.options);
    
    const response = {
      projectId: project.id,
      status: project.status,
      progress: project.progress,
      currentStep: project.current_step,
      estimatedTime: '2-3 minutes',
      competition: competitionName,
      options: body.options,
      createdAt: project.created_at
    };

    return NextResponse.json(response, { status: 200 });
    
  } catch (error) {
    console.error('Project initialization error:', error);
    return NextResponse.json(
      { error: 'Failed to initialize project generation' },
      { status: 500 }
    );
  }
}

async function generateProjectAsync(
  projectId: string, 
  competition: string, 
  options: any
) {
  try {
    // Update status to running
    await ProjectService.updateProjectStatus(projectId, {
      status: 'running',
      progress: 20,
      current_step: 'Analyzing competition...',
    });

    // Simulate some processing time
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Update progress
    await ProjectService.updateProjectStatus(projectId, {
      progress: 50,
      current_step: 'Generating project structure...',
    });

    // Generate actual project files
    const files = await ProjectGenerator.generateProject(competition, options);

    await new Promise(resolve => setTimeout(resolve, 1500));

    // Update progress
    await ProjectService.updateProjectStatus(projectId, {
      progress: 80,
      current_step: 'Creating notebooks and scripts...',
    });

    await new Promise(resolve => setTimeout(resolve, 1000));

    // Save files to database
    await ProjectService.saveProjectFiles(projectId, files);

    // Mark as completed
    await ProjectService.updateProjectStatus(projectId, {
      status: 'completed',
      progress: 100,
      current_step: 'Project generated successfully!',
    });

  } catch (error) {
    console.error('Project generation error:', error);
    await ProjectService.updateProjectStatus(projectId, {
      status: 'error',
      progress: 0,
      current_step: 'Generation failed',
      error_message: error instanceof Error ? error.message : 'Failed to generate project. Please try again.',
    });
  }
}