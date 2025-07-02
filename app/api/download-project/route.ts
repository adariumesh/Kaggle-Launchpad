import { NextRequest, NextResponse } from 'next/server';
import { ProjectService } from '@/lib/supabase';
import { ZipGenerator } from '@/lib/zip-generator';

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

    if (project.status !== 'completed') {
      return NextResponse.json(
        { error: 'Project generation not completed yet' },
        { status: 400 }
      );
    }

    if (!project.files || project.files.length === 0) {
      return NextResponse.json(
        { error: 'Project files not available' },
        { status: 500 }
      );
    }

    // Generate ZIP content
    const zipBlob = await ZipGenerator.createProjectZip(project.files, projectId);
    const zipBuffer = await zipBlob.arrayBuffer();
    
    const response = new NextResponse(zipBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': `attachment; filename="kaggle-project-${project.competition_name}-${projectId}.zip"`,
        'Content-Length': zipBuffer.byteLength.toString(),
      },
    });

    return response;
    
  } catch (error) {
    console.error('Download error:', error);
    return NextResponse.json(
      { error: 'Failed to download project' },
      { status: 500 }
    );
  }
}