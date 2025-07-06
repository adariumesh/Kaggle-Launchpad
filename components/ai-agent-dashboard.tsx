'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Brain, 
  Globe, 
  TrendingUp, 
  BookOpen, 
  Zap, 
  Clock, 
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Play,
  Pause,
  Square
} from 'lucide-react';
import { supabase } from '@/lib/supabase-client';
import { WorkflowState, WorkflowFactory } from '@/lib/project-workflow';
import { ClientStorage, ProjectData } from '@/lib/client-storage';

interface AIAgentStatus {
  isActive: boolean;
  lastUpdate: Date;
  knowledgeBaseSize: number;
  learningProgress: number;
  currentTask: string;
  activeProjects: number;
  recentUpdates: Array<{
    type: string;
    description: string;
    timestamp: Date;
    status: 'completed' | 'in-progress' | 'failed';
  }>;
}

export function AIAgentDashboard() {
  const [agentStatus, setAgentStatus] = useState<AIAgentStatus>({
    isActive: true,
    lastUpdate: new Date(),
    knowledgeBaseSize: 15420,
    learningProgress: 78,
    currentTask: 'Monitoring active workflows',
    activeProjects: 0,
    recentUpdates: []
  });

  const [activeProjects, setActiveProjects] = useState<ProjectData[]>([]);
  const [recentProjects, setRecentProjects] = useState<ProjectData[]>([]);
  const [knowledgeDomains] = useState([
    { name: 'Tabular Data', coverage: 92, sources: 3420 },
    { name: 'Computer Vision', coverage: 88, sources: 4150 },
    { name: 'Natural Language Processing', coverage: 85, sources: 3890 },
    { name: 'Time Series', coverage: 79, sources: 2180 },
    { name: 'Multi-Modal', coverage: 71, sources: 1780 }
  ]);

  useEffect(() => {
    loadProjectData();
    const interval = setInterval(loadProjectData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadProjectData = async () => {
    try {
      // Load from client storage
      const allProjects = ClientStorage.getAllProjects();
      const runningProjects = ClientStorage.getRunningProjects();
      const recent = ClientStorage.getRecentProjects(10);

      setActiveProjects(runningProjects);
      setRecentProjects(recent);

      // Update agent status
      setAgentStatus(prev => ({
        ...prev,
        activeProjects: runningProjects.length,
        lastUpdate: new Date(),
        currentTask: runningProjects.length > 0 
          ? `Processing ${runningProjects.length} active project${runningProjects.length > 1 ? 's' : ''}`
          : 'Monitoring for new projects',
        recentUpdates: generateRecentUpdates(allProjects)
      }));

      // Also try to sync with Supabase
      await syncWithSupabase();
    } catch (error) {
      console.error('Failed to load project data:', error);
    }
  };

  const syncWithSupabase = async () => {
    try {
      const { data, error } = await supabase
        .from('projects')
        .select('*')
        .order('updated_at', { ascending: false })
        .limit(20);

      if (error) {
        console.warn('Failed to sync with Supabase:', error);
        return;
      }

      if (data) {
        // Update client storage with latest data from Supabase
        data.forEach(project => {
          const projectData: ProjectData = {
            id: project.id,
            competitionName: project.competition_name,
            status: project.status as WorkflowState,
            progress: project.progress,
            currentStep: project.current_step,
            options: project.options,
            files: project.files,
            error: project.error_message || undefined,
            createdAt: project.created_at,
            estimatedCompletion: project.estimated_completion || undefined
          };
          
          ClientStorage.saveProject(projectData);
        });
      }
    } catch (error) {
      console.warn('Supabase sync failed:', error);
    }
  };

  const generateRecentUpdates = (projects: ProjectData[]) => {
    const updates = [];
    
    // Add project-based updates
    projects.slice(0, 5).forEach(project => {
      let status: 'completed' | 'in-progress' | 'failed' = 'in-progress';
      let description = '';
      
      if (project.status === WorkflowState.COMPLETED) {
        status = 'completed';
        description = `Generated project for ${project.competitionName}`;
      } else if (project.status === WorkflowState.FAILED) {
        status = 'failed';
        description = `Failed to generate project for ${project.competitionName}`;
      } else if (project.status !== WorkflowState.QUEUED) {
        status = 'in-progress';
        description = `Processing ${project.competitionName}: ${project.currentStep}`;
      }
      
      if (description) {
        updates.push({
          type: 'Project Generation',
          description,
          timestamp: new Date(project.createdAt),
          status
        });
      }
    });

    // Add some mock knowledge updates
    updates.push(
      {
        type: 'Research Papers',
        description: 'Scanned 45 new arXiv papers on transformer architectures',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
        status: 'completed' as const
      },
      {
        type: 'Competition Analysis',
        description: 'Analyzed winning solutions from 3 recent tabular competitions',
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
        status: 'completed' as const
      }
    );

    return updates.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 10);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'in-progress':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getWorkflowStatusColor = (status: WorkflowState) => {
    switch (status) {
      case WorkflowState.COMPLETED:
        return 'bg-green-500';
      case WorkflowState.FAILED:
        return 'bg-red-500';
      case WorkflowState.CANCELLED:
        return 'bg-gray-500';
      default:
        return 'bg-blue-500';
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`;
    } else if (diffInMinutes < 1440) {
      return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
      return `${Math.floor(diffInMinutes / 1440)}d ago`;
    }
  };

  const handleCancelProject = async (projectId: string) => {
    try {
      const workflow = WorkflowFactory.getWorkflow(projectId);
      if (workflow) {
        await workflow.cancel();
        WorkflowFactory.removeWorkflow(projectId);
      }
      
      // Update in Supabase
      await supabase
        .from('projects')
        .update({ 
          status: WorkflowState.CANCELLED,
          current_step: 'Project cancelled by user',
          updated_at: new Date().toISOString()
        })
        .eq('id', projectId);
      
      // Reload data
      await loadProjectData();
    } catch (error) {
      console.error('Failed to cancel project:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Agent Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Agent Status</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <div className={`h-2 w-2 rounded-full ${agentStatus.isActive ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-2xl font-bold">
                {agentStatus.isActive ? 'Active' : 'Inactive'}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Last update: {formatTimeAgo(agentStatus.lastUpdate)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Projects</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{agentStatus.activeProjects}</div>
            <p className="text-xs text-muted-foreground">
              Currently processing
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Knowledge Base</CardTitle>
            <BookOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{agentStatus.knowledgeBaseSize.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Sources indexed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Learning Progress</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{agentStatus.learningProgress}%</div>
            <Progress value={agentStatus.learningProgress} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="workflows" className="space-y-4">
        <TabsList>
          <TabsTrigger value="workflows">Active Workflows</TabsTrigger>
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
          <TabsTrigger value="knowledge">Knowledge Domains</TabsTrigger>
          <TabsTrigger value="settings">Agent Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="workflows" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Project Workflows</CardTitle>
              <CardDescription>
                Real-time monitoring of project generation workflows
              </CardDescription>
            </CardHeader>
            <CardContent>
              {activeProjects.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Zap className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No active workflows</p>
                  <p className="text-sm">Start a new project to see workflows here</p>
                </div>
              ) : (
                <ScrollArea className="h-[400px]">
                  <div className="space-y-4">
                    {activeProjects.map((project) => (
                      <div key={project.id} className="flex items-start space-x-3 p-4 rounded-lg border">
                        <div className={`h-3 w-3 rounded-full mt-1 ${getWorkflowStatusColor(project.status)}`} />
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium">{project.competitionName}</h4>
                            <div className="flex items-center space-x-2">
                              <Badge variant="outline">{project.status}</Badge>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleCancelProject(project.id)}
                                disabled={project.status === WorkflowState.COMPLETED || project.status === WorkflowState.FAILED}
                              >
                                <Square className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground">{project.currentStep}</p>
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span>Progress</span>
                              <span>{project.progress}%</span>
                            </div>
                            <Progress value={project.progress} className="h-2" />
                          </div>
                          {project.estimatedCompletion && (
                            <p className="text-xs text-muted-foreground">
                              ETA: {new Date(project.estimatedCompletion).toLocaleTimeString()}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Learning Activities</CardTitle>
              <CardDescription>
                Latest updates and learning activities performed by the AI agent
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                <div className="space-y-4">
                  {agentStatus.recentUpdates.map((update, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 rounded-lg border">
                      {getStatusIcon(update.status)}
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{update.type}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {formatTimeAgo(update.timestamp)}
                          </span>
                        </div>
                        <p className="text-sm">{update.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="knowledge" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Knowledge Domain Coverage</CardTitle>
              <CardDescription>
                Coverage and source count for different ML domains
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {knowledgeDomains.map((domain, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{domain.name}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-muted-foreground">
                          {domain.sources.toLocaleString()} sources
                        </span>
                        <span className="text-sm font-medium">{domain.coverage}%</span>
                      </div>
                    </div>
                    <Progress value={domain.coverage} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Agent Configuration</CardTitle>
              <CardDescription>
                Configure the AI agent's learning behavior and data sources
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Update Frequency</label>
                  <select className="w-full p-2 border rounded-md">
                    <option>Every 6 hours</option>
                    <option>Every 12 hours</option>
                    <option>Daily</option>
                    <option>Weekly</option>
                  </select>
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium">Learning Depth</label>
                  <select className="w-full p-2 border rounded-md">
                    <option>Surface (Fast)</option>
                    <option>Moderate (Balanced)</option>
                    <option>Deep (Comprehensive)</option>
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Data Sources</label>
                <div className="grid grid-cols-2 gap-2">
                  {['arXiv Papers', 'Kaggle Discussions', 'GitHub Repositories', 'ML Blogs', 'Conference Proceedings', 'Documentation'].map((source) => (
                    <label key={source} className="flex items-center space-x-2">
                      <input type="checkbox" defaultChecked className="rounded" />
                      <span className="text-sm">{source}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="flex space-x-2">
                <Button>Save Configuration</Button>
                <Button variant="outline">Force Update</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}