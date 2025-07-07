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
  Square,
  Download
} from 'lucide-react';
import { WorkflowState, ProjectData, ClientStorage } from '@/lib/client-storage';
import { apiClient, AgentStatus } from '@/lib/api-client';
import { toast } from 'sonner';

export function AIAgentDashboard() {
  const [agentStatus, setAgentStatus] = useState<AgentStatus>({
    isActive: true,
    lastUpdate: new Date().toISOString(),
    knowledgeBaseSize: 15420,
    learningProgress: 78,
    currentTask: 'Monitoring active workflows',
    activeProjects: 0,
    recentUpdates: []
  });

  const [activeProjects, setActiveProjects] = useState<ProjectData[]>([]);
  const [recentProjects, setRecentProjects] = useState<ProjectData[]>([]);
  const [knowledgeDomains, setKnowledgeDomains] = useState([
    { name: 'Tabular Data', coverage: 92, sources: 3420 },
    { name: 'Computer Vision', coverage: 88, sources: 4150 },
    { name: 'Natural Language Processing', coverage: 85, sources: 3890 },
    { name: 'Time Series', coverage: 79, sources: 2180 },
    { name: 'Multi-Modal', coverage: 71, sources: 1780 }
  ]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadProjectData();
    const interval = setInterval(loadProjectData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadProjectData = async () => {
    try {
      setIsLoading(true);
      
      // Load data from backend API
      const [allProjects, agentStatusData, knowledgeDomainsData] = await Promise.all([
        apiClient.getAllProjects().catch(() => []),
        apiClient.getAgentStatus().catch(() => agentStatus),
        apiClient.getKnowledgeDomains().catch(() => knowledgeDomains)
      ]);

      // Cache projects locally for offline access
      ClientStorage.cacheProjects(allProjects);

      // Filter active and recent projects
      const runningProjects = allProjects.filter(project => {
        const runningStates = [
          WorkflowState.QUEUED,
          WorkflowState.INITIALIZING,
          WorkflowState.ANALYZING_COMPETITION,
          WorkflowState.GATHERING_PRACTICES,
          WorkflowState.GENERATING_CODE,
          WorkflowState.CREATING_STRUCTURE,
          WorkflowState.FINALIZING
        ];
        return runningStates.includes(project.status);
      });

      const recent = allProjects
        .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
        .slice(0, 10);

      setActiveProjects(runningProjects);
      setRecentProjects(recent);
      setAgentStatus(agentStatusData);
      setKnowledgeDomains(knowledgeDomainsData);

    } catch (error) {
      console.error('Failed to load project data:', error);
      
      // Fallback to local storage if API is unavailable
      const localProjects = ClientStorage.getAllProjects();
      const localRunning = ClientStorage.getRunningProjects();
      const localRecent = ClientStorage.getRecentProjects(10);
      
      setActiveProjects(localRunning);
      setRecentProjects(localRecent);
      
      toast.error('Failed to connect to backend. Showing cached data.');
    } finally {
      setIsLoading(false);
    }
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

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
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
      await apiClient.cancelProject(projectId);
      toast.success('Project cancelled successfully');
      
      // Reload data to reflect changes
      await loadProjectData();
    } catch (error) {
      console.error('Failed to cancel project:', error);
      toast.error('Failed to cancel project');
    }
  };

  const handleDownloadProject = async (projectId: string) => {
    try {
      const blob = await apiClient.downloadProjectFiles(projectId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `project-${projectId}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Project files downloaded successfully');
    } catch (error) {
      console.error('Failed to download project:', error);
      toast.error('Failed to download project files');
    }
  };

  const handleDownloadNotebook = async (projectId: string) => {
    try {
      const blob = await apiClient.downloadNotebook(projectId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `notebook-${projectId}.ipynb`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Notebook downloaded successfully');
    } catch (error) {
      console.error('Failed to download notebook:', error);
      toast.error('Failed to download notebook');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-2 text-lg">Loading dashboard...</span>
      </div>
    );
  }

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
                              {project.status === WorkflowState.COMPLETED && (
                                <>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => handleDownloadProject(project.id)}
                                  >
                                    <Download className="h-3 w-3 mr-1" />
                                    Files
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => handleDownloadNotebook(project.id)}
                                  >
                                    <Download className="h-3 w-3 mr-1" />
                                    Notebook
                                  </Button>
                                </>
                              )}
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