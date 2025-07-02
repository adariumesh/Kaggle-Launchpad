'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { 
  Rocket, 
  Download, 
  GitBranch, 
  BarChart3, 
  Brain, 
  Play, 
  CheckCircle, 
  XCircle, 
  Clock,
  FolderTree,
  ExternalLink,
  Zap,
  Trophy,
  TrendingUp,
  History,
  Database
} from 'lucide-react';

interface ProjectGenerationOptions {
  includeEDA: boolean;
  includeBaseline: boolean;
  initializeGit: boolean;
}

interface GenerationStatus {
  status: 'idle' | 'queued' | 'running' | 'completed' | 'error';
  progress: number;
  currentStep: string;
  error?: string;
  projectId?: string;
  estimatedCompletion?: string;
}

interface RecentProject {
  id: string;
  competitionName: string;
  status: string;
  progress: number;
  createdAt: string;
  options: ProjectGenerationOptions;
}

interface ProjectStructure {
  name: string;
  type: 'file' | 'folder';
  children?: ProjectStructure[];
}

const popularCompetitions = [
  { name: 'Titanic', url: 'titanic', difficulty: 'Beginner', icon: '🚢' },
  { name: 'House Prices', url: 'house-prices-advanced-regression-techniques', difficulty: 'Intermediate', icon: '🏠' },
  { name: 'Digit Recognizer', url: 'digit-recognizer', difficulty: 'Beginner', icon: '🔢' },
  { name: 'Natural Language Processing', url: 'nlp-getting-started', difficulty: 'Intermediate', icon: '💬' },
];

const sampleProjectStructure: ProjectStructure = {
  name: 'kaggle-competition',
  type: 'folder',
  children: [
    {
      name: 'notebooks',
      type: 'folder',
      children: [
        { name: 'eda.ipynb', type: 'file' },
        { name: 'baseline.ipynb', type: 'file' }
      ]
    },
    {
      name: 'src',
      type: 'folder',
      children: [
        { name: 'data_preprocessing.py', type: 'file' },
        { name: 'model.py', type: 'file' },
        { name: 'utils.py', type: 'file' },
        { name: 'submission.py', type: 'file' }
      ]
    },
    { name: 'data', type: 'folder' },
    { name: 'submissions', type: 'folder' },
    { name: 'models', type: 'folder' },
    { name: 'README.md', type: 'file' },
    { name: 'requirements.txt', type: 'file' },
    { name: '.gitignore', type: 'file' }
  ]
};

function FileTreeItem({ item, level = 0 }: { item: ProjectStructure; level?: number }) {
  const paddingLeft = level * 20 + 8;
  
  return (
    <div>
      <div 
        className="flex items-center py-1 text-sm hover:bg-slate-50 rounded transition-colors"
        style={{ paddingLeft }}
      >
        {item.type === 'folder' ? (
          <FolderTree className="w-4 h-4 mr-2 text-indigo-500" />
        ) : (
          <div className="w-2 h-2 mr-4 bg-slate-400 rounded-full" />
        )}
        <span className={item.type === 'folder' ? 'font-medium text-slate-700' : 'text-slate-600'}>
          {item.name}
        </span>
      </div>
      {item.children?.map((child, index) => (
        <FileTreeItem key={index} item={child} level={level + 1} />
      ))}
    </div>
  );
}

export default function Home() {
  const [competitionInput, setCompetitionInput] = useState('');
  const [options, setOptions] = useState<ProjectGenerationOptions>({
    includeEDA: true,
    includeBaseline: true,
    initializeGit: false,
  });
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>({
    status: 'idle',
    progress: 0,
    currentStep: 'Ready to generate',
  });
  const [recentProjects, setRecentProjects] = useState<RecentProject[]>([]);

  // Load recent projects on mount
  useEffect(() => {
    loadRecentProjects();
  }, []);

  const loadRecentProjects = async () => {
    try {
      const response = await fetch('/api/recent-projects?limit=5');
      if (response.ok) {
        const projects = await response.json();
        setRecentProjects(projects);
      }
    } catch (error) {
      console.error('Failed to load recent projects:', error);
    }
  };

  // Poll for status updates
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (generationStatus.projectId && 
        (generationStatus.status === 'queued' || generationStatus.status === 'running')) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`/api/get-status?projectId=${generationStatus.projectId}`);
          const data = await response.json();
          
          if (response.ok) {
            setGenerationStatus(prev => ({
              ...prev,
              status: data.status,
              progress: data.progress,
              currentStep: data.currentStep,
              error: data.error,
              estimatedCompletion: data.estimatedCompletion
            }));
            
            if (data.status === 'completed') {
              toast.success('Your Kaggle project is ready for download!');
              loadRecentProjects(); // Refresh recent projects
            } else if (data.status === 'error') {
              toast.error(data.error || 'Project generation failed');
            }
          }
        } catch (error) {
          console.error('Status polling error:', error);
        }
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [generationStatus.projectId, generationStatus.status]);

  const validateInput = (input: string): boolean => {
    if (!input.trim()) return false;
    // Check if it's a URL or competition name
    const urlPattern = /^https?:\/\/.*kaggle\.com\/competitions\/([^\/\?]+)/;
    const namePattern = /^[a-zA-Z0-9-_]+$/;
    return urlPattern.test(input) || namePattern.test(input);
  };

  const handleGenerate = async () => {
    if (!validateInput(competitionInput)) {
      toast.error('Please enter a valid competition name (letters, numbers, hyphens) or Kaggle competition URL');
      return;
    }

    setGenerationStatus({ 
      status: 'queued', 
      progress: 10, 
      currentStep: 'Initializing project generation...' 
    });
    
    try {
      const response = await fetch('/api/init-project', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          competition: competitionInput,
          options: options
        })
      });

      const data = await response.json();

      if (response.ok) {
        setGenerationStatus(prev => ({
          ...prev,
          projectId: data.projectId,
          status: data.status,
          progress: data.progress,
          currentStep: data.currentStep
        }));
        toast.success('Project generation started!');
      } else {
        throw new Error(data.error || 'Failed to start project generation');
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate project';
      setGenerationStatus({ 
        status: 'error', 
        progress: 0, 
        currentStep: 'Generation failed',
        error: errorMessage
      });
      toast.error(errorMessage);
    }
  };

  const handleDownload = async () => {
    if (!generationStatus.projectId) {
      toast.error('No project available for download');
      return;
    }

    try {
      const response = await fetch(`/api/download-project?projectId=${generationStatus.projectId}`);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `kaggle-project-${generationStatus.projectId}.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        toast.success('Project ZIP file downloaded!');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Download failed');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Download failed';
      toast.error(errorMessage);
    }
  };

  const selectCompetition = (competition: string) => {
    setCompetitionInput(competition);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-100 text-green-800';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'Advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = () => {
    switch (generationStatus.status) {
      case 'queued': return <Clock className="w-5 h-5 text-yellow-500 animate-pulse" />;
      case 'running': return <Zap className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'completed': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error': return <XCircle className="w-5 h-5 text-red-500" />;
      default: return <Rocket className="w-5 h-5 text-slate-400" />;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-indigo-500 rounded-xl flex items-center justify-center">
                <Rocket className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Kaggle Launchpad</h1>
                <p className="text-sm text-slate-600">AI-powered competition project generator</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary" className="bg-green-100 text-green-700">
                <Database className="w-3 h-3 mr-1" />
                Supabase Powered
              </Badge>
              <Badge variant="secondary" className="bg-indigo-100 text-indigo-700">
                <Trophy className="w-3 h-3 mr-1" />
                Production Ready
              </Badge>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Input Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Input Card */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Play className="w-5 h-5 text-indigo-500" />
                  <span>Generate Your Project</span>
                </CardTitle>
                <CardDescription>
                  Enter a Kaggle competition name or URL to generate a complete project scaffold
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <label htmlFor="competition" className="text-sm font-medium text-slate-700">
                    Competition Name or URL
                  </label>
                  <Input
                    id="competition"
                    placeholder="Example: titanic or https://www.kaggle.com/competitions/titanic"
                    value={competitionInput}
                    onChange={(e) => setCompetitionInput(e.target.value)}
                    className="h-12 text-base"
                  />
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-slate-700">Generation Options</h4>
                  <div className="grid sm:grid-cols-3 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="eda"
                        checked={options.includeEDA}
                        onCheckedChange={(checked) => 
                          setOptions(prev => ({ ...prev, includeEDA: checked as boolean }))
                        }
                      />
                      <label htmlFor="eda" className="text-sm text-slate-700 flex items-center space-x-1">
                        <BarChart3 className="w-4 h-4" />
                        <span>Include EDA</span>
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="baseline"
                        checked={options.includeBaseline}
                        onCheckedChange={(checked) => 
                          setOptions(prev => ({ ...prev, includeBaseline: checked as boolean }))
                        }
                      />
                      <label htmlFor="baseline" className="text-sm text-slate-700 flex items-center space-x-1">
                        <Brain className="w-4 h-4" />
                        <span>Include Baseline Model</span>
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="git"
                        checked={options.initializeGit}
                        onCheckedChange={(checked) => 
                          setOptions(prev => ({ ...prev, initializeGit: checked as boolean }))
                        }
                      />
                      <label htmlFor="git" className="text-sm text-slate-700 flex items-center space-x-1">
                        <GitBranch className="w-4 h-4" />
                        <span>Initialize Git</span>
                      </label>
                    </div>
                  </div>
                </div>

                <Button 
                  onClick={handleGenerate}
                  disabled={generationStatus.status === 'running' || generationStatus.status === 'queued'}
                  className="w-full h-12 text-base bg-indigo-500 hover:bg-indigo-600 transition-all duration-200"
                >
                  {generationStatus.status === 'running' || generationStatus.status === 'queued' ? (
                    <>
                      <Zap className="w-4 h-4 mr-2 animate-spin" />
                      Generating Project...
                    </>
                  ) : (
                    <>
                      <Rocket className="w-4 h-4 mr-2" />
                      Generate Project
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Progress Card */}
            {generationStatus.status !== 'idle' && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    {getStatusIcon()}
                    <span>Generation Progress</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">{generationStatus.currentStep}</span>
                      <span className="text-slate-500">{generationStatus.progress}%</span>
                    </div>
                    <Progress 
                      value={generationStatus.progress} 
                      className="h-2"
                    />
                  </div>
                  {generationStatus.estimatedCompletion && generationStatus.status === 'running' && (
                    <div className="text-xs text-slate-500">
                      Estimated completion: {formatDate(generationStatus.estimatedCompletion)}
                    </div>
                  )}
                  {generationStatus.error && (
                    <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                      <p className="text-sm text-red-600">{generationStatus.error}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Results Card */}
            {generationStatus.status === 'completed' && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span>Project Generated Successfully!</span>
                  </CardTitle>
                  <CardDescription>
                    Your Kaggle competition project is ready for download
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                    <h4 className="text-sm font-medium text-slate-700 mb-3">Project Structure</h4>
                    <FileTreeItem item={sampleProjectStructure} />
                  </div>
                  <Button 
                    onClick={handleDownload}
                    className="w-full h-12 bg-green-500 hover:bg-green-600 transition-all duration-200"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Project ZIP
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Popular Competitions */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-indigo-500" />
                  <span>Popular Competitions</span>
                </CardTitle>
                <CardDescription>
                  Quick-start with these beginner-friendly competitions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {popularCompetitions.map((competition, index) => (
                  <div 
                    key={index}
                    onClick={() => selectCompetition(competition.url)}
                    className="p-3 rounded-lg border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50 cursor-pointer transition-all duration-200 group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-lg">{competition.icon}</span>
                        <div>
                          <p className="font-medium text-slate-800 group-hover:text-indigo-700">
                            {competition.name}
                          </p>
                          <Badge 
                            variant="secondary" 
                            className={`text-xs ${getDifficultyColor(competition.difficulty)}`}
                          >
                            {competition.difficulty}
                          </Badge>
                        </div>
                      </div>
                      <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-indigo-500" />
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Recent Projects */}
            {recentProjects.length > 0 && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <History className="w-5 h-5 text-indigo-500" />
                    <span>Recent Projects</span>
                  </CardTitle>
                  <CardDescription>
                    Your recently generated projects
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {recentProjects.map((project) => (
                    <div 
                      key={project.id}
                      className="p-3 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-slate-800 truncate">
                            {project.competitionName}
                          </p>
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge 
                              variant="outline" 
                              className={`text-xs ${
                                project.status === 'completed' ? 'bg-green-50 text-green-700' :
                                project.status === 'error' ? 'bg-red-50 text-red-700' :
                                'bg-yellow-50 text-yellow-700'
                              }`}
                            >
                              {project.status}
                            </Badge>
                            <span className="text-xs text-slate-500">
                              {formatDate(project.createdAt)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Generation Stats */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg">What You Get</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Ready-to-run notebooks</span>
                  <Badge variant="outline">EDA + Baseline</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Python scripts</span>
                  <Badge variant="outline">5+ files</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Project structure</span>
                  <Badge variant="outline">Production-ready</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Persistent storage</span>
                  <Badge variant="outline">Supabase DB</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t bg-white/80 backdrop-blur-sm mt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Rocket className="w-5 h-5 text-indigo-500" />
                <span className="font-semibold text-slate-900">Kaggle Launchpad</span>
              </div>
              <p className="text-sm text-slate-600">
                Accelerate your machine learning journey with AI-powered project generation and persistent storage.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 mb-4">Features</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li>• Real project generation</li>
                <li>• Supabase backend integration</li>
                <li>• Persistent project storage</li>
                <li>• Production-ready code</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 mb-4">Links</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li>
                  <a href="https://kaggle.com" className="hover:text-indigo-500 transition-colors flex items-center space-x-1">
                    <span>Kaggle</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </li>
                <li>
                  <a href="https://supabase.com" className="hover:text-indigo-500 transition-colors flex items-center space-x-1">
                    <span>Supabase</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </li>
                <li><a href="#" className="hover:text-indigo-500 transition-colors">Support</a></li>
              </ul>
            </div>
          </div>
          <Separator className="my-6" />
          <div className="flex justify-between items-center text-sm text-slate-500">
            <p>&copy; 2025 Kaggle Launchpad. All rights reserved.</p>
            <p>Built with Next.js, Supabase & Tailwind CSS</p>
          </div>
        </div>
      </footer>
    </div>
  );
}