'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
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
  FileText,
  ExternalLink,
  Zap,
  Trophy,
  TrendingUp,
  History,
  Code,
  BookOpen,
  Target,
  Sparkles,
  Settings,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Sliders
} from 'lucide-react';
import { NotebookGenerator } from '@/lib/notebook-generator';
import { ClientStorage, ProjectData } from '@/lib/client-storage';

interface ProjectGenerationOptions {
  includeEDA: boolean;
  includeBaseline: boolean;
  initializeGit: boolean;
  selectedModel: string;
  missingValueStrategy: 'median' | 'mean' | 'mode' | 'advanced';
  includeAdvancedFeatureEngineering: boolean;
  crossValidationFolds: number;
  hyperparameterTuning: boolean;
}

interface GenerationStatus {
  status: 'idle' | 'generating' | 'completed' | 'error';
  progress: number;
  currentStep: string;
  error?: string;
  projectId?: string;
  estimatedCompletion?: string;
}

const popularCompetitions = [
  { name: 'Titanic', url: 'titanic', difficulty: 'Beginner', icon: '🚢', score: '0.77-0.82' },
  { name: 'House Prices', url: 'house-prices-advanced-regression-techniques', difficulty: 'Intermediate', icon: '🏠', score: '0.12-0.15 RMSE' },
  { name: 'Digit Recognizer', url: 'digit-recognizer', difficulty: 'Beginner', icon: '🔢', score: '0.95-0.98' },
  { name: 'Natural Language Processing', url: 'nlp-getting-started', difficulty: 'Intermediate', icon: '💬', score: '0.80-0.85' },
];

const getAvailableModels = (competitionType: string) => {
  const modelMap: Record<string, string[]> = {
    'classification': ['XGBoost', 'RandomForest', 'LogisticRegression', 'LightGBM', 'CatBoost', 'SVM'],
    'regression': ['XGBoost', 'RandomForest', 'LinearRegression', 'LightGBM', 'CatBoost', 'SVR'],
    'nlp': ['LogisticRegression', 'RandomForest', 'XGBoost', 'NaiveBayes', 'SVM'],
    'computer-vision': ['RandomForest', 'XGBoost', 'CNN', 'ResNet', 'EfficientNet'],
    'other': ['XGBoost', 'RandomForest', 'LogisticRegression', 'LightGBM']
  };
  return modelMap[competitionType] || modelMap['other'];
};

const detectCompetitionType = (name: string): string => {
  const lowerName = name.toLowerCase();
  
  if (lowerName.includes('nlp') || lowerName.includes('sentiment') || lowerName.includes('text')) {
    return 'nlp';
  }
  if (lowerName.includes('image') || lowerName.includes('vision') || lowerName.includes('digit')) {
    return 'computer-vision';
  }
  if (lowerName.includes('price') || lowerName.includes('sales') || lowerName.includes('revenue')) {
    return 'regression';
  }
  return 'classification';
};

export default function Home() {
  const [competitionInput, setCompetitionInput] = useState('');
  const [competitionType, setCompetitionType] = useState('classification');
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [options, setOptions] = useState<ProjectGenerationOptions>({
    includeEDA: true,
    includeBaseline: true,
    initializeGit: false,
    selectedModel: 'XGBoost',
    missingValueStrategy: 'median',
    includeAdvancedFeatureEngineering: false,
    crossValidationFolds: 5,
    hyperparameterTuning: false,
  });
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>({
    status: 'idle',
    progress: 0,
    currentStep: 'Ready to generate',
  });
  const [recentProjects, setRecentProjects] = useState<ProjectData[]>([]);
  const [currentProject, setCurrentProject] = useState<ProjectData | null>(null);

  // Load recent projects on mount
  useEffect(() => {
    loadRecentProjects();
  }, []);

  // Update available models when competition input changes
  useEffect(() => {
    if (competitionInput) {
      const detectedType = detectCompetitionType(competitionInput);
      setCompetitionType(detectedType);
      const availableModels = getAvailableModels(detectedType);
      if (!availableModels.includes(options.selectedModel)) {
        setOptions(prev => ({ ...prev, selectedModel: availableModels[0] }));
      }
    }
  }, [competitionInput, options.selectedModel]);

  const loadRecentProjects = () => {
    try {
      const projects = ClientStorage.getRecentProjects(5);
      setRecentProjects(projects);
    } catch (error) {
      console.error('Failed to load recent projects:', error);
    }
  };

  const validateInput = (input: string): boolean => {
    if (!input.trim()) return false;
    // Check if it's a URL or competition name
    const urlPattern = /^https?:\/\/.*kaggle\.com\/competitions\/([^\/\?]+)/;
    const namePattern = /^[a-zA-Z0-9-_]+$/;
    return urlPattern.test(input) || namePattern.test(input);
  };

  const simulateGeneration = async (projectId: string, competitionName: string, options: ProjectGenerationOptions) => {
    const steps = [
      { step: 'Analyzing competition type...', progress: 15, delay: 800 },
      { step: 'Configuring advanced model settings...', progress: 30, delay: 1000 },
      { step: 'Generating optimized Kaggle notebook...', progress: 60, delay: 1500 },
      { step: 'Adding advanced feature engineering...', progress: 80, delay: 1200 },
      { step: 'Finalizing model and evaluation code...', progress: 95, delay: 800 },
    ];

    try {
      for (const { step, progress, delay } of steps) {
        await new Promise(resolve => setTimeout(resolve, delay));
        
        setGenerationStatus(prev => ({
          ...prev,
          progress,
          currentStep: step,
        }));

        // Update stored project
        const project = ClientStorage.getProject(projectId);
        if (project) {
          project.progress = progress;
          project.currentStep = step;
          ClientStorage.saveProject(project);
        }
      }

      // Generate actual notebook
      const notebook = await NotebookGenerator.generateKaggleNotebook(competitionName, {
        includeEDA: options.includeEDA,
        includeBaseline: options.includeBaseline,
        selectedModel: options.selectedModel,
        missingValueStrategy: options.missingValueStrategy,
        includeAdvancedFeatureEngineering: options.includeAdvancedFeatureEngineering,
        crossValidationFolds: options.crossValidationFolds,
        hyperparameterTuning: options.hyperparameterTuning,
      });
      
      // Complete the project
      const completedProject: ProjectData = {
        id: projectId,
        competitionName,
        status: 'completed',
        progress: 100,
        currentStep: 'Advanced Kaggle notebook ready!',
        options,
        notebook,
        createdAt: new Date().toISOString(),
      };

      ClientStorage.saveProject(completedProject);
      setCurrentProject(completedProject);
      
      setGenerationStatus({
        status: 'completed',
        progress: 100,
        currentStep: 'Advanced Kaggle notebook ready!',
        projectId,
      });

      toast.success(`Your advanced ${options.selectedModel} notebook is ready for download!`);
      loadRecentProjects();

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate notebook';
      
      setGenerationStatus({
        status: 'error',
        progress: 0,
        currentStep: 'Generation failed',
        error: errorMessage,
        projectId,
      });

      // Update stored project with error
      const project = ClientStorage.getProject(projectId);
      if (project) {
        project.status = 'error';
        project.error = errorMessage;
        ClientStorage.saveProject(project);
      }

      toast.error(errorMessage);
    }
  };

  const handleGenerate = async () => {
    if (!validateInput(competitionInput)) {
      toast.error('Please enter a valid competition name (letters, numbers, hyphens) or Kaggle competition URL');
      return;
    }

    const projectId = `notebook_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const competitionName = competitionInput.includes('kaggle.com') 
      ? competitionInput.match(/competitions\/([^\/\?]+)/)?.[1] || competitionInput
      : competitionInput;

    // Create initial project
    const initialProject: ProjectData = {
      id: projectId,
      competitionName,
      status: 'generating',
      progress: 5,
      currentStep: 'Initializing advanced notebook generation...',
      options,
      createdAt: new Date().toISOString(),
      estimatedCompletion: new Date(Date.now() + 3 * 60 * 1000).toISOString(),
    };

    ClientStorage.saveProject(initialProject);
    setCurrentProject(initialProject);

    setGenerationStatus({ 
      status: 'generating', 
      progress: 5, 
      currentStep: 'Initializing advanced notebook generation...',
      projectId,
    });
    
    toast.success(`Advanced ${options.selectedModel} notebook generation started!`);

    // Start generation simulation
    simulateGeneration(projectId, competitionName, options);
  };

  const handleDownload = async () => {
    if (!currentProject?.notebook) {
      toast.error('No notebook available for download');
      return;
    }

    try {
      const blob = new Blob([currentProject.notebook.content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = currentProject.notebook.path;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      toast.success('Advanced Kaggle notebook downloaded!');
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
      case 'generating': return <Zap className="w-5 h-5 text-blue-500 animate-spin" />;
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

  const availableModels = getAvailableModels(competitionType);

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
                <p className="text-sm text-slate-600">Advanced single-notebook Kaggle competition generator</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary" className="bg-orange-100 text-orange-700">
                <BookOpen className="w-3 h-3 mr-1" />
                Single Notebook
              </Badge>
              <Badge variant="secondary" className="bg-indigo-100 text-indigo-700">
                <Target className="w-3 h-3 mr-1" />
                Advanced Config
              </Badge>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Input Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Hero Section */}
            <Card className="shadow-lg border-0 bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
              <CardContent className="p-8">
                <div className="flex items-center space-x-3 mb-4">
                  <Sparkles className="w-8 h-8" />
                  <h2 className="text-2xl font-bold">Advanced Kaggle-First Approach</h2>
                </div>
                <p className="text-lg opacity-90 mb-4">
                  Generate highly customized, competition-ready Kaggle notebooks with advanced model configurations, 
                  feature engineering, and hyperparameter tuning. Zero setup required!
                </p>
                <div className="flex items-center space-x-6 text-sm">
                  <div className="flex items-center space-x-2">
                    <Cpu className="w-4 h-4" />
                    <span>Model Selection</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Layers className="w-4 h-4" />
                    <span>Advanced Features</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Sliders className="w-4 h-4" />
                    <span>Hyperparameter Tuning</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Input Card */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Play className="w-5 h-5 text-indigo-500" />
                  <span>Generate Your Advanced Kaggle Notebook</span>
                </CardTitle>
                <CardDescription>
                  Enter a Kaggle competition name or URL to generate a customized, ready-to-run notebook
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="competition" className="text-sm font-medium text-slate-700">
                    Competition Name or URL
                  </Label>
                  <Input
                    id="competition"
                    placeholder="Example: titanic or https://www.kaggle.com/competitions/titanic"
                    value={competitionInput}
                    onChange={(e) => setCompetitionInput(e.target.value)}
                    className="h-12 text-base"
                  />
                  {competitionInput && (
                    <div className="text-xs text-slate-500">
                      Detected type: <span className="font-medium capitalize">{competitionType}</span>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-slate-700">Basic Options</h4>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="eda"
                        checked={options.includeEDA}
                        onCheckedChange={(checked) => 
                          setOptions(prev => ({ ...prev, includeEDA: checked as boolean }))
                        }
                      />
                      <Label htmlFor="eda" className="text-sm text-slate-700 flex items-center space-x-1">
                        <BarChart3 className="w-4 h-4" />
                        <span>Include Advanced EDA</span>
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="baseline"
                        checked={options.includeBaseline}
                        onCheckedChange={(checked) => 
                          setOptions(prev => ({ ...prev, includeBaseline: checked as boolean }))
                        }
                      />
                      <Label htmlFor="baseline" className="text-sm text-slate-700 flex items-center space-x-1">
                        <Brain className="w-4 h-4" />
                        <span>Include Model Training</span>
                      </Label>
                    </div>
                  </div>
                </div>

                {/* Advanced Options */}
                <Collapsible open={showAdvancedOptions} onOpenChange={setShowAdvancedOptions}>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="w-full justify-between">
                      <div className="flex items-center space-x-2">
                        <Settings className="w-4 h-4" />
                        <span>Advanced Configuration</span>
                      </div>
                      {showAdvancedOptions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-4 mt-4">
                    <div className="grid sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="model-select" className="text-sm font-medium text-slate-700">
                          Model Selection
                        </Label>
                        <Select
                          value={options.selectedModel}
                          onValueChange={(value) => setOptions(prev => ({ ...prev, selectedModel: value }))}
                        >
                          <SelectTrigger id="model-select">
                            <SelectValue placeholder="Select model" />
                          </SelectTrigger>
                          <SelectContent>
                            {availableModels.map((model) => (
                              <SelectItem key={model} value={model}>
                                {model}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="missing-strategy" className="text-sm font-medium text-slate-700">
                          Missing Value Strategy
                        </Label>
                        <Select
                          value={options.missingValueStrategy}
                          onValueChange={(value: 'median' | 'mean' | 'mode' | 'advanced') => 
                            setOptions(prev => ({ ...prev, missingValueStrategy: value }))
                          }
                        >
                          <SelectTrigger id="missing-strategy">
                            <SelectValue placeholder="Select strategy" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="median">Median Imputation</SelectItem>
                            <SelectItem value="mean">Mean Imputation</SelectItem>
                            <SelectItem value="mode">Mode Imputation</SelectItem>
                            <SelectItem value="advanced">Advanced (KNN + Iterative)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="cv-folds" className="text-sm font-medium text-slate-700">
                          Cross-Validation Folds
                        </Label>
                        <Select
                          value={options.crossValidationFolds.toString()}
                          onValueChange={(value) => 
                            setOptions(prev => ({ ...prev, crossValidationFolds: parseInt(value) }))
                          }
                        >
                          <SelectTrigger id="cv-folds">
                            <SelectValue placeholder="Select folds" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="3">3-fold</SelectItem>
                            <SelectItem value="5">5-fold</SelectItem>
                            <SelectItem value="10">10-fold</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-center space-x-2 pt-6">
                        <Checkbox
                          id="hyperparameter-tuning"
                          checked={options.hyperparameterTuning}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, hyperparameterTuning: checked as boolean }))
                          }
                        />
                        <Label htmlFor="hyperparameter-tuning" className="text-sm text-slate-700">
                          Enable Hyperparameter Tuning
                        </Label>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="advanced-features"
                        checked={options.includeAdvancedFeatureEngineering}
                        onCheckedChange={(checked) => 
                          setOptions(prev => ({ ...prev, includeAdvancedFeatureEngineering: checked as boolean }))
                        }
                      />
                      <Label htmlFor="advanced-features" className="text-sm text-slate-700">
                        Include Advanced Feature Engineering (polynomial, interactions, statistical features)
                      </Label>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                <Button 
                  onClick={handleGenerate}
                  disabled={generationStatus.status === 'generating'}
                  className="w-full h-12 text-base bg-indigo-500 hover:bg-indigo-600 transition-all duration-200"
                >
                  {generationStatus.status === 'generating' ? (
                    <>
                      <Zap className="w-4 h-4 mr-2 animate-spin" />
                      Generating Advanced Notebook...
                    </>
                  ) : (
                    <>
                      <FileText className="w-4 h-4 mr-2" />
                      Generate Advanced Kaggle Notebook
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
                  {generationStatus.estimatedCompletion && generationStatus.status === 'generating' && (
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
            {generationStatus.status === 'completed' && currentProject?.notebook && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span>Advanced Kaggle Notebook Ready!</span>
                  </CardTitle>
                  <CardDescription>
                    Your competition-ready notebook with advanced configurations is generated
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-slate-700 mb-3">Advanced Notebook Features</h4>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="flex items-center space-x-2">
                        <Cpu className="w-4 h-4 text-indigo-500" />
                        <span>{currentProject.options.selectedModel} model</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Target className="w-4 h-4 text-green-500" />
                        <span>{currentProject.options.missingValueStrategy} imputation</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <BarChart3 className="w-4 h-4 text-blue-500" />
                        <span>{currentProject.options.crossValidationFolds}-fold CV</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Layers className="w-4 h-4 text-purple-500" />
                        <span>{currentProject.options.includeAdvancedFeatureEngineering ? 'Advanced' : 'Basic'} features</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Sliders className="w-4 h-4 text-orange-500" />
                        <span>{currentProject.options.hyperparameterTuning ? 'Auto-tuned' : 'Default'} params</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-teal-500" />
                        <span>Single .ipynb file</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm text-slate-600 bg-indigo-50 p-3 rounded-lg">
                    <div>
                      <span className="font-medium">Expected Score: </span>
                      <span className="text-indigo-700">{currentProject.notebook.expectedScore}</span>
                    </div>
                    <div>
                      <span className="font-medium">Competition: </span>
                      <span>{currentProject.competitionName}</span>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <Button 
                      onClick={handleDownload}
                      className="w-full h-12 bg-green-500 hover:bg-green-600 transition-all duration-200"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download Advanced Kaggle Notebook (.ipynb)
                    </Button>
                    
                    <div className="text-xs text-slate-500 text-center">
                      💡 Upload this notebook directly to Kaggle and run all cells to generate your submission!
                    </div>
                  </div>
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
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge 
                              variant="secondary" 
                              className={`text-xs ${getDifficultyColor(competition.difficulty)}`}
                            >
                              {competition.difficulty}
                            </Badge>
                            <span className="text-xs text-slate-500">{competition.score}</span>
                          </div>
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
                    <span>Recent Notebooks</span>
                  </CardTitle>
                  <CardDescription>
                    Your recently generated notebooks
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {recentProjects.map((project) => (
                    <div 
                      key={project.id}
                      className="p-3 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors cursor-pointer"
                      onClick={() => {
                        if (project.status === 'completed' && project.notebook) {
                          setCurrentProject(project);
                          setGenerationStatus({
                            status: 'completed',
                            progress: 100,
                            currentStep: 'Notebook ready for download',
                            projectId: project.id,
                          });
                        }
                      }}
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
                              {project.options.selectedModel}
                            </span>
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

            {/* Advanced Features Card */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg">Advanced Features</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Model Selection</span>
                  <Badge variant="outline">6+ Algorithms</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Feature Engineering</span>
                  <Badge variant="outline">Polynomial + Interactions</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Hyperparameter Tuning</span>
                  <Badge variant="outline">Auto-optimization</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Cross-Validation</span>
                  <Badge variant="outline">3-10 fold CV</Badge>
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
                Generate advanced, ready-to-run Kaggle notebooks with customizable models, 
                feature engineering, and hyperparameter tuning. Zero setup, maximum productivity.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 mb-4">Advanced Features</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li>• Multiple ML algorithms</li>
                <li>• Advanced feature engineering</li>
                <li>• Hyperparameter optimization</li>
                <li>• Cross-validation strategies</li>
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
                  <a href="https://www.kaggle.com/learn" className="hover:text-indigo-500 transition-colors flex items-center space-x-1">
                    <span>Kaggle Learn</span>
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
            <p>Built for the Kaggle community 🚀</p>
          </div>
        </div>
      </footer>
    </div>
  );
}