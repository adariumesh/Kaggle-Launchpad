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
  Sliders,
  Beaker,
  Microscope,
  Gauge,
  Shield,
  Lightbulb,
  Workflow,
  Database,
  Activity
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
  // Ultra-advanced options
  ensembleMethod: 'none' | 'voting' | 'stacking' | 'blending';
  autoFeatureSelection: boolean;
  includeDeepLearning: boolean;
  dataAugmentation: boolean;
  outlierDetection: 'none' | 'isolation-forest' | 'local-outlier-factor' | 'one-class-svm';
  dimensionalityReduction: 'none' | 'pca' | 'tsne' | 'umap';
  advancedValidation: 'simple' | 'stratified' | 'time-series' | 'group' | 'adversarial';
  includeExplainability: boolean;
  optimizationObjective: 'accuracy' | 'speed' | 'memory' | 'interpretability';
  includeAutoML: boolean;
  generateDocumentation: boolean;
  includeUnitTests: boolean;
  codeOptimization: 'none' | 'basic' | 'advanced';
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
    'classification': ['XGBoost', 'RandomForest', 'LogisticRegression', 'LightGBM', 'CatBoost', 'SVM', 'NeuralNetwork'],
    'regression': ['XGBoost', 'RandomForest', 'LinearRegression', 'LightGBM', 'CatBoost', 'SVR', 'NeuralNetwork'],
    'nlp': ['LogisticRegression', 'RandomForest', 'XGBoost', 'NaiveBayes', 'SVM', 'BERT', 'Transformer'],
    'computer-vision': ['RandomForest', 'XGBoost', 'CNN', 'ResNet', 'EfficientNet', 'ViT'],
    'time-series': ['XGBoost', 'RandomForest', 'LSTM', 'Prophet', 'ARIMA', 'Transformer'],
    'other': ['XGBoost', 'RandomForest', 'LogisticRegression', 'LightGBM']
  };
  return modelMap[competitionType] || modelMap['other'];
};

const detectCompetitionType = (name: string): string => {
  const lowerName = name.toLowerCase();
  
  if (lowerName.includes('nlp') || lowerName.includes('sentiment') || lowerName.includes('text') || lowerName.includes('language')) {
    return 'nlp';
  }
  if (lowerName.includes('image') || lowerName.includes('vision') || lowerName.includes('digit') || lowerName.includes('photo')) {
    return 'computer-vision';
  }
  if (lowerName.includes('price') || lowerName.includes('sales') || lowerName.includes('revenue') || lowerName.includes('forecast')) {
    return 'regression';
  }
  if (lowerName.includes('time') || lowerName.includes('series') || lowerName.includes('temporal') || lowerName.includes('sequence')) {
    return 'time-series';
  }
  if (lowerName.includes('tabular') || lowerName.includes('structured')) {
    return 'tabular';
  }
  return 'classification';
};

export default function Home() {
  const [competitionInput, setCompetitionInput] = useState('');
  const [competitionType, setCompetitionType] = useState('classification');
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [showUltraAdvancedOptions, setShowUltraAdvancedOptions] = useState(false);
  const [options, setOptions] = useState<ProjectGenerationOptions>({
    includeEDA: true,
    includeBaseline: true,
    initializeGit: false,
    selectedModel: 'XGBoost',
    missingValueStrategy: 'median',
    includeAdvancedFeatureEngineering: false,
    crossValidationFolds: 5,
    hyperparameterTuning: false,
    // Ultra-advanced defaults
    ensembleMethod: 'none',
    autoFeatureSelection: false,
    includeDeepLearning: false,
    dataAugmentation: false,
    outlierDetection: 'none',
    dimensionalityReduction: 'none',
    advancedValidation: 'simple',
    includeExplainability: false,
    optimizationObjective: 'accuracy',
    includeAutoML: false,
    generateDocumentation: false,
    includeUnitTests: false,
    codeOptimization: 'none',
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

  const getComplexityLevel = (): 'beginner' | 'intermediate' | 'advanced' | 'expert' => {
    let complexity = 0;
    
    if (options.includeAdvancedFeatureEngineering) complexity += 1;
    if (options.hyperparameterTuning) complexity += 1;
    if (options.ensembleMethod !== 'none') complexity += 2;
    if (options.includeDeepLearning) complexity += 3;
    if (options.autoFeatureSelection) complexity += 1;
    if (options.dimensionalityReduction !== 'none') complexity += 1;
    if (options.advancedValidation !== 'simple') complexity += 1;
    if (options.includeExplainability) complexity += 1;
    if (options.includeAutoML) complexity += 2;
    if (options.codeOptimization === 'advanced') complexity += 1;
    
    if (complexity <= 2) return 'beginner';
    if (complexity <= 5) return 'intermediate';
    if (complexity <= 9) return 'advanced';
    return 'expert';
  };

  const simulateGeneration = async (projectId: string, competitionName: string, options: ProjectGenerationOptions) => {
    const complexity = getComplexityLevel();
    const baseSteps = [
      { step: 'Analyzing competition architecture...', progress: 10, delay: 800 },
      { step: 'Configuring ultra-advanced model settings...', progress: 25, delay: 1000 },
      { step: 'Generating optimized Kaggle notebook...', progress: 45, delay: 1500 },
    ];

    const advancedSteps = [];
    if (options.ensembleMethod !== 'none') {
      advancedSteps.push({ step: `Setting up ${options.ensembleMethod} ensemble...`, progress: 60, delay: 1200 });
    }
    if (options.includeDeepLearning) {
      advancedSteps.push({ step: 'Configuring deep learning architecture...', progress: 70, delay: 1000 });
    }
    if (options.includeAutoML) {
      advancedSteps.push({ step: 'Integrating AutoML pipeline...', progress: 80, delay: 1500 });
    }
    if (options.includeExplainability) {
      advancedSteps.push({ step: 'Adding model explainability features...', progress: 85, delay: 800 });
    }

    const finalSteps = [
      { step: 'Optimizing memory usage and performance...', progress: 90, delay: 800 },
      { step: 'Finalizing ultra-advanced solution...', progress: 95, delay: 600 },
    ];

    const allSteps = [...baseSteps, ...advancedSteps, ...finalSteps];

    try {
      for (const { step, progress, delay } of allSteps) {
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
        ensembleMethod: options.ensembleMethod,
        autoFeatureSelection: options.autoFeatureSelection,
        includeDeepLearning: options.includeDeepLearning,
        dataAugmentation: options.dataAugmentation,
        outlierDetection: options.outlierDetection,
        dimensionalityReduction: options.dimensionalityReduction,
        advancedValidation: options.advancedValidation,
        includeExplainability: options.includeExplainability,
        optimizationObjective: options.optimizationObjective,
        includeAutoML: options.includeAutoML,
        generateDocumentation: options.generateDocumentation,
        includeUnitTests: options.includeUnitTests,
        codeOptimization: options.codeOptimization,
      });
      
      // Complete the project
      const completedProject: ProjectData = {
        id: projectId,
        competitionName,
        status: 'completed',
        progress: 100,
        currentStep: `Ultra-advanced ${complexity} notebook ready!`,
        options,
        notebook,
        createdAt: new Date().toISOString(),
      };

      ClientStorage.saveProject(completedProject);
      setCurrentProject(completedProject);
      
      setGenerationStatus({
        status: 'completed',
        progress: 100,
        currentStep: `Ultra-advanced ${complexity} notebook ready!`,
        projectId,
      });

      toast.success(`Your ${complexity}-level ${options.selectedModel} notebook is ready!`);
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

    const complexity = getComplexityLevel();
    const estimatedTime = complexity === 'expert' ? 5 : complexity === 'advanced' ? 4 : 3;

    // Create initial project
    const initialProject: ProjectData = {
      id: projectId,
      competitionName,
      status: 'generating',
      progress: 5,
      currentStep: `Initializing ${complexity}-level notebook generation...`,
      options,
      createdAt: new Date().toISOString(),
      estimatedCompletion: new Date(Date.now() + estimatedTime * 60 * 1000).toISOString(),
    };

    ClientStorage.saveProject(initialProject);
    setCurrentProject(initialProject);

    setGenerationStatus({ 
      status: 'generating', 
      progress: 5, 
      currentStep: `Initializing ${complexity}-level notebook generation...`,
      projectId,
    });
    
    toast.success(`Ultra-advanced ${options.selectedModel} notebook generation started!`);

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
      
      toast.success('Ultra-advanced Kaggle notebook downloaded!');
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

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-blue-100 text-blue-800';
      case 'advanced': return 'bg-orange-100 text-orange-800';
      case 'expert': return 'bg-purple-100 text-purple-800';
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
  const currentComplexity = getComplexityLevel();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Rocket className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Kaggle Launchpad Ultra</h1>
                <p className="text-sm text-slate-600">Ultra-advanced AI-powered Kaggle competition generator</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary" className="bg-orange-100 text-orange-700">
                <Beaker className="w-3 h-3 mr-1" />
                Ultra-Advanced
              </Badge>
              <Badge variant="secondary" className={getComplexityColor(currentComplexity)}>
                <Gauge className="w-3 h-3 mr-1" />
                {currentComplexity}
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
            <Card className="shadow-lg border-0 bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-600 text-white">
              <CardContent className="p-8">
                <div className="flex items-center space-x-3 mb-4">
                  <Sparkles className="w-8 h-8" />
                  <h2 className="text-2xl font-bold">Ultra-Advanced AI Competition Generator</h2>
                </div>
                <p className="text-lg opacity-90 mb-4">
                  Generate production-ready Kaggle notebooks with cutting-edge ML techniques: ensemble methods, 
                  AutoML integration, deep learning, advanced feature engineering, and model explainability.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="flex items-center space-x-2">
                    <Workflow className="w-4 h-4" />
                    <span>Ensemble Methods</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Brain className="w-4 h-4" />
                    <span>AutoML Integration</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Microscope className="w-4 h-4" />
                    <span>Model Explainability</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Shield className="w-4 h-4" />
                    <span>Production Ready</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Input Card */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Play className="w-5 h-5 text-indigo-500" />
                  <span>Generate Ultra-Advanced Kaggle Notebook</span>
                </CardTitle>
                <CardDescription>
                  Create a cutting-edge, competition-ready notebook with advanced ML techniques and optimizations
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
                    <div className="flex items-center space-x-2 text-xs text-slate-500">
                      <span>Detected type: <span className="font-medium capitalize">{competitionType}</span></span>
                      <Badge variant="outline" className={getComplexityColor(currentComplexity)}>
                        {currentComplexity} complexity
                      </Badge>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-slate-700">Core Options</h4>
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
                        <span>Ultra-Advanced EDA</span>
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
                        <span>Advanced Model Training</span>
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
                          Cross-Validation
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

                      <div className="space-y-2">
                        <Label htmlFor="ensemble-method" className="text-sm font-medium text-slate-700">
                          Ensemble Method
                        </Label>
                        <Select
                          value={options.ensembleMethod}
                          onValueChange={(value: 'none' | 'voting' | 'stacking' | 'blending') => 
                            setOptions(prev => ({ ...prev, ensembleMethod: value }))
                          }
                        >
                          <SelectTrigger id="ensemble-method">
                            <SelectValue placeholder="Select ensemble" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">No Ensemble</SelectItem>
                            <SelectItem value="voting">Voting Ensemble</SelectItem>
                            <SelectItem value="stacking">Stacking Ensemble</SelectItem>
                            <SelectItem value="blending">Blending Ensemble</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="grid sm:grid-cols-2 gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="hyperparameter-tuning"
                          checked={options.hyperparameterTuning}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, hyperparameterTuning: checked as boolean }))
                          }
                        />
                        <Label htmlFor="hyperparameter-tuning" className="text-sm text-slate-700">
                          Hyperparameter Tuning (Optuna)
                        </Label>
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
                          Advanced Feature Engineering
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="auto-feature-selection"
                          checked={options.autoFeatureSelection}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, autoFeatureSelection: checked as boolean }))
                          }
                        />
                        <Label htmlFor="auto-feature-selection" className="text-sm text-slate-700">
                          Automatic Feature Selection
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="explainability"
                          checked={options.includeExplainability}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, includeExplainability: checked as boolean }))
                          }
                        />
                        <Label htmlFor="explainability" className="text-sm text-slate-700">
                          Model Explainability (SHAP)
                        </Label>
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                {/* Ultra-Advanced Options */}
                <Collapsible open={showUltraAdvancedOptions} onOpenChange={setShowUltraAdvancedOptions}>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="w-full justify-between border-purple-200 hover:bg-purple-50">
                      <div className="flex items-center space-x-2">
                        <Beaker className="w-4 h-4 text-purple-600" />
                        <span className="text-purple-700">Ultra-Advanced Options</span>
                      </div>
                      {showUltraAdvancedOptions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-4 mt-4 p-4 bg-purple-50 rounded-lg">
                    <div className="grid sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="outlier-detection" className="text-sm font-medium text-slate-700">
                          Outlier Detection
                        </Label>
                        <Select
                          value={options.outlierDetection}
                          onValueChange={(value: 'none' | 'isolation-forest' | 'local-outlier-factor' | 'one-class-svm') => 
                            setOptions(prev => ({ ...prev, outlierDetection: value }))
                          }
                        >
                          <SelectTrigger id="outlier-detection">
                            <SelectValue placeholder="Select method" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">No Detection</SelectItem>
                            <SelectItem value="isolation-forest">Isolation Forest</SelectItem>
                            <SelectItem value="local-outlier-factor">Local Outlier Factor</SelectItem>
                            <SelectItem value="one-class-svm">One-Class SVM</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="dimensionality-reduction" className="text-sm font-medium text-slate-700">
                          Dimensionality Reduction
                        </Label>
                        <Select
                          value={options.dimensionalityReduction}
                          onValueChange={(value: 'none' | 'pca' | 'tsne' | 'umap') => 
                            setOptions(prev => ({ ...prev, dimensionalityReduction: value }))
                          }
                        >
                          <SelectTrigger id="dimensionality-reduction">
                            <SelectValue placeholder="Select method" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">No Reduction</SelectItem>
                            <SelectItem value="pca">PCA</SelectItem>
                            <SelectItem value="tsne">t-SNE</SelectItem>
                            <SelectItem value="umap">UMAP</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="advanced-validation" className="text-sm font-medium text-slate-700">
                          Validation Strategy
                        </Label>
                        <Select
                          value={options.advancedValidation}
                          onValueChange={(value: 'simple' | 'stratified' | 'time-series' | 'group' | 'adversarial') => 
                            setOptions(prev => ({ ...prev, advancedValidation: value }))
                          }
                        >
                          <SelectTrigger id="advanced-validation">
                            <SelectValue placeholder="Select strategy" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="simple">Simple K-Fold</SelectItem>
                            <SelectItem value="stratified">Stratified K-Fold</SelectItem>
                            <SelectItem value="time-series">Time Series Split</SelectItem>
                            <SelectItem value="group">Group K-Fold</SelectItem>
                            <SelectItem value="adversarial">Adversarial Validation</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="optimization-objective" className="text-sm font-medium text-slate-700">
                          Optimization Objective
                        </Label>
                        <Select
                          value={options.optimizationObjective}
                          onValueChange={(value: 'accuracy' | 'speed' | 'memory' | 'interpretability') => 
                            setOptions(prev => ({ ...prev, optimizationObjective: value }))
                          }
                        >
                          <SelectTrigger id="optimization-objective">
                            <SelectValue placeholder="Select objective" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="accuracy">Maximum Accuracy</SelectItem>
                            <SelectItem value="speed">Inference Speed</SelectItem>
                            <SelectItem value="memory">Memory Efficiency</SelectItem>
                            <SelectItem value="interpretability">Model Interpretability</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="grid sm:grid-cols-2 gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="deep-learning"
                          checked={options.includeDeepLearning}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, includeDeepLearning: checked as boolean }))
                          }
                        />
                        <Label htmlFor="deep-learning" className="text-sm text-slate-700">
                          Deep Learning Integration
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="automl"
                          checked={options.includeAutoML}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, includeAutoML: checked as boolean }))
                          }
                        />
                        <Label htmlFor="automl" className="text-sm text-slate-700">
                          AutoML Integration (FLAML)
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="data-augmentation"
                          checked={options.dataAugmentation}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, dataAugmentation: checked as boolean }))
                          }
                        />
                        <Label htmlFor="data-augmentation" className="text-sm text-slate-700">
                          Data Augmentation
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="documentation"
                          checked={options.generateDocumentation}
                          onCheckedChange={(checked) => 
                            setOptions(prev => ({ ...prev, generateDocumentation: checked as boolean }))
                          }
                        />
                        <Label htmlFor="documentation" className="text-sm text-slate-700">
                          Auto-Generated Documentation
                        </Label>
                      </div>
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                <Button 
                  onClick={handleGenerate}
                  disabled={generationStatus.status === 'generating'}
                  className="w-full h-12 text-base bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 transition-all duration-200"
                >
                  {generationStatus.status === 'generating' ? (
                    <>
                      <Zap className="w-4 h-4 mr-2 animate-spin" />
                      Generating Ultra-Advanced Notebook...
                    </>
                  ) : (
                    <>
                      <Beaker className="w-4 h-4 mr-2" />
                      Generate Ultra-Advanced Kaggle Notebook
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
                    <span>Ultra-Advanced Kaggle Notebook Ready!</span>
                  </CardTitle>
                  <CardDescription>
                    Your cutting-edge, production-ready notebook with advanced ML techniques
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-slate-700 mb-3">Ultra-Advanced Features</h4>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="flex items-center space-x-2">
                        <Cpu className="w-4 h-4 text-indigo-500" />
                        <span>{currentProject.options.selectedModel} model</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Database className="w-4 h-4 text-green-500" />
                        <span>{currentProject.options.missingValueStrategy} imputation</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Activity className="w-4 h-4 text-blue-500" />
                        <span>{currentProject.options.crossValidationFolds}-fold {currentProject.options.advancedValidation} CV</span>
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
                        <Workflow className="w-4 h-4 text-pink-500" />
                        <span>{currentProject.options.ensembleMethod !== 'none' ? currentProject.options.ensembleMethod : 'Single'} model</span>
                      </div>
                      {currentProject.options.includeExplainability && (
                        <div className="flex items-center space-x-2">
                          <Microscope className="w-4 h-4 text-teal-500" />
                          <span>SHAP explainability</span>
                        </div>
                      )}
                      {currentProject.options.includeAutoML && (
                        <div className="flex items-center space-x-2">
                          <Brain className="w-4 h-4 text-red-500" />
                          <span>AutoML integration</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-indigo-50 p-3 rounded-lg">
                      <div className="text-sm text-slate-600">Expected Score</div>
                      <div className="text-lg font-semibold text-indigo-700">{currentProject.notebook.expectedScore}</div>
                    </div>
                    <div className="bg-purple-50 p-3 rounded-lg">
                      <div className="text-sm text-slate-600">Complexity Level</div>
                      <div className="text-lg font-semibold text-purple-700 capitalize">{currentProject.notebook.complexity}</div>
                    </div>
                    <div className="bg-green-50 p-3 rounded-lg">
                      <div className="text-sm text-slate-600">Runtime</div>
                      <div className="text-lg font-semibold text-green-700">{currentProject.notebook.estimatedRuntime}</div>
                    </div>
                    <div className="bg-orange-50 p-3 rounded-lg">
                      <div className="text-sm text-slate-600">Memory Usage</div>
                      <div className="text-lg font-semibold text-orange-700">{currentProject.notebook.memoryUsage}</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <Button 
                      onClick={handleDownload}
                      className="w-full h-12 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 transition-all duration-200"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download Ultra-Advanced Notebook (.ipynb)
                    </Button>
                    
                    <div className="text-xs text-slate-500 text-center">
                      🚀 Upload this notebook directly to Kaggle and run all cells to generate your submission!
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
                  Quick-start with these competitions
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

            {/* Ultra-Advanced Features Card */}
            <Card className="shadow-lg border-0 bg-gradient-to-br from-purple-50 to-pink-50">
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Beaker className="w-5 h-5 text-purple-600" />
                  <span>Ultra-Advanced Features</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Ensemble Methods</span>
                  <Badge variant="outline" className="bg-purple-100 text-purple-700">Voting • Stacking • Blending</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">AutoML Integration</span>
                  <Badge variant="outline" className="bg-blue-100 text-blue-700">FLAML • Auto-sklearn</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Model Explainability</span>
                  <Badge variant="outline" className="bg-green-100 text-green-700">SHAP • Permutation</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Advanced Validation</span>
                  <Badge variant="outline" className="bg-orange-100 text-orange-700">Adversarial • Time-Series</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Deep Learning</span>
                  <Badge variant="outline" className="bg-red-100 text-red-700">Neural Networks • CNNs</Badge>
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
                <span className="font-semibold text-slate-900">Kaggle Launchpad Ultra</span>
              </div>
              <p className="text-sm text-slate-600">
                Generate ultra-advanced, production-ready Kaggle notebooks with cutting-edge ML techniques, 
                ensemble methods, AutoML integration, and model explainability.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 mb-4">Ultra-Advanced Features</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li>• Ensemble methods (voting, stacking, blending)</li>
                <li>• AutoML integration with FLAML</li>
                <li>• Model explainability with SHAP</li>
                <li>• Advanced validation strategies</li>
                <li>• Deep learning integration</li>
                <li>• Hyperparameter optimization</li>
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
            <p>&copy; 2025 Kaggle Launchpad Ultra. All rights reserved.</p>
            <p>Ultra-Advanced AI for the Kaggle community 🚀</p>
          </div>
        </div>
      </footer>
    </div>
  );
}