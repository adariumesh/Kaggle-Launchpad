'use client';

import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  ChevronDown,
  ChevronRight,
  Globe,
  Mail,
  Download,
  RefreshCw,
  ArrowRight,
  RotateCcw,
  X,
  Zap,
  Brain,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { toast } from 'sonner';
import { apiClient, CreateProjectRequest } from '@/lib/api-client';
import { ClientStorage, WorkflowState, ProjectData } from '@/lib/client-storage';

export default function KaggleNotebookGenerator() {
  // State management
  const [kaggleUrl, setKaggleUrl] = useState('');
  const [email, setEmail] = useState('');
  const [showOptionsSection, setShowOptionsSection] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentProject, setCurrentProject] = useState<ProjectData | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');

  // Advanced options state
  const [includeEDA, setIncludeEDA] = useState(true);
  const [includeBaseline, setIncludeBaseline] = useState(true);
  const [selectedModel, setSelectedModel] = useState('random-forest');
  const [crossValidationFolds, setCrossValidationFolds] = useState(5);
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false);
  const [includeAdvancedFeatureEngineering, setIncludeAdvancedFeatureEngineering] = useState(false);
  const [ensembleMethod, setEnsembleMethod] = useState<'none' | 'voting' | 'stacking' | 'blending'>('none');
  const [includeExplainability, setIncludeExplainability] = useState(true);

  // Refs
  const optionsSectionRef = useRef<HTMLDivElement>(null);

  // Validation
  const isValidKaggleUrl = (url: string) => {
    const kagglePattern = /^https:\/\/www\.kaggle\.com\/c\/[a-zA-Z0-9-_]+\/?$/;
    return kagglePattern.test(url);
  };

  const isValidEmail = (email: string) => {
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailPattern.test(email);
  };

  // Handlers
  const handleUrlSubmit = () => {
    if (!kaggleUrl.trim()) {
      toast.error('Please enter a Kaggle competition URL');
      return;
    }

    if (!isValidKaggleUrl(kaggleUrl)) {
      toast.error('Please enter a valid Kaggle competition URL (e.g., https://www.kaggle.com/c/competition-name)');
      return;
    }

    setShowOptionsSection(true);
    toast.success('Competition URL validated successfully!');
    
    // Smooth scroll to options section
    setTimeout(() => {
      optionsSectionRef.current?.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }, 100);
  };

  const handleGenerate = async () => {
    if (!email.trim()) {
      toast.error('Please enter your email address');
      return;
    }

    if (!isValidEmail(email)) {
      toast.error('Please enter a valid email address');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setCurrentStep('Initializing project...');

    try {
      // Extract competition name from URL
      const urlParts = kaggleUrl.split('/');
      const competitionName = urlParts[urlParts.length - 1] || urlParts[urlParts.length - 2];

      const request: CreateProjectRequest = {
        competitionName,
        competitionUrl: kaggleUrl,
        options: {
          includeEDA,
          includeBaseline,
          initializeGit: false,
          selectedModel,
          missingValueStrategy: 'advanced',
          includeAdvancedFeatureEngineering,
          crossValidationFolds,
          hyperparameterTuning,
          ensembleMethod,
          autoFeatureSelection: includeAdvancedFeatureEngineering,
          includeDeepLearning: false,
          dataAugmentation: false,
          outlierDetection: 'isolation-forest',
          dimensionalityReduction: 'none',
          advancedValidation: 'stratified',
          includeExplainability,
          optimizationObjective: 'accuracy',
          includeAutoML: false,
          generateDocumentation: true,
          includeUnitTests: false,
          codeOptimization: 'basic',
          useLatestPractices: true,
          adaptToCompetitionType: true,
          includeWinningTechniques: true,
          optimizeForLeaderboard: true,
        }
      };

      const response = await apiClient.createProject(request);
      
      // Create project data for tracking
      const projectData: ProjectData = {
        id: response.projectId,
        competitionName,
        status: WorkflowState.QUEUED,
        progress: 0,
        currentStep: 'Project queued for generation...',
        options: request.options,
        createdAt: new Date().toISOString(),
      };

      setCurrentProject(projectData);
      ClientStorage.saveProject(projectData);

      // Start polling for updates
      pollProjectStatus(response.projectId);

      toast.success('Project generation started! You will receive an email when it\'s ready.');

    } catch (error) {
      console.error('Failed to create project:', error);
      toast.error('Failed to start project generation. Please try again.');
      setIsGenerating(false);
      setProgress(0);
      setCurrentStep('');
    }
  };

  const pollProjectStatus = async (projectId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const project = await apiClient.getProject(projectId);
        setCurrentProject(project);
        setProgress(project.progress);
        setCurrentStep(project.currentStep);

        if (project.status === WorkflowState.COMPLETED) {
          clearInterval(pollInterval);
          setIsGenerating(false);
          toast.success('🎉 Your Kaggle notebook is ready for download!');
        } else if (project.status === WorkflowState.FAILED) {
          clearInterval(pollInterval);
          setIsGenerating(false);
          toast.error('Project generation failed. Please try again.');
        }

        ClientStorage.saveProject(project);
      } catch (error) {
        console.error('Failed to poll project status:', error);
        // Continue polling even if one request fails
      }
    }, 3000); // Poll every 3 seconds

    // Stop polling after 30 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      if (isGenerating) {
        setIsGenerating(false);
        toast.error('Generation timeout. Please check your email or try again.');
      }
    }, 30 * 60 * 1000);
  };

  const handleDownloadNotebook = async () => {
    if (!currentProject) return;

    try {
      await apiClient.downloadNotebook(currentProject.id);
      toast.success('Notebook downloaded successfully!');
    } catch (error) {
      console.error('Failed to download notebook:', error);
      toast.error('Failed to download notebook. Please try again.');
    }
  };

  const handleDownloadProject = async () => {
    if (!currentProject) return;

    try {
      await apiClient.downloadProjectFiles(currentProject.id);
      toast.success('Project files downloaded successfully!');
    } catch (error) {
      console.error('Failed to download project files:', error);
      toast.error('Failed to download project files. Please try again.');
    }
  };

  const handleCancel = async () => {
    if (currentProject && isGenerating) {
      try {
        await apiClient.cancelProject(currentProject.id);
        setIsGenerating(false);
        setProgress(0);
        setCurrentStep('');
        setCurrentProject(null);
        toast.success('Project generation cancelled.');
      } catch (error) {
        console.error('Failed to cancel project:', error);
        toast.error('Failed to cancel project.');
      }
    }
  };

  const handleRetry = () => {
    setShowOptionsSection(false);
    setIsGenerating(false);
    setProgress(0);
    setCurrentStep('');
    setCurrentProject(null);
    setKaggleUrl('');
    setEmail('');
    toast.info('Ready to start a new project!');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black relative overflow-hidden">
      {/* Organic flowing background shapes */}
      <div className="bg-shape bg-shape-1 -top-32 -left-32" />
      <div className="bg-shape bg-shape-2 top-1/4 -right-40" />
      <div className="bg-shape bg-shape-3 bottom-1/4 -left-20" />
      <div className="bg-shape bg-shape-4 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
      <div className="bg-shape bg-shape-5 -bottom-32 -right-32" />
      
      {/* Content layer */}
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-12 max-w-4xl">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="flex items-center justify-center mb-4">
              <Brain className="h-12 w-12 text-purple-400 mr-3" />
              <h1 className="text-5xl font-bold text-white">
                Kaggle Launchpad
              </h1>
            </div>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Generate production-ready Kaggle competition notebooks with AI-powered analysis, 
              feature engineering, and winning techniques.
            </p>
          </div>

          {/* Landing Section - URL Input */}
          <Card className="glass-card border-white/10 mb-8">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl text-white flex items-center justify-center">
                <Globe className="h-6 w-6 mr-2 text-blue-400" />
                Enter Competition URL
              </CardTitle>
              <CardDescription className="text-gray-300">
                Paste the URL of any Kaggle competition to get started
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="kaggle-url" className="text-white">
                  Kaggle Competition URL
                </Label>
                <Input
                  id="kaggle-url"
                  type="url"
                  placeholder="https://www.kaggle.com/c/competition-name"
                  value={kaggleUrl}
                  onChange={(e) => setKaggleUrl(e.target.value)}
                  className="bg-white/10 border-white/20 text-white placeholder:text-gray-400"
                  disabled={isGenerating}
                />
                <p className="text-sm text-gray-400">
                  Example: https://www.kaggle.com/c/titanic
                </p>
              </div>
              
              <Button 
                onClick={handleUrlSubmit}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
                disabled={isGenerating || !kaggleUrl.trim()}
              >
                <ArrowRight className="h-4 w-4 mr-2" />
                Continue to Options
              </Button>
            </CardContent>
          </Card>

          {/* Options Section */}
          {showOptionsSection && (
            <div ref={optionsSectionRef} className="space-y-6">
              <Card className="glass-card border-white/10">
                <CardHeader>
                  <CardTitle className="text-2xl text-white flex items-center">
                    <Zap className="h-6 w-6 mr-2 text-yellow-400" />
                    Generation Options
                  </CardTitle>
                  <CardDescription className="text-gray-300">
                    Configure your notebook generation preferences
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Basic Options */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="include-eda" className="text-white">
                        Include EDA (Exploratory Data Analysis)
                      </Label>
                      <Switch
                        id="include-eda"
                        checked={includeEDA}
                        onCheckedChange={setIncludeEDA}
                        disabled={isGenerating}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label htmlFor="include-baseline" className="text-white">
                        Include Baseline Model
                      </Label>
                      <Switch
                        id="include-baseline"
                        checked={includeBaseline}
                        onCheckedChange={setIncludeBaseline}
                        disabled={isGenerating}
                      />
                    </div>
                  </div>

                  {/* Advanced Options Collapsible */}
                  <Collapsible open={showAdvancedOptions} onOpenChange={setShowAdvancedOptions}>
                    <CollapsibleTrigger asChild>
                      <Button 
                        variant="outline" 
                        className="w-full border-white/20 text-gray-300 hover:bg-white/10"
                        disabled={isGenerating}
                      >
                        {showAdvancedOptions ? (
                          <ChevronDown className="h-4 w-4 mr-2" />
                        ) : (
                          <ChevronRight className="h-4 w-4 mr-2" />
                        )}
                        Advanced Options
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="space-y-4 mt-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="model-select" className="text-white">
                            Primary Model
                          </Label>
                          <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isGenerating}>
                            <SelectTrigger className="bg-white/10 border-white/20 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="random-forest">Random Forest</SelectItem>
                              <SelectItem value="xgboost">XGBoost</SelectItem>
                              <SelectItem value="lightgbm">LightGBM</SelectItem>
                              <SelectItem value="catboost">CatBoost</SelectItem>
                              <SelectItem value="neural-network">Neural Network</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="cv-folds" className="text-white">
                            Cross-Validation Folds
                          </Label>
                          <Select 
                            value={crossValidationFolds.toString()} 
                            onValueChange={(value) => setCrossValidationFolds(parseInt(value))}
                            disabled={isGenerating}
                          >
                            <SelectTrigger className="bg-white/10 border-white/20 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="3">3 Folds</SelectItem>
                              <SelectItem value="5">5 Folds</SelectItem>
                              <SelectItem value="10">10 Folds</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="flex items-center justify-between">
                          <Label htmlFor="hyperparameter-tuning" className="text-white">
                            Hyperparameter Tuning
                          </Label>
                          <Switch
                            id="hyperparameter-tuning"
                            checked={hyperparameterTuning}
                            onCheckedChange={setHyperparameterTuning}
                            disabled={isGenerating}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <Label htmlFor="feature-engineering" className="text-white">
                            Advanced Feature Engineering
                          </Label>
                          <Switch
                            id="feature-engineering"
                            checked={includeAdvancedFeatureEngineering}
                            onCheckedChange={setIncludeAdvancedFeatureEngineering}
                            disabled={isGenerating}
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="ensemble-method" className="text-white">
                            Ensemble Method
                          </Label>
                          <Select value={ensembleMethod} onValueChange={setEnsembleMethod} disabled={isGenerating}>
                            <SelectTrigger className="bg-white/10 border-white/20 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="none">None</SelectItem>
                              <SelectItem value="voting">Voting</SelectItem>
                              <SelectItem value="stacking">Stacking</SelectItem>
                              <SelectItem value="blending">Blending</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="flex items-center justify-between">
                          <Label htmlFor="explainability" className="text-white">
                            Include Model Explainability
                          </Label>
                          <Switch
                            id="explainability"
                            checked={includeExplainability}
                            onCheckedChange={setIncludeExplainability}
                            disabled={isGenerating}
                          />
                        </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  {/* Email Input */}
                  <div className="space-y-2">
                    <Label htmlFor="email" className="text-white flex items-center">
                      <Mail className="h-4 w-4 mr-2 text-blue-400" />
                      Email Address
                    </Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="your.email@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="bg-white/10 border-white/20 text-white placeholder:text-gray-400"
                      disabled={isGenerating}
                    />
                    <p className="text-sm text-gray-400">
                      You'll receive a notification when your notebook is ready
                    </p>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex flex-col sm:flex-row gap-3">
                    <Button 
                      onClick={handleGenerate}
                      className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white"
                      disabled={isGenerating || !email.trim()}
                    >
                      {isGenerating ? (
                        <>
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Zap className="h-4 w-4 mr-2" />
                          Generate Notebook
                        </>
                      )}
                    </Button>

                    <Button 
                      onClick={handleRetry}
                      variant="outline"
                      className="border-white/20 text-gray-300 hover:bg-white/10"
                      disabled={isGenerating}
                    >
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Start Over
                    </Button>

                    {isGenerating && (
                      <Button 
                        onClick={handleCancel}
                        variant="outline"
                        className="border-red-500/50 text-red-400 hover:bg-red-500/10"
                      >
                        <X className="h-4 w-4 mr-2" />
                        Cancel
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Progress Section */}
              {(isGenerating || currentProject?.status === WorkflowState.COMPLETED) && (
                <Card className="glass-card border-white/10">
                  <CardHeader>
                    <CardTitle className="text-xl text-white flex items-center">
                      {currentProject?.status === WorkflowState.COMPLETED ? (
                        <CheckCircle className="h-5 w-5 mr-2 text-green-400" />
                      ) : currentProject?.status === WorkflowState.FAILED ? (
                        <AlertCircle className="h-5 w-5 mr-2 text-red-400" />
                      ) : (
                        <RefreshCw className="h-5 w-5 mr-2 text-blue-400 animate-spin" />
                      )}
                      Generation Progress
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-300">{currentStep}</span>
                        <span className="text-white font-medium">{progress}%</span>
                      </div>
                      <Progress value={progress} className="h-3" />
                    </div>

                    {currentProject?.status === WorkflowState.COMPLETED && (
                      <div className="flex flex-col sm:flex-row gap-3">
                        <Button 
                          onClick={handleDownloadNotebook}
                          className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download Notebook
                        </Button>
                        <Button 
                          onClick={handleDownloadProject}
                          variant="outline"
                          className="flex-1 border-white/20 text-gray-300 hover:bg-white/10"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download All Files
                        </Button>
                      </div>
                    )}

                    {currentProject?.status === WorkflowState.FAILED && (
                      <div className="text-center">
                        <p className="text-red-400 mb-4">
                          Generation failed. Please try again with different settings.
                        </p>
                        <Button 
                          onClick={handleRetry}
                          className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
                        >
                          <RotateCcw className="h-4 w-4 mr-2" />
                          Try Again
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}