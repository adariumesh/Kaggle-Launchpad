import { supabase } from './supabase-client';
import { ProjectData, GeneratedFile, KaggleNotebook } from './client-storage';
import { KaggleLaunchpadAI, CompetitionIntelligence } from './ai-agent';

export enum WorkflowState {
  QUEUED = 'queued',
  INITIALIZING = 'initializing',
  ANALYZING_COMPETITION = 'analyzing_competition',
  GATHERING_PRACTICES = 'gathering_practices',
  GENERATING_CODE = 'generating_code',
  CREATING_STRUCTURE = 'creating_structure',
  FINALIZING = 'finalizing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum WorkflowEvent {
  START = 'start',
  COMPETITION_ANALYZED = 'competition_analyzed',
  PRACTICES_GATHERED = 'practices_gathered',
  CODE_GENERATED = 'code_generated',
  STRUCTURE_CREATED = 'structure_created',
  FINALIZED = 'finalized',
  ERROR = 'error',
  CANCEL = 'cancel'
}

export interface WorkflowTransition {
  from: WorkflowState;
  to: WorkflowState;
  event: WorkflowEvent;
  progress: number;
  description: string;
}

export interface WorkflowContext {
  projectId: string;
  competitionName: string;
  options: ProjectData['options'];
  intelligence?: CompetitionIntelligence;
  generatedCode?: any;
  files?: GeneratedFile[];
  notebook?: KaggleNotebook;
  error?: string;
  startTime: Date;
  estimatedDuration: number; // in minutes
}

export class ProjectWorkflowManager {
  private static readonly TRANSITIONS: WorkflowTransition[] = [
    {
      from: WorkflowState.QUEUED,
      to: WorkflowState.INITIALIZING,
      event: WorkflowEvent.START,
      progress: 5,
      description: 'Initializing project generation...'
    },
    {
      from: WorkflowState.INITIALIZING,
      to: WorkflowState.ANALYZING_COMPETITION,
      event: WorkflowEvent.START,
      progress: 15,
      description: 'Analyzing competition and gathering intelligence...'
    },
    {
      from: WorkflowState.ANALYZING_COMPETITION,
      to: WorkflowState.GATHERING_PRACTICES,
      event: WorkflowEvent.COMPETITION_ANALYZED,
      progress: 35,
      description: 'Gathering latest ML practices and techniques...'
    },
    {
      from: WorkflowState.GATHERING_PRACTICES,
      to: WorkflowState.GENERATING_CODE,
      event: WorkflowEvent.PRACTICES_GATHERED,
      progress: 55,
      description: 'Generating optimized code with AI insights...'
    },
    {
      from: WorkflowState.GENERATING_CODE,
      to: WorkflowState.CREATING_STRUCTURE,
      event: WorkflowEvent.CODE_GENERATED,
      progress: 75,
      description: 'Creating enhanced project structure...'
    },
    {
      from: WorkflowState.CREATING_STRUCTURE,
      to: WorkflowState.FINALIZING,
      event: WorkflowEvent.STRUCTURE_CREATED,
      progress: 90,
      description: 'Finalizing project and generating documentation...'
    },
    {
      from: WorkflowState.FINALIZING,
      to: WorkflowState.COMPLETED,
      event: WorkflowEvent.FINALIZED,
      progress: 100,
      description: 'Project generation completed successfully!'
    }
  ];

  private context: WorkflowContext;
  private currentState: WorkflowState;
  private ai: KaggleLaunchpadAI;

  constructor(ai: KaggleLaunchpadAI, context: WorkflowContext) {
    this.ai = ai;
    this.context = context;
    this.currentState = WorkflowState.QUEUED;
  }

  static createContext(
    projectId: string,
    competitionName: string,
    options: ProjectData['options'],
    estimatedDuration: number = 15
  ): WorkflowContext {
    return {
      projectId,
      competitionName,
      options,
      startTime: new Date(),
      estimatedDuration
    };
  }

  async initialize(): Promise<void> {
    await this.persistState();
    console.log(`🚀 Workflow initialized for project: ${this.context.projectId}`);
  }

  async transition(event: WorkflowEvent, data?: any): Promise<boolean> {
    const transition = this.findTransition(this.currentState, event);
    
    if (!transition) {
      console.warn(`Invalid transition: ${this.currentState} -> ${event}`);
      return false;
    }

    const previousState = this.currentState;
    this.currentState = transition.to;

    // Update context with any provided data
    if (data) {
      Object.assign(this.context, data);
    }

    console.log(`🔄 Workflow transition: ${previousState} -> ${this.currentState}`);

    try {
      await this.persistState(transition.progress, transition.description);
      return true;
    } catch (error) {
      console.error('Failed to persist workflow state:', error);
      // Rollback state change
      this.currentState = previousState;
      return false;
    }
  }

  async fail(error: string): Promise<void> {
    this.currentState = WorkflowState.FAILED;
    this.context.error = error;
    
    console.error(`❌ Workflow failed for project ${this.context.projectId}: ${error}`);
    
    await this.persistState(
      this.getCurrentProgress(),
      `Failed: ${error}`
    );
  }

  async cancel(): Promise<void> {
    this.currentState = WorkflowState.CANCELLED;
    
    console.log(`🛑 Workflow cancelled for project: ${this.context.projectId}`);
    
    await this.persistState(
      this.getCurrentProgress(),
      'Project generation cancelled'
    );
  }

  getCurrentState(): WorkflowState {
    return this.currentState;
  }

  getContext(): WorkflowContext {
    return { ...this.context };
  }

  isCompleted(): boolean {
    return this.currentState === WorkflowState.COMPLETED;
  }

  isFailed(): boolean {
    return this.currentState === WorkflowState.FAILED;
  }

  isCancelled(): boolean {
    return this.currentState === WorkflowState.CANCELLED;
  }

  isRunning(): boolean {
    return ![
      WorkflowState.QUEUED,
      WorkflowState.COMPLETED,
      WorkflowState.FAILED,
      WorkflowState.CANCELLED
    ].includes(this.currentState);
  }

  getEstimatedCompletion(): Date {
    const elapsed = Date.now() - this.context.startTime.getTime();
    const progress = this.getCurrentProgress();
    
    if (progress === 0) {
      return new Date(Date.now() + this.context.estimatedDuration * 60 * 1000);
    }
    
    const totalEstimated = (elapsed / progress) * 100;
    const remaining = totalEstimated - elapsed;
    
    return new Date(Date.now() + remaining);
  }

  private findTransition(from: WorkflowState, event: WorkflowEvent): WorkflowTransition | null {
    return ProjectWorkflowManager.TRANSITIONS.find(
      t => t.from === from && t.event === event
    ) || null;
  }

  private getCurrentProgress(): number {
    const transition = ProjectWorkflowManager.TRANSITIONS.find(
      t => t.to === this.currentState
    );
    return transition?.progress || 0;
  }

  private async persistState(progress?: number, description?: string): Promise<void> {
    const currentProgress = progress || this.getCurrentProgress();
    const currentDescription = description || this.getCurrentStateDescription();

    const updateData = {
      status: this.currentState,
      progress: currentProgress,
      current_step: currentDescription,
      error_message: this.context.error || null,
      estimated_completion: this.getEstimatedCompletion().toISOString(),
      updated_at: new Date().toISOString()
    };

    // Add files and notebook if completed
    if (this.currentState === WorkflowState.COMPLETED) {
      updateData['files'] = this.context.files || null;
    }

    const { error } = await supabase
      .from('projects')
      .update(updateData)
      .eq('id', this.context.projectId);

    if (error) {
      throw new Error(`Failed to update project in database: ${error.message}`);
    }

    // Also save to client storage for offline access
    try {
      const { ClientStorage } = await import('./client-storage');
      const projectData: ProjectData = {
        id: this.context.projectId,
        competitionName: this.context.competitionName,
        status: this.currentState as any,
        progress: currentProgress,
        currentStep: currentDescription,
        options: this.context.options,
        files: this.context.files,
        notebook: this.context.notebook,
        error: this.context.error,
        createdAt: this.context.startTime.toISOString(),
        estimatedCompletion: this.getEstimatedCompletion().toISOString()
      };
      
      ClientStorage.saveProject(projectData);
    } catch (error) {
      console.warn('Failed to save to client storage:', error);
    }
  }

  private getCurrentStateDescription(): string {
    const transition = ProjectWorkflowManager.TRANSITIONS.find(
      t => t.to === this.currentState
    );
    
    switch (this.currentState) {
      case WorkflowState.FAILED:
        return `Failed: ${this.context.error || 'Unknown error'}`;
      case WorkflowState.CANCELLED:
        return 'Project generation cancelled';
      case WorkflowState.COMPLETED:
        return 'Project generation completed successfully!';
      default:
        return transition?.description || `Processing: ${this.currentState}`;
    }
  }

  // Workflow execution methods
  async executeWorkflow(): Promise<ProjectData> {
    try {
      await this.initialize();
      
      // Start the workflow
      await this.transition(WorkflowEvent.START);
      await this.transition(WorkflowEvent.START); // Move to analyzing
      
      // Step 1: Analyze competition
      const intelligence = await this.analyzeCompetition();
      await this.transition(WorkflowEvent.COMPETITION_ANALYZED, { intelligence });
      
      // Step 2: Gather practices (if enabled)
      if (this.context.options.useLatestPractices) {
        await this.gatherPractices(intelligence);
      }
      await this.transition(WorkflowEvent.PRACTICES_GATHERED);
      
      // Step 3: Generate code
      const generatedCode = await this.generateCode(intelligence);
      await this.transition(WorkflowEvent.CODE_GENERATED, { generatedCode });
      
      // Step 4: Create structure
      const { files, notebook } = await this.createProjectStructure(intelligence, generatedCode);
      await this.transition(WorkflowEvent.STRUCTURE_CREATED, { files, notebook });
      
      // Step 5: Finalize
      await this.finalize();
      await this.transition(WorkflowEvent.FINALIZED);
      
      return this.buildProjectData();
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      await this.fail(errorMessage);
      return this.buildProjectData();
    }
  }

  private async analyzeCompetition(): Promise<CompetitionIntelligence> {
    if (!this.context.options.competitionUrl) {
      throw new Error('Competition URL is required for analysis');
    }
    
    return await this.ai.analyzeCompetition(this.context.options.competitionUrl);
  }

  private async gatherPractices(intelligence: CompetitionIntelligence): Promise<void> {
    await this.ai.gatherLatestPractices(intelligence.competitionType);
  }

  private async generateCode(intelligence: CompetitionIntelligence): Promise<any> {
    return await this.ai.generateOptimizedCode(intelligence, this.context.options);
  }

  private async createProjectStructure(intelligence: CompetitionIntelligence, generatedCode: any): Promise<{
    files: GeneratedFile[];
    notebook: KaggleNotebook;
  }> {
    // Import the intelligent generator for structure creation
    const { IntelligentProjectGenerator } = await import('./intelligent-generator');
    const generator = new IntelligentProjectGenerator(this.ai);
    
    const enhancedFiles = await generator['createEnhancedProjectStructure'](
      intelligence,
      generatedCode,
      this.context.options
    );
    
    const notebook = generator['generateIntelligentNotebook'](intelligence, generatedCode);
    
    return {
      files: enhancedFiles,
      notebook: {
        path: 'notebooks/intelligent_analysis.ipynb',
        content: notebook,
        expectedScore: generatedCode.estimatedScore,
        description: `AI-generated analysis for ${intelligence.competitionType} competition`,
        complexity: intelligence.estimatedDifficulty,
        estimatedRuntime: this.estimateRuntime(intelligence),
        memoryUsage: this.estimateMemoryUsage(intelligence),
        techniques: this.extractTechniques(intelligence, generatedCode)
      }
    };
  }

  private async finalize(): Promise<void> {
    // Perform any final cleanup or validation
    console.log(`✅ Project ${this.context.projectId} generation completed successfully`);
  }

  private buildProjectData(): ProjectData {
    return {
      id: this.context.projectId,
      competitionName: this.context.competitionName,
      status: this.currentState as any,
      progress: this.getCurrentProgress(),
      currentStep: this.getCurrentStateDescription(),
      options: this.context.options,
      files: this.context.files,
      notebook: this.context.notebook,
      error: this.context.error,
      createdAt: this.context.startTime.toISOString(),
      estimatedCompletion: this.getEstimatedCompletion().toISOString()
    };
  }

  private estimateRuntime(intelligence: CompetitionIntelligence): string {
    const complexity = intelligence.estimatedDifficulty;
    const runtimeMap = {
      'beginner': '5-10 minutes',
      'intermediate': '15-30 minutes',
      'advanced': '30-60 minutes',
      'expert': '1-2 hours'
    };
    return runtimeMap[complexity] || '15-30 minutes';
  }

  private estimateMemoryUsage(intelligence: CompetitionIntelligence): string {
    const size = intelligence.datasetCharacteristics.size;
    const memoryMap = {
      'small': '< 1GB',
      'medium': '1-4GB',
      'large': '4-8GB',
      'very-large': '> 8GB'
    };
    return memoryMap[size] || '1-4GB';
  }

  private extractTechniques(intelligence: CompetitionIntelligence, generatedCode: any): string[] {
    const techniques = [
      ...intelligence.currentTrends,
      ...intelligence.recommendedBaselines,
      'Feature Engineering',
      'Cross Validation',
      'Hyperparameter Optimization'
    ];
    
    return [...new Set(techniques)]; // Remove duplicates
  }
}

// Workflow factory for creating and managing workflows
export class WorkflowFactory {
  private static activeWorkflows = new Map<string, ProjectWorkflowManager>();

  static async createProjectWorkflow(
    ai: KaggleLaunchpadAI,
    projectId: string,
    competitionName: string,
    options: ProjectData['options']
  ): Promise<ProjectWorkflowManager> {
    
    // Create initial project record in database
    const { error } = await supabase
      .from('projects')
      .insert({
        id: projectId,
        competition_name: competitionName,
        competition_url: options.competitionUrl || null,
        status: WorkflowState.QUEUED,
        progress: 0,
        current_step: 'Project queued for generation...',
        options: options,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });

    if (error) {
      throw new Error(`Failed to create project record: ${error.message}`);
    }

    const context = ProjectWorkflowManager.createContext(
      projectId,
      competitionName,
      options
    );

    const workflow = new ProjectWorkflowManager(ai, context);
    this.activeWorkflows.set(projectId, workflow);

    return workflow;
  }

  static getWorkflow(projectId: string): ProjectWorkflowManager | null {
    return this.activeWorkflows.get(projectId) || null;
  }

  static removeWorkflow(projectId: string): void {
    this.activeWorkflows.delete(projectId);
  }

  static getActiveWorkflows(): Map<string, ProjectWorkflowManager> {
    return new Map(this.activeWorkflows);
  }

  static async loadWorkflowFromDatabase(
    ai: KaggleLaunchpadAI,
    projectId: string
  ): Promise<ProjectWorkflowManager | null> {
    const { data, error } = await supabase
      .from('projects')
      .select('*')
      .eq('id', projectId)
      .single();

    if (error || !data) {
      return null;
    }

    const context: WorkflowContext = {
      projectId: data.id,
      competitionName: data.competition_name,
      options: data.options,
      startTime: new Date(data.created_at),
      estimatedDuration: 15,
      error: data.error_message || undefined
    };

    const workflow = new ProjectWorkflowManager(ai, context);
    workflow['currentState'] = data.status as WorkflowState;

    this.activeWorkflows.set(projectId, workflow);
    return workflow;
  }
}