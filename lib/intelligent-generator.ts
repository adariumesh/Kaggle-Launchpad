import { KaggleLaunchpadAI, CompetitionIntelligence } from './ai-agent';
import { ProjectData } from './client-storage';
import { ProjectWorkflowManager, WorkflowFactory, WorkflowContext } from './project-workflow';

export interface IntelligentGenerationOptions extends ProjectData['options'] {
  competitionUrl: string;
  useLatestPractices: boolean;
  adaptToCompetitionType: boolean;
  includeWinningTechniques: boolean;
  optimizeForLeaderboard: boolean;
}

export class IntelligentProjectGenerator {
  private ai: KaggleLaunchpadAI;

  constructor(ai: KaggleLaunchpadAI) {
    this.ai = ai;
  }

  async generateIntelligentProject(
    competitionName: string,
    options: IntelligentGenerationOptions
  ): Promise<ProjectData> {
    const projectId = `project_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      // Create and initialize workflow
      const workflow = await WorkflowFactory.createProjectWorkflow(
        this.ai,
        projectId,
        competitionName,
        options
      );

      // Execute the workflow
      const result = await workflow.executeWorkflow();
      
      // Clean up workflow from active workflows
      WorkflowFactory.removeWorkflow(projectId);
      
      return result;
      
    } catch (error) {
      console.error('Project generation failed:', error);
      
      // Try to get the workflow to update its state
      const workflow = WorkflowFactory.getWorkflow(projectId);
      if (workflow) {
        await workflow.fail(error instanceof Error ? error.message : 'Unknown error occurred');
        WorkflowFactory.removeWorkflow(projectId);
        return workflow['buildProjectData']();
      }
      
      // Fallback: create a failed project data
      return {
        id: projectId,
        competitionName,
        status: 'failed' as any,
        progress: 0,
        currentStep: 'Project generation failed',
        options,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        createdAt: new Date().toISOString()
      };
    }
  }

  // Make these methods accessible for the workflow
  async createEnhancedProjectStructure(
    intelligence: CompetitionIntelligence,
    generatedCode: any,
    options: IntelligentGenerationOptions
  ): Promise<Array<{ path: string; content: string; }>> {
    const files = [...generatedCode.files];
    
    // Add intelligent configuration
    files.push({
      path: 'config/intelligent_config.py',
      content: this.generateIntelligentConfig(intelligence, options)
    });
    
    // Add competition-specific utilities
    files.push({
      path: 'src/competition_utils.py',
      content: this.generateCompetitionUtils(intelligence)
    });
    
    // Add AI-recommended preprocessing
    files.push({
      path: 'src/intelligent_preprocessing.py',
      content: this.generateIntelligentPreprocessing(intelligence)
    });
    
    // Add model recommendations
    files.push({
      path: 'src/recommended_models.py',
      content: this.generateRecommendedModels(intelligence)
    });
    
    // Add evaluation strategies
    files.push({
      path: 'src/evaluation_strategies.py',
      content: this.generateEvaluationStrategies(intelligence)
    });
    
    return files;
  }

  generateIntelligentNotebook(intelligence: CompetitionIntelligence, generatedCode: any): string {
    return `{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AI-Generated Competition Analysis\\n",
    "\\n",
    "**Competition Type:** ${intelligence.competitionType}\\n",
    "**Estimated Difficulty:** ${intelligence.estimatedDifficulty}\\n",
    "**Expected Score:** ${generatedCode.estimatedScore}\\n",
    "\\n",
    "This notebook was intelligently generated based on:\\n",
    "- Analysis of ${intelligence.winningApproaches.length} winning approaches\\n",
    "- Latest ML best practices from 2025\\n",
    "- Competition-specific optimizations\\n",
    "\\n",
    "## Key Insights\\n",
    "\\n",
    "${intelligence.currentTrends.map(trend => `- ${trend}`).join('\\n')}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import AI-optimized libraries\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Import intelligent modules\\n",
    "from src.intelligent_preprocessing import intelligent_preprocessor\\n",
    "from src.recommended_models import model_engine\\n",
    "from src.evaluation_strategies import intelligent_evaluator\\n",
    "from config.intelligent_config import config\\n",
    "\\n",
    "# Set up logging\\n",
    "import logging\\n",
    "logging.basicConfig(level=logging.INFO)\\n",
    "logger = logging.getLogger(__name__)\\n",
    "\\n",
    "print('🤖 AI-Powered Kaggle Solution Initialized')\\n",
    "print(f'Competition Type: {config.competition_type}')\\n",
    "print(f'Target Type: {config.target_type}')\\n",
    "print(f'Recommended Models: {config.recommended_models}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}`;
  }

  private generateIntelligentConfig(
    intelligence: CompetitionIntelligence,
    options: IntelligentGenerationOptions
  ): string {
    return `"""
Intelligent Configuration for ${intelligence.competitionType} Competition
Generated by Kaggle Launchpad AI Agent

This configuration is optimized based on:
- Competition type: ${intelligence.competitionType}
- Dataset characteristics: ${JSON.stringify(intelligence.datasetCharacteristics)}
- Winning approaches analysis
- Latest ML best practices
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class IntelligentConfig:
    # Competition Details
    competition_type: str = "${intelligence.competitionType}"
    estimated_difficulty: str = "${intelligence.estimatedDifficulty}"
    
    # Dataset Configuration
    target_type: str = "${intelligence.datasetCharacteristics.targetType}"
    dataset_size: str = "${intelligence.datasetCharacteristics.size}"
    feature_count: int = ${intelligence.datasetCharacteristics.features || 0}
    
    # Model Recommendations (based on winning approaches)
    recommended_models: List[str] = ${JSON.stringify(intelligence.recommendedBaselines)}
    
    # Feature Engineering (based on latest practices)
    feature_engineering_techniques: List[str] = [
        "target_encoding",
        "frequency_encoding", 
        "interaction_features",
        "polynomial_features",
        "statistical_features"
    ]
    
    # Cross-Validation Strategy
    cv_strategy: str = "${this.getCVStrategy(intelligence)}"
    cv_folds: int = ${options.crossValidationFolds || 5}
    
    # Hyperparameter Optimization
    optimization_trials: int = ${options.hyperparameterTuning ? 100 : 10}
    optimization_timeout: int = 3600  # 1 hour
    
    # Advanced Options
    ensemble_method: str = "${options.ensembleMethod || 'none'}"
    use_auto_feature_selection: bool = ${options.autoFeatureSelection || false}
    include_deep_learning: bool = ${options.includeDeepLearning || false}
    
    # Paths
    data_path: str = "data/"
    models_path: str = "models/"
    submissions_path: str = "submissions/"
    notebooks_path: str = "notebooks/"

# Global configuration instance
config = IntelligentConfig()
`;
  }

  private generateCompetitionUtils(intelligence: CompetitionIntelligence): string {
    return `"""
Competition-specific utilities optimized for ${intelligence.competitionType}
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CompetitionUtils:
    """Utilities specific to this competition type and characteristics"""
    
    def __init__(self):
        self.competition_type = "${intelligence.competitionType}"
        self.target_type = "${intelligence.datasetCharacteristics.targetType}"
    
    def load_and_validate_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate competition data"""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Test shape: {test_df.shape}")
            
            # Competition-specific validations
            self._validate_data_quality(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data_quality(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate data quality based on competition characteristics"""
        
        # Check for missing values
        train_missing = train_df.isnull().sum().sum()
        test_missing = test_df.isnull().sum().sum()
        
        if train_missing > 0 or test_missing > 0:
            logger.warning(f"Missing values detected - Train: {train_missing}, Test: {test_missing}")
        
        # Check feature consistency
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        if 'target' in train_cols:
            train_cols.remove('target')
        
        if train_cols != test_cols:
            logger.warning("Feature mismatch between train and test sets")
    
    def get_competition_metric(self) -> str:
        """Get the appropriate metric for this competition"""
        metric_mapping = {
            'classification': 'accuracy',
            'regression': 'rmse',
            'ranking': 'ndcg'
        }
        return metric_mapping.get(self.target_type, 'accuracy')
    
    def create_submission_format(self, predictions: np.ndarray, test_ids: np.ndarray) -> pd.DataFrame:
        """Create submission in the correct format"""
        submission = pd.DataFrame({
            'id': test_ids,
            'target': predictions
        })
        return submission

# Global utility instance
competition_utils = CompetitionUtils()
`;
  }

  private generateIntelligentPreprocessing(intelligence: CompetitionIntelligence): string {
    const techniques = intelligence.currentTrends.join('", "');
    
    return `"""
Intelligent preprocessing pipeline based on latest ML practices
Optimized for ${intelligence.competitionType} competitions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class IntelligentPreprocessor:
    """AI-optimized preprocessing pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.techniques = ["${techniques}"]
    
    def fit_transform(self, train_df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """Fit and transform training data"""
        logger.info("Starting intelligent preprocessing...")
        
        # Separate features and target
        if target_col in train_df.columns:
            X = train_df.drop(columns=[target_col])
            y = train_df[target_col]
        else:
            X = train_df.copy()
            y = None
        
        # Apply preprocessing steps
        X_processed = self._apply_preprocessing_pipeline(X, y, fit=True)
        
        if y is not None:
            result = X_processed.copy()
            result[target_col] = y
            return result
        
        return X_processed
    
    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted preprocessors"""
        return self._apply_preprocessing_pipeline(test_df, fit=False)
    
    def _apply_preprocessing_pipeline(self, df: pd.DataFrame, y=None, fit: bool = False) -> pd.DataFrame:
        """Apply the complete preprocessing pipeline"""
        
        X = df.copy()
        
        # 1. Handle missing values intelligently
        X = self._handle_missing_values(X, fit=fit)
        
        # 2. Encode categorical variables
        X = self._encode_categorical_features(X, fit=fit)
        
        # 3. Feature engineering based on competition type
        X = self._engineer_features(X, fit=fit)
        
        # 4. Scale numerical features
        X = self._scale_features(X, fit=fit)
        
        # 5. Feature selection (only if target is available)
        if fit and y is not None:
            X = self._select_features(X, y, fit=fit)
        elif not fit and 'feature_selector' in self.feature_selectors:
            X = self._select_features(X, fit=fit)
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        return X
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Intelligent missing value handling"""
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Use median for numerical features
                    if fit:
                        fill_value = df[col].median()
                        self.scalers[f'{col}_fill'] = fill_value
                    else:
                        fill_value = self.scalers.get(f'{col}_fill', df[col].median())
                    
                    df[col].fillna(fill_value, inplace=True)
                else:
                    # Use mode for categorical features
                    if fit:
                        fill_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown'
                        self.scalers[f'{col}_fill'] = fill_value
                    else:
                        fill_value = self.scalers.get(f'{col}_fill', 'unknown')
                    
                    df[col].fillna(fill_value, inplace=True)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features"""
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
            else:
                encoder = self.encoders.get(col)
                if encoder:
                    # Handle unseen categories
                    unique_values = set(encoder.classes_)
                    df[col] = df[col].apply(lambda x: x if x in unique_values else 'unknown')
                    df[col] = encoder.transform(df[col].astype(str))
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Competition-specific feature engineering"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) >= 2:
            # Create interaction features for top features
            for i, col1 in enumerate(numerical_cols[:5]):
                for col2 in numerical_cols[i+1:6]:
                    df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                    df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
        
        # Statistical features
        if len(numerical_cols) > 0:
            df['numerical_mean'] = df[numerical_cols].mean(axis=1)
            df['numerical_std'] = df[numerical_cols].std(axis=1)
            df['numerical_max'] = df[numerical_cols].max(axis=1)
            df['numerical_min'] = df[numerical_cols].min(axis=1)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            if fit:
                scaler = RobustScaler()  # More robust to outliers
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                self.scalers['numerical_scaler'] = scaler
            else:
                scaler = self.scalers.get('numerical_scaler')
                if scaler:
                    df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        return df
    
    def _select_features(self, df: pd.DataFrame, y=None, fit: bool = False) -> pd.DataFrame:
        """Intelligent feature selection"""
        
        if fit and y is not None:
            # Select top features based on statistical tests
            selector = SelectKBest(
                score_func=f_classif if y.dtype == 'object' or len(y.unique()) < 20 else f_regression,
                k=min(100, len(df.columns))  # Select top 100 features or all if less
            )
            df_selected = pd.DataFrame(
                selector.fit_transform(df, y),
                columns=df.columns[selector.get_support()],
                index=df.index
            )
            self.feature_selectors['feature_selector'] = selector
            return df_selected
        elif not fit and 'feature_selector' in self.feature_selectors:
            selector = self.feature_selectors['feature_selector']
            df_selected = pd.DataFrame(
                selector.transform(df),
                columns=df.columns[selector.get_support()],
                index=df.index
            )
            return df_selected
        
        return df

# Global preprocessor instance
intelligent_preprocessor = IntelligentPreprocessor()
`;
  }

  private generateRecommendedModels(intelligence: CompetitionIntelligence): string {
    return `"""
AI-recommended models based on competition analysis and winning approaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelRecommendationEngine:
    """AI-powered model recommendation system"""
    
    def __init__(self):
        self.competition_type = "${intelligence.competitionType}"
        self.target_type = "${intelligence.datasetCharacteristics.targetType}"
        self.winning_approaches = ${JSON.stringify(intelligence.winningApproaches)}
        self.recommended_models = {}
    
    def get_recommended_models(self) -> Dict[str, Any]:
        """Get AI-recommended models for this competition"""
        
        if self.target_type == 'classification':
            return self._get_classification_models()
        elif self.target_type == 'regression':
            return self._get_regression_models()
        else:
            return self._get_default_models()
    
    def _get_classification_models(self) -> Dict[str, Any]:
        """Get classification models optimized for this competition"""
        
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }
        
        return models
    
    def _get_regression_models(self) -> Dict[str, Any]:
        """Get regression models optimized for this competition"""
        
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        return models
    
    def _get_default_models(self) -> Dict[str, Any]:
        """Get default models when competition type is unclear"""
        return self._get_classification_models()

# Global model recommendation engine
model_engine = ModelRecommendationEngine()
`;
  }

  private generateEvaluationStrategies(intelligence: CompetitionIntelligence): string {
    return `"""
Intelligent evaluation strategies based on competition characteristics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

class IntelligentEvaluator:
    """AI-optimized evaluation strategies"""
    
    def __init__(self):
        self.competition_type = "${intelligence.competitionType}"
        self.target_type = "${intelligence.datasetCharacteristics.targetType}"
        self.cv_strategy = "${this.getCVStrategy(intelligence)}"
    
    def get_cv_strategy(self, n_splits: int = 5):
        """Get the optimal cross-validation strategy"""
        
        if self.cv_strategy == 'stratified':
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif self.cv_strategy == 'time-series':
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Global evaluator instance
intelligent_evaluator = IntelligentEvaluator()
`;
  }

  private getCVStrategy(intelligence: CompetitionIntelligence): string {
    if (intelligence.competitionType === 'time-series') {
      return 'time-series';
    } else if (intelligence.datasetCharacteristics.targetType === 'classification') {
      return 'stratified';
    } else {
      return 'simple';
    }
  }
}