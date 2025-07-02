export interface CompetitionData {
  name: string;
  title: string;
  description: string;
  type: 'classification' | 'regression' | 'nlp' | 'computer-vision' | 'other';
  dataFiles: string[];
}

export interface ProjectOptions {
  includeEDA: boolean;
  includeBaseline: boolean;
  initializeGit: boolean;
}

export interface GeneratedFile {
  path: string;
  content: string;
}

export class ProjectGenerator {
  private static detectCompetitionType(name: string): CompetitionData['type'] {
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
  }

  static async generateProject(
    competition: string, 
    options: ProjectOptions
  ): Promise<GeneratedFile[]> {
    const competitionName = this.extractCompetitionName(competition);
    const competitionData = await this.fetchCompetitionData(competitionName);
    
    const files: GeneratedFile[] = [];
    
    // Generate README
    files.push({
      path: 'README.md',
      content: this.generateReadme(competitionData)
    });

    // Generate requirements.txt
    files.push({
      path: 'requirements.txt',
      content: this.generateRequirements(competitionData.type)
    });

    // Generate .gitignore
    files.push({
      path: '.gitignore',
      content: this.generateGitignore()
    });

    // Generate data preprocessing script
    files.push({
      path: 'src/data_preprocessing.py',
      content: this.generateDataPreprocessing(competitionData)
    });

    // Generate utility functions
    files.push({
      path: 'src/utils.py',
      content: this.generateUtils(competitionData.type)
    });

    if (options.includeEDA) {
      files.push({
        path: 'notebooks/eda.ipynb',
        content: this.generateEDANotebook(competitionData)
      });
    }

    if (options.includeBaseline) {
      files.push({
        path: 'notebooks/baseline.ipynb',
        content: this.generateBaselineNotebook(competitionData)
      });
      
      files.push({
        path: 'src/model.py',
        content: this.generateModelScript(competitionData)
      });
    }

    // Generate submission script
    files.push({
      path: 'src/submission.py',
      content: this.generateSubmissionScript(competitionData)
    });

    return files;
  }

  private static extractCompetitionName(input: string): string {
    // Extract competition name from URL or use as-is
    const urlMatch = input.match(/kaggle\.com\/competitions\/([^\/\?]+)/);
    return urlMatch ? urlMatch[1] : input.trim();
  }

  private static async fetchCompetitionData(name: string): Promise<CompetitionData> {
    // Since we can't access real Kaggle API, we'll use intelligent defaults
    // based on competition name patterns
    const type = this.detectCompetitionType(name);
    
    return {
      name,
      title: this.formatTitle(name),
      description: `Machine learning competition: ${this.formatTitle(name)}`,
      type,
      dataFiles: this.getDefaultDataFiles(type)
    };
  }

  private static formatTitle(name: string): string {
    return name
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  private static getDefaultDataFiles(type: CompetitionData['type']): string[] {
    const baseFiles = ['train.csv', 'test.csv', 'sample_submission.csv'];
    
    switch (type) {
      case 'computer-vision':
        return [...baseFiles, 'train_images/', 'test_images/'];
      case 'nlp':
        return [...baseFiles, 'train.txt', 'test.txt'];
      default:
        return baseFiles;
    }
  }

  private static generateReadme(competition: CompetitionData): string {
    return `# ${competition.title}

## Competition Overview
${competition.description}

## Project Structure
\`\`\`
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── utils.py
│   └── submission.py
├── data/              # Competition data (download separately)
├── submissions/       # Generated submission files
├── requirements.txt   # Python dependencies
└── README.md         # This file
\`\`\`

## Getting Started

1. **Install Dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Download Data**
   - Visit the [competition page](https://www.kaggle.com/competitions/${competition.name})
   - Download the dataset files to the \`data/\` directory

3. **Run EDA**
   \`\`\`bash
   jupyter notebook notebooks/eda.ipynb
   \`\`\`

4. **Train Baseline Model**
   \`\`\`bash
   python src/model.py
   \`\`\`

5. **Generate Submission**
   \`\`\`bash
   python src/submission.py
   \`\`\`

## Data Files
${competition.dataFiles.map(file => `- ${file}`).join('\n')}

## Next Steps
- [ ] Explore the data in the EDA notebook
- [ ] Feature engineering
- [ ] Model experimentation
- [ ] Cross-validation setup
- [ ] Hyperparameter tuning
- [ ] Ensemble methods

## Resources
- [Competition Page](https://www.kaggle.com/competitions/${competition.name})
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Competition Discussion](https://www.kaggle.com/competitions/${competition.name}/discussion)
`;
  }

  private static generateRequirements(type: CompetitionData['type']): string {
    const basePackages = [
      'pandas>=1.5.0',
      'numpy>=1.21.0',
      'scikit-learn>=1.1.0',
      'matplotlib>=3.5.0',
      'seaborn>=0.11.0',
      'jupyter>=1.0.0',
      'plotly>=5.0.0'
    ];

    const typeSpecificPackages: Record<CompetitionData['type'], string[]> = {
      'classification': ['xgboost>=1.6.0', 'lightgbm>=3.3.0'],
      'regression': ['xgboost>=1.6.0', 'lightgbm>=3.3.0'],
      'nlp': ['transformers>=4.20.0', 'torch>=1.12.0', 'nltk>=3.7'],
      'computer-vision': ['torch>=1.12.0', 'torchvision>=0.13.0', 'opencv-python>=4.6.0', 'pillow>=9.0.0'],
      'other': ['xgboost>=1.6.0']
    };

    return [...basePackages, ...typeSpecificPackages[type]].join('\n');
  }

  private static generateGitignore(): string {
    return `# Data files
data/
*.csv
*.json
*.parquet

# Model files
*.pkl
*.joblib
*.h5
*.pth

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Submissions
submissions/*.csv
!submissions/.gitkeep
`;
  }

  private static generateDataPreprocessing(competition: CompetitionData): string {
    return `"""
Data preprocessing utilities for ${competition.title}
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def load_data(self, train_path='data/train.csv', test_path='data/test.csv'):
        """Load training and test datasets"""
        print("Loading data...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def basic_info(self):
        """Display basic information about the datasets"""
        print("\\n=== TRAINING DATA INFO ===")
        print(self.train_df.info())
        print("\\n=== MISSING VALUES ===")
        print(self.train_df.isnull().sum().sort_values(ascending=False))
        print("\\n=== BASIC STATISTICS ===")
        print(self.train_df.describe())
    
    def handle_missing_values(self, strategy='median'):
        """Handle missing values in the dataset"""
        numeric_columns = self.train_df.select_dtypes(include=[np.number]).columns
        categorical_columns = self.train_df.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        for col in numeric_columns:
            if self.train_df[col].isnull().sum() > 0:
                if strategy == 'median':
                    fill_value = self.train_df[col].median()
                elif strategy == 'mean':
                    fill_value = self.train_df[col].mean()
                else:
                    fill_value = 0
                
                self.train_df[col].fillna(fill_value, inplace=True)
                if col in self.test_df.columns:
                    self.test_df[col].fillna(fill_value, inplace=True)
        
        # Handle categorical missing values
        for col in categorical_columns:
            if self.train_df[col].isnull().sum() > 0:
                mode_value = self.train_df[col].mode()[0] if len(self.train_df[col].mode()) > 0 else 'Unknown'
                self.train_df[col].fillna(mode_value, inplace=True)
                if col in self.test_df.columns:
                    self.test_df[col].fillna(mode_value, inplace=True)
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        categorical_columns = self.train_df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'target':  # Assuming 'target' is the target column name
                le = LabelEncoder()
                # Fit on combined data to ensure consistent encoding
                combined_values = pd.concat([self.train_df[col], self.test_df[col] if col in self.test_df.columns else pd.Series()])
                le.fit(combined_values.astype(str))
                
                self.train_df[col] = le.transform(self.train_df[col].astype(str))
                if col in self.test_df.columns:
                    self.test_df[col] = le.transform(self.test_df[col].astype(str))
                
                self.encoders[col] = le
    
    def create_features(self):
        """Create additional features"""
        # Add your feature engineering here
        # Example: interaction features, polynomial features, etc.
        pass
    
    def prepare_features(self, target_column=None):
        """Prepare final feature matrices"""
        # Identify target column if not specified
        if target_column is None:
            possible_targets = ['target', 'label', 'y', 'outcome']
            for col in possible_targets:
                if col in self.train_df.columns:
                    target_column = col
                    break
        
        if target_column and target_column in self.train_df.columns:
            X = self.train_df.drop(columns=[target_column])
            y = self.train_df[target_column]
        else:
            X = self.train_df
            y = None
        
        X_test = self.test_df
        
        # Ensure same columns in train and test
        common_columns = list(set(X.columns) & set(X_test.columns))
        X = X[common_columns]
        X_test = X_test[common_columns]
        
        self.feature_names = X.columns.tolist()
        
        return X, y, X_test

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data()
    preprocessor.basic_info()
    preprocessor.handle_missing_values()
    preprocessor.encode_categorical_features()
    preprocessor.create_features()
    X, y, X_test = preprocessor.prepare_features()
    
    print(f"\\nFinal feature matrix shape: {X.shape}")
    print(f"Test matrix shape: {X_test.shape}")
`;
  }

  private static generateUtils(type: CompetitionData['type']): string {
    return `"""
Utility functions for ${type} competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        print("Model doesn't have feature importance")
        return
    
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_train, y_train, X_val, y_val, task_type='${type}'):
    """Evaluate model performance"""
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    print("=== MODEL EVALUATION ===")
    
    if task_type in ['classification']:
        print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
        print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")
        
        print("\\nClassification Report:")
        print(classification_report(y_val, val_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, val_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    elif task_type in ['regression']:
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Training MAE: {mean_absolute_error(y_train, train_pred):.4f}")
        print(f"Validation MAE: {mean_absolute_error(y_val, val_pred):.4f}")
        print(f"Validation R²: {r2_score(y_val, val_pred):.4f}")
        
        # Residual plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(val_pred, y_val, alpha=0.6)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Predicted vs Actual')
        
        plt.subplot(1, 2, 2)
        residuals = y_val - val_pred
        plt.scatter(val_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()

def cross_validate_model(model, X, y, cv=5, scoring=None):
    """Perform cross-validation"""
    if scoring is None:
        scoring = 'accuracy' if '${type}' == 'classification' else 'neg_mean_squared_error'
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    print(f"\\n=== {cv}-FOLD CROSS VALIDATION ===")
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")
    
    return scores

def create_submission(predictions, test_ids, filename='submission.csv'):
    """Create submission file"""
    submission = pd.DataFrame({
        'id': test_ids,
        'target': predictions
    })
    
    submission.to_csv(f'submissions/{filename}', index=False)
    print(f"Submission saved to submissions/{filename}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves to diagnose bias/variance"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
`;
  }

  private static generateEDANotebook(competition: CompetitionData): string {
    const notebookContent = {
      cells: [
        {
          cell_type: "markdown",
          metadata: {},
          source: [
            `# ${competition.title} - Exploratory Data Analysis\n`,
            "\n",
            "This notebook provides a comprehensive exploratory data analysis for the competition.\n"
          ]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Import libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Set style\n",
            "plt.style.use('seaborn-v0_8')\n",
            "sns.set_palette('husl')\n",
            "\n",
            "# Display options\n",
            "pd.set_option('display.max_columns', None)\n",
            "pd.set_option('display.max_rows', 100)"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 1. Data Loading and Overview"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Load data\n",
            "train_df = pd.read_csv('../data/train.csv')\n",
            "test_df = pd.read_csv('../data/test.csv')\n",
            "\n",
            "print(f\"Training data shape: {train_df.shape}\")\n",
            "print(f\"Test data shape: {test_df.shape}\")\n",
            "\n",
            "# Display first few rows\n",
            "train_df.head()"
          ]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Data info\n",
            "print(\"=== TRAINING DATA INFO ===\")\n",
            "train_df.info()\n",
            "\n",
            "print(\"\\n=== MISSING VALUES ===\")\n",
            "missing_data = train_df.isnull().sum().sort_values(ascending=False)\n",
            "missing_percent = (missing_data / len(train_df)) * 100\n",
            "missing_df = pd.DataFrame({'Missing Count': missing_data, 'Percentage': missing_percent})\n",
            "print(missing_df[missing_df['Missing Count'] > 0])"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 2. Target Variable Analysis"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Identify target column (adjust as needed)\n",
            "target_col = None\n",
            "possible_targets = ['target', 'label', 'y', 'outcome', 'Survived', 'SalePrice']\n",
            "for col in possible_targets:\n",
            "    if col in train_df.columns:\n",
            "        target_col = col\n",
            "        break\n",
            "\n",
            "if target_col:\n",
            "    print(f\"Target column: {target_col}\")\n",
            "    print(f\"Target distribution:\")\n",
            "    print(train_df[target_col].value_counts())\n",
            "    \n",
            "    # Plot target distribution\n",
            "    plt.figure(figsize=(12, 4))\n",
            "    \n",
            "    plt.subplot(1, 2, 1)\n",
            "    train_df[target_col].hist(bins=30, edgecolor='black')\n",
            "    plt.title(f'{target_col} Distribution')\n",
            "    plt.xlabel(target_col)\n",
            "    plt.ylabel('Frequency')\n",
            "    \n",
            "    plt.subplot(1, 2, 2)\n",
            "    train_df[target_col].plot(kind='box')\n",
            "    plt.title(f'{target_col} Box Plot')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Target column not automatically identified. Please specify manually.\")"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 3. Feature Analysis"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Separate numeric and categorical features\n",
            "numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()\n",
            "categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()\n",
            "\n",
            "if target_col and target_col in numeric_features:\n",
            "    numeric_features.remove(target_col)\n",
            "\n",
            "print(f\"Numeric features ({len(numeric_features)}): {numeric_features}\")\n",
            "print(f\"Categorical features ({len(categorical_features)}): {categorical_features}\")"
          ]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Numeric features correlation\n",
            "if len(numeric_features) > 1:\n",
            "    plt.figure(figsize=(12, 10))\n",
            "    correlation_matrix = train_df[numeric_features + ([target_col] if target_col else [])].corr()\n",
            "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')\n",
            "    plt.title('Feature Correlation Matrix')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    # High correlation pairs\n",
            "    if target_col:\n",
            "        target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)\n",
            "        print(f\"\\nTop features correlated with {target_col}:\")\n",
            "        print(target_corr.head(10))"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 4. Data Quality Assessment"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Check for duplicates\n",
            "print(f\"Duplicate rows in training data: {train_df.duplicated().sum()}\")\n",
            "print(f\"Duplicate rows in test data: {test_df.duplicated().sum()}\")\n",
            "\n",
            "# Check data types\n",
            "print(\"\\nData types:\")\n",
            "print(train_df.dtypes.value_counts())\n",
            "\n",
            "# Basic statistics\n",
            "print(\"\\nBasic statistics for numeric features:\")\n",
            "train_df[numeric_features].describe()"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 5. Feature Distributions"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Plot numeric feature distributions\n",
            "if len(numeric_features) > 0:\n",
            "    n_cols = 3\n",
            "    n_rows = (len(numeric_features) + n_cols - 1) // n_cols\n",
            "    \n",
            "    plt.figure(figsize=(15, 4 * n_rows))\n",
            "    \n",
            "    for i, feature in enumerate(numeric_features[:12]):  # Limit to first 12\n",
            "        plt.subplot(n_rows, n_cols, i + 1)\n",
            "        train_df[feature].hist(bins=30, edgecolor='black', alpha=0.7)\n",
            "        plt.title(f'{feature} Distribution')\n",
            "        plt.xlabel(feature)\n",
            "        plt.ylabel('Frequency')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()"
          ]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Plot categorical feature distributions\n",
            "if len(categorical_features) > 0:\n",
            "    n_cols = 2\n",
            "    n_rows = (len(categorical_features) + n_cols - 1) // n_cols\n",
            "    \n",
            "    plt.figure(figsize=(15, 4 * n_rows))\n",
            "    \n",
            "    for i, feature in enumerate(categorical_features[:8]):  # Limit to first 8\n",
            "        plt.subplot(n_rows, n_cols, i + 1)\n",
            "        value_counts = train_df[feature].value_counts().head(10)\n",
            "        value_counts.plot(kind='bar')\n",
            "        plt.title(f'{feature} Distribution')\n",
            "        plt.xlabel(feature)\n",
            "        plt.ylabel('Count')\n",
            "        plt.xticks(rotation=45)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 6. Key Insights and Next Steps"]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: [
            "### Key Findings:\n",
            "- Add your observations here after running the analysis\n",
            "- Note any data quality issues\n",
            "- Identify important features\n",
            "- Note any patterns or anomalies\n",
            "\n",
            "### Next Steps:\n",
            "1. Feature engineering based on insights\n",
            "2. Handle missing values appropriately\n",
            "3. Consider feature scaling/normalization\n",
            "4. Experiment with different models\n",
            "5. Set up proper cross-validation"
          ]
        }
      ],
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3"
        },
        language_info: {
          codemirror_mode: { name: "ipython", version: 3 },
          file_extension: ".py",
          mimetype: "text/x-python",
          name: "python",
          nbconvert_exporter: "python",
          pygments_lexer: "ipython3",
          version: "3.8.0"
        }
      },
      nbformat: 4,
      nbformat_minor: 4
    };

    return JSON.stringify(notebookContent, null, 2);
  }

  private static generateBaselineNotebook(competition: CompetitionData): string {
    const modelCode = competition.type === 'regression' 
      ? "from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error\n\n# Initialize model\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)"
      : "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, classification_report\n\n# Initialize model\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)";

    const evaluationCode = competition.type === 'regression'
      ? "# Evaluate model\ntrain_pred = model.predict(X_train)\nval_pred = model.predict(X_val)\n\nprint(f'Training RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}')\nprint(f'Validation RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}')"
      : "# Evaluate model\ntrain_pred = model.predict(X_train)\nval_pred = model.predict(X_val)\n\nprint(f'Training Accuracy: {accuracy_score(y_train, train_pred):.4f}')\nprint(f'Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}')\nprint('\\nClassification Report:')\nprint(classification_report(y_val, val_pred))";

    const notebookContent = {
      cells: [
        {
          cell_type: "markdown",
          metadata: {},
          source: [`# ${competition.title} - Baseline Model\n\nThis notebook implements a baseline model for the competition.`]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Import libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import sys\n",
            "sys.path.append('../src')\n",
            "\n",
            "from data_preprocessing import DataPreprocessor\n",
            "from utils import evaluate_model, plot_feature_importance, create_submission\n",
            modelCode,
            "from sklearn.model_selection import train_test_split\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 1. Data Loading and Preprocessing"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Initialize preprocessor\n",
            "preprocessor = DataPreprocessor()\n",
            "\n",
            "# Load and preprocess data\n",
            "train_df, test_df = preprocessor.load_data()\n",
            "preprocessor.handle_missing_values()\n",
            "preprocessor.encode_categorical_features()\n",
            "preprocessor.create_features()\n",
            "\n",
            "# Prepare features\n",
            "X, y, X_test = preprocessor.prepare_features()\n",
            "\n",
            "print(f\"Feature matrix shape: {X.shape}\")\n",
            "print(f\"Target shape: {y.shape if y is not None else 'No target found'}\")\n",
            "print(f\"Test matrix shape: {X_test.shape}\")"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 2. Train-Validation Split"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Split data\n",
            "X_train, X_val, y_train, y_val = train_test_split(\n",
            "    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 20 else None\n",
            ")\n",
            "\n",
            "print(f\"Training set: {X_train.shape}\")\n",
            "print(f\"Validation set: {X_val.shape}\")"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 3. Model Training"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Train model\n",
            "print(\"Training baseline model...\")\n",
            "model.fit(X_train, y_train)\n",
            "print(\"Training completed!\")"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 4. Model Evaluation"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [evaluationCode]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Feature importance\n",
            "plot_feature_importance(model, X.columns, top_n=15)"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 5. Generate Predictions"]
        },
        {
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [
            "# Generate test predictions\n",
            "test_predictions = model.predict(X_test)\n",
            "\n",
            "# Create submission file\n",
            "test_ids = test_df.iloc[:, 0]  # Assuming first column is ID\n",
            "create_submission(test_predictions, test_ids, 'baseline_submission.csv')"
          ]
        },
        {
          cell_type: "markdown",
          metadata: {},
          source: ["## 6. Next Steps\n\n- Try different algorithms (XGBoost, LightGBM)\n- Hyperparameter tuning\n- Feature engineering\n- Cross-validation\n- Ensemble methods"]
        }
      ],
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3"
        },
        language_info: {
          name: "python",
          version: "3.8.0"
        }
      },
      nbformat: 4,
      nbformat_minor: 4
    };

    return JSON.stringify(notebookContent, null, 2);
  }

  private static generateModelScript(competition: CompetitionData): string {
    const modelImports = competition.type === 'nlp' 
      ? "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression"
      : competition.type === 'computer-vision'
      ? "import torch\nimport torch.nn as nn\nfrom torchvision import transforms"
      : "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\nfrom xgboost import XGBClassifier, XGBRegressor";

    return `"""
Model training script for ${competition.title}
"""

import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from utils import evaluate_model, cross_validate_model, plot_feature_importance
${modelImports}
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_names = []
    
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        
        # Load data
        train_df, test_df = preprocessor.load_data()
        
        # Preprocess
        preprocessor.handle_missing_values()
        preprocessor.encode_categorical_features()
        preprocessor.create_features()
        
        # Prepare features
        X, y, X_test = preprocessor.prepare_features()
        self.feature_names = X.columns.tolist()
        
        return X, y, X_test
    
    def train_baseline_model(self, X, y):
        """Train a baseline model"""
        print("Training baseline model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model based on competition type
        if '${competition.type}' == 'regression':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        evaluate_model(self.model, X_train, y_train, X_val, y_val, '${competition.type}')
        
        return X_train, X_val, y_train, y_val
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning"""
        print("Performing hyperparameter tuning...")
        
        if '${competition.type}' == 'regression':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            scoring = 'neg_mean_squared_error'
        else:
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            scoring = 'accuracy'
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def save_model(self, filename='trained_model.pkl'):
        """Save the trained model"""
        joblib.dump(self.model, f'../models/{filename}')
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='trained_model.pkl'):
        """Load a trained model"""
        self.model = joblib.load(f'../models/{filename}')
        print(f"Model loaded from {filename}")
        return self.model

def main():
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load and preprocess data
    X, y, X_test = trainer.load_and_preprocess_data()
    
    # Train baseline model
    X_train, X_val, y_train, y_val = trainer.train_baseline_model(X, y)
    
    # Cross-validation
    cross_validate_model(trainer.model, X, y)
    
    # Feature importance
    plot_feature_importance(trainer.model, trainer.feature_names)
    
    # Hyperparameter tuning (optional - can be time-consuming)
    # trainer.hyperparameter_tuning(X, y)
    
    # Save model
    trainer.save_model()
    
    print("\\nModel training completed!")

if __name__ == "__main__":
    main()
`;
  }

  private static generateSubmissionScript(competition: CompetitionData): string {
    return `"""
Generate submission file for ${competition.title}
"""

import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from utils import create_submission
import joblib
import warnings
warnings.filterwarnings('ignore')

def generate_submission(model_path='../models/trained_model.pkl'):
    """Generate submission file"""
    print("Generating submission...")
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data()
    
    # Apply same preprocessing as training
    preprocessor.handle_missing_values()
    preprocessor.encode_categorical_features()
    preprocessor.create_features()
    
    # Prepare features
    X, y, X_test = preprocessor.prepare_features()
    
    # Load trained model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train a model first.")
        return
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Get test IDs (assuming first column is ID)
    test_ids = test_df.iloc[:, 0]
    
    # Create submission file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f'submission_{timestamp}.csv'
    
    create_submission(predictions, test_ids, filename)
    
    print(f"\\nSubmission file created: submissions/{filename}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")

if __name__ == "__main__":
    generate_submission()
`;
  }
}