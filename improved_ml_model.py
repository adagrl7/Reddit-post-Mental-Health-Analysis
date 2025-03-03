import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
import joblib

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select specific columns"""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Make sure all requested columns exist
            available_columns = [col for col in self.columns if col in X.columns]
            return X[available_columns]
        else:
            # If X is already a numpy array, return it unchanged
            return X

class ImprovedMLAnalyzer:
    def __init__(self):
        self.output_dir = 'data/processed/ml_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.output_dir}/models', exist_ok=True)
    
    def prepare_data(self, df):
        """Enhanced data preparation with sophisticated features"""
        # Create balanced sentiment categories
        df['sentiment_category'] = pd.qcut(df['sentiment'], q=3, 
                                         labels=['negative', 'neutral', 'positive'])
        
        # Handle missing values
        text_cols = ['title', 'text', 'cleaned_text']
        df[text_cols] = df[text_cols].fillna('')
        
        # Feature engineering
        df = df.assign(
            title_length=df['title'].str.len().fillna(0),
            text_length=df['text'].str.len().fillna(0),
            hour_of_day=pd.to_datetime(df['created_utc'], errors='coerce')
                         .dt.hour.fillna(0).astype(int),
            sentiment_intensity=df['sentiment'].abs(),
            question_mark=df['text'].str.contains(r'\?', na=False).astype(int),
            exclamation_mark=df['text'].str.contains(r'\!', na=False).astype(int),
            flesch_reading=df['text'].apply(lambda x: textstat.flesch_reading_ease(x) 
                                          if isinstance(x, str) and x.strip() else 0)
        )
        
        # Fill remaining numeric columns
        numeric_cols = ['score', 'num_comments']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Drop rows with missing target
        return df.dropna(subset=['sentiment_category']).reset_index(drop=True)
    
    def build_improved_model(self, df):
        """Enhanced modeling pipeline with proper feature handling and tuning"""
        # Prepare data and encode labels
        df = self.prepare_data(df)
        le = LabelEncoder()
        y = le.fit_transform(df['sentiment_category'])
        
        # Define column lists explicitly
        numeric_columns = [
            'title_length',
            'text_length',
            'sentiment_intensity',
            'flesch_reading',
            'score',
            'num_comments',
            'hour_of_day'
        ]
        
        bool_columns = [
            'question_mark',
            'exclamation_mark'
        ]
        
        # Create full preprocessing pipeline with explicit column names
        preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.9,
                sublinear_tf=True
            ), 'cleaned_text'),
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_columns),
            ('bool', SimpleImputer(strategy='most_frequent'), bool_columns)
        ], remainder='drop')  # Drop any columns not explicitly specified
        
        # Define models with hyperparameter grids
        models = {
            'random_forest': {
                'model': RandomForestClassifier(class_weight='balanced', random_state=42),
                'params': {
                    'max_depth': [5, 8, None],
                    'n_estimators': [100, 200],
                    'min_samples_split': [2, 5]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['saga'],
                    'penalty': ['elasticnet'],
                    'l1_ratio': [0.4, 0.5, 0.6]
                }
            },
            'linear_svc': {
                'model': LinearSVC(class_weight='balanced', random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'loss': ['squared_hinge'],
                    'dual': [False]
                }
            },
            'xgboost': {
                'model': XGBClassifier(objective='multi:softmax', random_state=42),
                'params': {
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        results = {}
        
        for name, config in models.items():
            print(f"\nTraining {name}...")
            try:
                # Create full pipeline with SMOTE
                pipe = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
                    (name, config['model'])  # This correctly names the final estimator
                ])
                
                # Set up parameter grid with proper name prefixes
                param_grid = {
                    f'{name}__{param}': value 
                    for param, value in config['params'].items()
                }
                
                # Randomized search
                search = RandomizedSearchCV(
                    pipe,
                    param_grid,
                    n_iter=min(5, len(param_grid)),  # Ensure n_iter doesn't exceed parameter space
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42
                )
                
                # Split data - select only relevant columns
                relevant_columns = ['cleaned_text'] + numeric_columns + bool_columns
                X_train, X_test, y_train, y_test = train_test_split(
                    df[relevant_columns], y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Train model
                search.fit(X_train, y_train)
                
                # Store results
                results[name] = {
                    'best_params': search.best_params_,
                    'test_score': search.score(X_test, y_test),
                    'cv_results': search.cv_results_,
                    'report': classification_report(y_test, search.predict(X_test), 
                                                 target_names=le.classes_),
                    'model': search.best_estimator_
                }
                
                # Save model
                joblib.dump(search.best_estimator_, 
                          f'{self.output_dir}/models/{name}_best.pkl')
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix(y_test, search.predict(X_test)),
                           annot=True, fmt='d', cmap='Blues',
                           xticklabels=le.classes_, 
                           yticklabels=le.classes_)
                plt.title(f'Confusion Matrix - {name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.savefig(f'{self.output_dir}/plots/confusion_matrix_{name}.png')
                plt.close()
                
            except Exception as e:
                print(f"Error in {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Find best model
        valid_models = {k: v for k, v in results.items() if 'test_score' in v}
        best_model = max(valid_models.items(), 
                        key=lambda x: x[1]['test_score']) if valid_models else None
        
        return results, best_model

if __name__ == "__main__":
    try:
        processed_dir = 'data/processed'
        
        # Safely find the most recent cleaned data
        csv_files = [f for f in os.listdir(processed_dir) 
                    if f.startswith('cleaned_data') and f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No cleaned data files found in the processed directory")
            
        most_recent_file = max(csv_files)
        print(f"Loading data from {most_recent_file}")
        
        df = pd.read_csv(f"{processed_dir}/{most_recent_file}")
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check that required columns exist
        required_cols = ['title', 'text', 'cleaned_text', 'sentiment', 'score', 'num_comments', 'created_utc']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        analyzer = ImprovedMLAnalyzer()
        print("Starting model training...")
        results, best_model = analyzer.build_improved_model(df)
        
        print("\nModel Performance Summary:")
        print("-" * 50)
        for name, res in results.items():
            if 'error' in res:
                print(f"{name.upper()} failed: {res['error']}")
                continue
            print(f"\n{name.upper():<20} Accuracy: {res['test_score']:.3f}")
            print(f"Best params: {res['best_params']}")
        
        if best_model:
            print(f"\nBest model: {best_model[0]} (Accuracy: {best_model[1]['test_score']:.3f})")
            print("\nClassification Report:")
            print(best_model[1]['report'])
        else:
            print("\nNo models were successfully trained. Please check the errors above.")
            
    except Exception as e:
        print(f"Error: {str(e)}")