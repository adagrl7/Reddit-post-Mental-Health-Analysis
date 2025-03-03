import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class AdvancedAnalyzer:
    def __init__(self):
        self.output_dir = 'data/processed/advanced_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
    
    def load_cleaned_data(self):
        processed_files = [f for f in os.listdir('data/processed') if f.startswith('cleaned_data')]
        latest_file = sorted(processed_files)[-1]
        return pd.read_csv(f'data/processed/{latest_file}')
    
    def perform_statistical_analysis(self, df):
        """Perform detailed statistical analysis"""
        stats_results = {}
        
        # ANOVA test for sentiment across subreddits
        subreddits = df['subreddit'].unique()
        sentiment_groups = [df[df['subreddit'] == sub]['sentiment'] for sub in subreddits]
        f_stat, p_value = stats.f_oneway(*sentiment_groups)
        
        # Correlation analysis
        correlation_matrix = df[['sentiment', 'score', 'num_comments', 'word_count']].corr()
        
        # Chi-square test for post timing
        observed = pd.crosstab(df['day_of_week'], df['subreddit'])
        chi2, p_val = stats.chi2_contingency(observed)[:2]
        
        stats_results = {
            'sentiment_anova': {'f_stat': f_stat, 'p_value': p_value},
            'correlation_matrix': correlation_matrix,
            'timing_chi2': {'chi2': chi2, 'p_value': p_val}
        }
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/correlation_matrix.png')
        plt.close()
        
        return stats_results
    
    def detect_patterns(self, df):
        """Detect patterns using clustering and PCA"""
        # Prepare features for clustering
        features = ['sentiment', 'score', 'num_comments', 'word_count', 'upvote_ratio']
        X = df[features]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add results to dataframe
        df_patterns = df.copy()
        df_patterns['cluster'] = clusters
        df_patterns['pca_1'] = X_pca[:, 0]
        df_patterns['pca_2'] = X_pca[:, 1]
        
        # Visualize clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        plt.title('Post Clusters based on Features')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        plt.savefig(f'{self.output_dir}/plots/clusters.png')
        plt.close()
        
        return df_patterns
    
    def build_ml_models(self, df):
        """Build and evaluate ML models"""
        # Prepare features for sentiment prediction
        vectorizer = TfidfVectorizer(max_features=1000)
        X_text = vectorizer.fit_transform(df['cleaned_text'])
        
        # Create sentiment categories
        df['sentiment_category'] = pd.qcut(df['sentiment'], q=3, labels=['negative', 'neutral', 'positive'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, df['sentiment_category'], random_state=42
        )
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model_performance': {
                'train_score': train_score,
                'test_score': test_score
            },
            'important_features': feature_importance.head(20)
        }
    
    def perform_time_series_analysis(self, df):
        """Perform time series analysis"""
        # Prepare time series data
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        daily_sentiment = df.groupby(df['created_utc'].dt.date)['sentiment'].mean()
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(daily_sentiment, period=7)
        
        # Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/time_series_decomposition.png')
        plt.close()
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AdvancedAnalyzer()
    
    # Load data
    print("Loading cleaned data...")
    df = analyzer.load_cleaned_data()
    
    # Perform analyses
    print("\nPerforming statistical analysis...")
    stats_results = analyzer.perform_statistical_analysis(df)
    
    print("Detecting patterns...")
    pattern_results = analyzer.detect_patterns(df)
    
    print("Building ML models...")
    ml_results = analyzer.build_ml_models(df)
    
    print("Performing time series analysis...")
    ts_results = analyzer.perform_time_series_analysis(df)
    
    # Print key findings
    print("\nKey Findings:")
    print("-" * 50)
    print(f"Statistical Analysis:")
    print(f"- ANOVA p-value for sentiment across subreddits: {stats_results['sentiment_anova']['p_value']:.4f}")
    print(f"\nML Model Performance:")
    print(f"- Training accuracy: {ml_results['model_performance']['train_score']:.2f}")
    print(f"- Testing accuracy: {ml_results['model_performance']['test_score']:.2f}")