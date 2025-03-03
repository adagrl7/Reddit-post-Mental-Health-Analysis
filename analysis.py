import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

class RedditAnalyzer:
    def __init__(self):
        self.output_dir = 'data/processed/analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
        
    def load_cleaned_data(self):
        """Load the most recent cleaned data file"""
        processed_files = [f for f in os.listdir('data/processed') if f.startswith('cleaned_data')]
        latest_file = sorted(processed_files)[-1]
        return pd.read_csv(f'data/processed/{latest_file}')
    
    def analyze_sentiment_trends(self, df):
        """Analyze sentiment patterns across different dimensions"""
        # Sentiment by subreddit
        sent_by_sub = df.groupby('subreddit')['sentiment'].agg(['mean', 'std']).round(3)
        
        # Sentiment by day of week
        sent_by_day = df.groupby('day_of_week')['sentiment'].mean().round(3)
        
        # Sentiment by hour
        sent_by_hour = df.groupby('hour')['sentiment'].mean().round(3)
        
        # Create visualizations
        fig = go.Figure()
        
        # Add sentiment by hour trace
        fig.add_trace(go.Scatter(
            x=sent_by_hour.index,
            y=sent_by_hour.values,
            name='Hourly Sentiment',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Sentiment Analysis by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Average Sentiment'
        )
        
        fig.write_html(f'{self.output_dir}/plots/sentiment_by_hour.html')
        
        return {
            'sentiment_by_subreddit': sent_by_sub,
            'sentiment_by_day': sent_by_day,
            'sentiment_by_hour': sent_by_hour
        }
    
    def analyze_engagement(self, df):
        """Analyze user engagement patterns"""
        # Engagement metrics by subreddit
        engagement = df.groupby('subreddit').agg({
            'score': ['mean', 'max'],
            'num_comments': ['mean', 'max'],
            'upvote_ratio': 'mean'
        }).round(2)
        
        # Create engagement visualization
        fig = px.scatter(df, 
                        x='score', 
                        y='num_comments',
                        color='subreddit',
                        size='upvote_ratio',
                        hover_data=['title'],
                        title='Post Engagement Analysis')
        
        fig.write_html(f'{self.output_dir}/plots/engagement_analysis.html')
        
        return engagement
    
    def analyze_content(self, df):
        """Analyze post content patterns"""
        # Word count distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='subreddit', y='word_count', data=df)
        plt.title('Word Count Distribution by Subreddit')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/word_count_dist.png')
        plt.close()
        
        # Generate word clouds for each subreddit
        for subreddit in df['subreddit'].unique():
            text = ' '.join(df[df['subreddit'] == subreddit]['cleaned_text'])
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white').generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - r/{subreddit}')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/wordcloud_{subreddit}.png')
            plt.close()
        
        return {
            'avg_word_count': df.groupby('subreddit')['word_count'].mean().round(2),
            'max_word_count': df.groupby('subreddit')['word_count'].max()
        }
    
    def generate_summary_report(self, df, sentiment_analysis, engagement_analysis, content_analysis):
        """Generate a comprehensive summary report"""
        report = {
            'data_overview': {
                'total_posts': len(df),
                'date_range': f"{df['created_utc'].min()} to {df['created_utc'].max()}",
                'num_subreddits': df['subreddit'].nunique()
            },
            'sentiment_insights': {
                'overall_sentiment': df['sentiment'].mean(),
                'most_positive_subreddit': sentiment_analysis['sentiment_by_subreddit']['mean'].idxmax(),
                'most_negative_subreddit': sentiment_analysis['sentiment_by_subreddit']['mean'].idxmin()
            },
            'engagement_insights': {
                'avg_score': df['score'].mean(),
                'avg_comments': df['num_comments'].mean(),
                'most_engaging_subreddit': engagement_analysis['score']['mean'].idxmax()
            },
            'content_insights': {
                'avg_word_count': df['word_count'].mean(),
                'most_verbose_subreddit': content_analysis['avg_word_count'].idxmax()
            }
        }
        
        return report

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RedditAnalyzer()
    
    # Load data
    print("Loading cleaned data...")
    df = analyzer.load_cleaned_data()
    
    # Perform analyses
    print("\nPerforming sentiment analysis...")
    sentiment_results = analyzer.analyze_sentiment_trends(df)
    
    print("Analyzing engagement patterns...")
    engagement_results = analyzer.analyze_engagement(df)
    
    print("Analyzing content patterns...")
    content_results = analyzer.analyze_content(df)
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = analyzer.generate_summary_report(
        df, sentiment_results, engagement_results, content_results
    )
    
    # Print key insights
    print("\nKey Insights:")
    print("-" * 50)
    for category, insights in report.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for key, value in insights.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")