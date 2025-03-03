import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RedditDataCleaner:
    def __init__(self):
        # Create processed data directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
    def load_latest_data(self):
        """Load the most recent data file from raw directory"""
        raw_files = os.listdir('data/raw')
        latest_file = sorted(raw_files)[-1]
        return pd.read_csv(f'data/raw/{latest_file}')
    
    def clean_text(self, text):
        """Clean text data"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            # Remove special characters and extra whitespace
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        return ''
    
    def add_text_features(self, df):
        """Add features based on text analysis"""
        # Combine title and text for analysis
        df['full_text'] = df['title'] + ' ' + df['text'].fillna('')
        
        # Clean text
        df['cleaned_text'] = df['full_text'].apply(self.clean_text)
        
        # Add text length features
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Add sentiment analysis
        df['sentiment'] = df['cleaned_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        
        return df
    
    def clean_data(self, df):
        """Main cleaning function"""
        # Convert created_utc to datetime if it's not already
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        
        # Add time-based features
        df['hour'] = df['created_utc'].dt.hour
        df['day_of_week'] = df['created_utc'].dt.day_name()
        
        # Add text features
        df = self.add_text_features(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'text'])
        
        # Remove posts with no content
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def generate_initial_insights(self, df):
        """Generate initial insights and plots"""
        # Create plots directory if it doesn't exist
        os.makedirs('data/processed/plots', exist_ok=True)
        
        # 1. Posts per subreddit
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df['subreddit'].value_counts().index, 
                   y=df['subreddit'].value_counts().values)
        plt.title('Number of Posts per Subreddit')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/plots/posts_per_subreddit.png')
        plt.close()
        
        # 2. Sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='sentiment', bins=50)
        plt.title('Sentiment Distribution')
        plt.tight_layout()
        plt.savefig('data/processed/plots/sentiment_distribution.png')
        plt.close()
        
        # 3. Posts by day of week
        plt.figure(figsize=(10, 6))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        sns.countplot(data=df, x='day_of_week', order=day_order)
        plt.title('Posts by Day of Week')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/plots/posts_by_day.png')
        plt.close()
        
        # Generate summary statistics
        summary = {
            'total_posts': len(df),
            'avg_sentiment': df['sentiment'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'most_active_subreddit': df['subreddit'].mode()[0],
            'most_active_day': df['day_of_week'].mode()[0]
        }
        
        return summary

if __name__ == "__main__":
    # Initialize cleaner
    cleaner = RedditDataCleaner()
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = cleaner.load_latest_data()
    cleaned_df = cleaner.clean_data(df)
    
    # Generate insights
    print("\nGenerating insights...")
    insights = cleaner.generate_initial_insights(cleaned_df)
    
    # Save cleaned data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cleaned_df.to_csv(f'data/processed/cleaned_data_{timestamp}.csv', index=False)
    
    # Print summary
    print("\nData Cleaning Summary:")
    print("-" * 50)
    print(f"Original number of posts: {len(df)}")
    print(f"Cleaned number of posts: {len(cleaned_df)}")
    print("\nInitial Insights:")
    for key, value in insights.items():
        print(f"{key}: {value}")