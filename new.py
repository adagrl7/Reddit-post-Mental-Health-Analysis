# Mental Health Discourse Analysis with VADER Sentiment Analysis
# Updated sentiment analysis implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import Counter
import re
from scipy.stats import pearsonr

# For text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# For VADER sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# For topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Download necessary NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Create output directory for results
if not os.path.exists('results'):
    os.makedirs('results')

# Load your collected data
print("Loading data...")
file_path = os.path.join('data', 'raw', 'reddit_posts_20250217_022648.csv')
all_posts_df = pd.read_csv(file_path)

# Define subreddit column
subreddit_col = 'subreddit' if 'subreddit' in all_posts_df.columns else 'source'

# Basic preprocessing
print("Preprocessing text data...")

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters, links, etc.
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stops = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stops]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back to string
        return ' '.join(tokens)
    else:
        return ''

# Apply preprocessing to the text column
text_column = 'selftext' if 'selftext' in all_posts_df.columns else 'body' if 'body' in all_posts_df.columns else 'text'
all_posts_df['processed_text'] = all_posts_df[text_column].apply(preprocess_text)
all_posts_df['original_text'] = all_posts_df[text_column]

# Handle datetime conversion with flexible format detection
print("Converting timestamps...")
date_column = 'created_utc' if 'created_utc' in all_posts_df.columns else 'created'
try:
    # First try converting as unix timestamp
    all_posts_df['post_date'] = pd.to_datetime(all_posts_df[date_column], unit='s', errors='coerce')
    
    # If we got mostly NaT values, the format is likely already datetime strings
    if all_posts_df['post_date'].isna().mean() > 0.5:  # If more than 50% are NaT
        all_posts_df['post_date'] = pd.to_datetime(all_posts_df[date_column], errors='coerce')
        
    # Create month-year field for temporal analysis
    all_posts_df['month_year'] = all_posts_df['post_date'].dt.strftime('%Y-%m')
    
except Exception as e:
    print(f"Warning: Error converting timestamps: {e}")
    print("Creating artificial timestamps for demonstration...")
    # Create artificial timestamps for demonstration
    start_date = pd.Timestamp('2024-10-01')
    all_posts_df['post_date'] = [start_date + pd.Timedelta(days=i % 180) for i in range(len(all_posts_df))]
    all_posts_df['month_year'] = all_posts_df['post_date'].dt.strftime('%Y-%m')

# Implement VADER sentiment analysis
print("Implementing VADER sentiment analysis...")
sid = SentimentIntensityAnalyzer()

# Function to categorize sentiment based on compound score
def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply VADER to get sentiment scores and categorize
all_posts_df['vader_scores'] = all_posts_df['original_text'].apply(
    lambda text: sid.polarity_scores(text) if isinstance(text, str) else {'compound': 0}
)

# Extract compound score and categorize
all_posts_df['sentiment_score'] = all_posts_df['vader_scores'].apply(lambda x: x['compound'])
all_posts_df['predicted_sentiment'] = all_posts_df['sentiment_score'].apply(categorize_sentiment)

# Calculate sentiment distribution
sentiment_counts = all_posts_df['predicted_sentiment'].value_counts()
print("VADER Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count} posts ({count/len(all_posts_df)*100:.1f}%)")

# Save sentiment-classified dataset
all_posts_df.to_csv('results/vader_classified_mental_health_posts.csv', index=False)
print("Saved VADER classified posts to results/vader_classified_mental_health_posts.csv")

# Mental health-specific sentiment adjustments
print("Applying mental health context adjustments...")

# Define mental health terms with sentiment modifiers
mental_health_terms = {
    'positive_resilience': [
        'coping', 'recovery', 'healing', 'therapy', 'progress', 'improvement',
        'self-care', 'support', 'mindfulness', 'resilience', 'strength'
    ],
    'negative_distress': [
        'suicidal', 'suicide', 'self-harm', 'cutting', 'hopeless', 'worthless',
        'empty', 'numb', 'panic', 'terrified', 'breakdown', 'crisis'
    ]
}
def adjust_mental_health_sentiment(row):
    text = row['original_text'].lower() if isinstance(row['original_text'], str) else ""
    score = row['sentiment_score']
    
    # Check for positive resilience terms
    resilience_count = sum(1 for term in mental_health_terms['positive_resilience'] if term in text)
    
    # Check for negative distress terms
    distress_count = sum(1 for term in mental_health_terms['negative_distress'] if term in text)
    
    # Adjust score based on mental health context
    adjustment = (resilience_count * 0.05) - (distress_count * 0.1)
    adjusted_score = score + adjustment
    
    # Cap the scores to remain in [-1, 1]
    adjusted_score = max(min(adjusted_score, 1.0), -1.0)
    
    return adjusted_score

# Apply mental health adjustments
all_posts_df['adjusted_sentiment_score'] = all_posts_df.apply(adjust_mental_health_sentiment, axis=1)
all_posts_df['adjusted_sentiment'] = all_posts_df['adjusted_sentiment_score'].apply(categorize_sentiment)

# Compare original VADER vs adjusted sentiment
comparison = pd.crosstab(
    all_posts_df['predicted_sentiment'], 
    all_posts_df['adjusted_sentiment'],
    rownames=['Original VADER'], 
    colnames=['Mental Health Adjusted']
)

print("\nSentiment Classification Comparison (Original vs Adjusted):")
print(comparison)
print("\nPercentage of posts with changed sentiment classification:", 
      (all_posts_df['predicted_sentiment'] != all_posts_df['adjusted_sentiment']).mean() * 100, "%")

# Save the comparison to file
comparison.to_csv('results/sentiment_adjustment_comparison.csv')

# Use the adjusted sentiment for further analysis
all_posts_df['final_sentiment'] = all_posts_df['adjusted_sentiment']

# Analysis 1: Sentiment distribution by subreddit
print("Analyzing sentiment distribution by subreddit...")
subreddit_col = 'subreddit' if 'subreddit' in all_posts_df.columns else 'source'
sentiment_by_subreddit = all_posts_df.groupby([subreddit_col, 'final_sentiment']).size().unstack(fill_value=0)
sentiment_pct = sentiment_by_subreddit.div(sentiment_by_subreddit.sum(axis=1), axis=0) * 100

# Visualize sentiment distribution
plt.figure(figsize=(12, 8))
sentiment_pct.plot(kind='bar', stacked=True, colormap='RdYlGn')
plt.title('Sentiment Distribution by Mental Health Subreddit (VADER)')
plt.xlabel('Subreddit')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('results/vader_sentiment_by_subreddit.png')
plt.close()
print("Saved sentiment distribution visualization to results/vader_sentiment_by_subreddit.png")

# Analysis 2: Temporal analysis of sentiment changes
print("Analyzing sentiment changes over time...")
# Group by month and sentiment for each subreddit
temporal_sentiment = all_posts_df.groupby(['month_year', subreddit_col, 'final_sentiment']).size().unstack(fill_value=0)

# Plot time series for each subreddit
plt.figure(figsize=(15, 10))
for subreddit in all_posts_df[subreddit_col].unique():
    try:
        subset = temporal_sentiment.xs(subreddit, level=subreddit_col)
        subset.plot(marker='o', linestyle='-', ax=plt.gca(), label=subreddit)
    except KeyError:
        print(f"No data available for subreddit: {subreddit}")
        continue

plt.title('Sentiment Trends in Mental Health Subreddits Over Time (VADER)')
plt.xlabel('Month')
plt.ylabel('Number of Posts')
plt.grid(True, alpha=0.3)
plt.legend(title='Subreddit and Sentiment')
plt.tight_layout()
plt.savefig('results/vader_sentiment_trends_over_time.png')
plt.close()
print("Saved temporal sentiment trends to results/vader_sentiment_trends_over_time.png")

# Analysis 3: Topic modeling by sentiment category
print("Performing topic modeling by sentiment category...")

def get_top_words(topic_model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(topic_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topics

# Setup for results
num_topics = 5  # Adjust based on your needs
sentiments = all_posts_df['final_sentiment'].unique()
results = {}

# Open file for writing results
with open('results/vader_topic_modeling_results.txt', 'w') as f:
    for sentiment in sentiments:
        f.write(f"\n\nTop topics for {sentiment} posts:\n")
        
        # Filter posts by sentiment
        sentiment_posts = all_posts_df[all_posts_df['final_sentiment'] == sentiment]
        
        if len(sentiment_posts) < num_topics:  # Skip if too few posts
            f.write(f"Not enough posts with {sentiment} sentiment for topic modeling.\n")
            continue
        
        # Create vectorizer for topic modeling
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, 
                                        stop_words='english', 
                                        max_features=1000)
        
        # Create document-term matrix
        tfidf = tfidf_vectorizer.fit_transform(sentiment_posts['processed_text'])
        
        # Apply NMF for topic modeling
        try:
            nmf = NMF(n_components=num_topics, random_state=42, max_iter=1000)
            nmf.fit(tfidf)
            
            # Get feature names
            tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get top words for each topic
            topics = get_top_words(nmf, tfidf_feature_names)
            
            # Store results
            results[sentiment] = {
                'topics': topics,
                'model': nmf,
                'vectorizer': tfidf_vectorizer
            }
            
            # Write topics to file
            for idx, topic in enumerate(topics):
                f.write(f"Topic #{idx + 1}: {', '.join(topic)}\n")
        
        except Exception as e:
            f.write(f"Error in topic modeling for {sentiment} posts: {str(e)}\n")

print("Saved topic modeling results to results/vader_topic_modeling_results.txt")

# Analysis 4: User engagement by sentiment
print("Analyzing user engagement by sentiment...")

# Possible engagement metrics
engagement_metrics = []
if 'score' in all_posts_df.columns:
    engagement_metrics.append('score')
if 'ups' in all_posts_df.columns:
    engagement_metrics.append('ups')
if 'num_comments' in all_posts_df.columns:
    engagement_metrics.append('num_comments')
if 'comments' in all_posts_df.columns:
    engagement_metrics.append('comments')

if engagement_metrics:
    # Average engagement metrics by sentiment
    engagement_by_sentiment = all_posts_df.groupby('final_sentiment')[engagement_metrics].mean()
    
    # Visualize engagement metrics
    plt.figure(figsize=(10, 6))
    engagement_by_sentiment.plot(kind='bar')
    plt.title('Average User Engagement by Sentiment (VADER)')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Count')
    plt.legend(title='Engagement Metric')
    plt.tight_layout()
    plt.savefig('results/vader_engagement_by_sentiment.png')
    plt.close()
    
    # Save engagement metrics to CSV
    engagement_by_sentiment.to_csv('results/vader_engagement_by_sentiment.csv')
    print("Saved user engagement analysis to results/vader_engagement_by_sentiment.png and CSV")
else:
    print("User engagement metrics not available in the dataset.")

# Analysis 5: Sentiment intensity analysis
print("Analyzing sentiment intensity distribution...")

# Create bins for sentiment intensity
all_posts_df['sentiment_intensity'] = all_posts_df['adjusted_sentiment_score'].abs()
all_posts_df['intensity_level'] = pd.cut(
    all_posts_df['sentiment_intensity'], 
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
)

# Plot sentiment intensity distribution
plt.figure(figsize=(12, 8))
sns.countplot(data=all_posts_df, x='intensity_level', hue='final_sentiment', palette='RdYlGn')
plt.title('Sentiment Intensity Distribution in Mental Health Subreddits')
plt.xlabel('Sentiment Intensity')
plt.ylabel('Number of Posts')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('results/vader_sentiment_intensity_distribution.png')
plt.close()
print("Saved sentiment intensity distribution to results/vader_sentiment_intensity_distribution.png")


# Add this after the mental health terms dictionary
print("Implementing mental health-specific analysis...")

# 1. Enhanced Mental Health Lexicon Matching
# --------------------------------------------
mental_health_categories = {
    'anxiety': [
        'anxious', 'worry', 'panic', 'nervous', 'dread', 'fear', 'scared', 
        'terrified', 'restless', 'tense', 'uneasy', 'jittery', 'overthinking',
        'racing thoughts', 'heart racing', 'trouble breathing', 'chest pain'
    ],
    'depression': [
        'depressed', 'sad', 'hopeless', 'worthless', 'empty', 'numb', 'tired',
        'exhausted', 'fatigue', 'insomnia', 'sleeping too much', 'no appetite',
        'overeating', 'guilt', 'shame', 'isolation', 'lonely', 'no energy'
    ],
    'trauma': [
        'trauma', 'ptsd', 'flashback', 'nightmare', 'triggered', 'abuse',
        'assault', 'victim', 'hypervigilant', 'startle', 'dissociate', 'numb'
    ],
    'cognitive_distortions': [
        'always', 'never', 'everyone', 'nobody', 'catastrophe', 'disaster',
        'terrible', 'horrible', 'can\'t stand', 'unbearable', 'worst', 'should',
        'must', 'have to', 'failure', 'stupid', 'worthless', 'useless'
    ],
    'recovery_indicators': [
        'therapy', 'counseling', 'medication', 'psychiatrist', 'psychologist',
        'coping', 'self-care', 'progress', 'improving', 'better', 'hope',
        'grateful', 'mindfulness', 'meditation', 'exercise', 'routine'
    ],
    'support_seeking': [
        'help', 'advice', 'need', 'please', 'anyone', 'suggestion', 'how do I',
        'what should', 'struggling with', 'can\'t cope', 'desperate'
    ],
    'support_giving': [
        'hope this helps', 'try this', 'worked for me', 'suggestion', 'recommend',
        'you could', 'have you considered', 'in my experience', 'you\'re not alone'
    ]
}

# Function to detect mental health indicators in text
def detect_mental_health_indicators(text):
    if not isinstance(text, str):
        return {}
    
    text = text.lower()
    results = {}
    
    # Count occurrences of terms in each category
    for category, terms in mental_health_categories.items():
        count = sum(1 for term in terms if re.search(r'\b' + re.escape(term) + r'\b', text))
        results[category] = count
    
    # Calculate percentage of total text length that contains mental health terms
    total_terms = sum(results.values())
    total_words = len(text.split())
    if total_words > 0:
        results['mh_term_density'] = total_terms / total_words
    else:
        results['mh_term_density'] = 0
        
    return results

# Apply the detection function
all_posts_df['mh_indicators'] = all_posts_df['original_text'].apply(detect_mental_health_indicators)

# Extract individual category counts as separate columns
for category in mental_health_categories.keys():
    all_posts_df[f'{category}_count'] = all_posts_df['mh_indicators'].apply(lambda x: x.get(category, 0))

all_posts_df['mh_term_density'] = all_posts_df['mh_indicators'].apply(lambda x: x.get('mh_term_density', 0))

# 2. Support-Seeking vs Support-Giving Classification
# --------------------------------------------------
def classify_post_intention(row):
    if not isinstance(row['original_text'], str):
        return 'unknown'
    
    text = row['original_text'].lower()
    seeking_count = row['support_seeking_count']
    giving_count = row['support_giving_count']
    
    # Check for question marks as an indicator of seeking
    question_count = text.count('?')
    
    # Additional patterns for support seeking
    seeking_patterns = [
        r'\bhelp me\b', r'\bi need\b', r'\badvice\b', r'\banyone else\b',
        r'\bwhat (?:do|should|can|would)\b'
    ]
    seeking_pattern_count = sum(1 for pattern in seeking_patterns if re.search(pattern, text))
    
    # Additional patterns for support giving
    giving_patterns = [
        r'\bthis helped me\b', r'\bi found\b', r'\btry\b', r'\bmight help\b',
        r'\bin my experience\b', r'\bi recommend\b'
    ]
    giving_pattern_count = sum(1 for pattern in giving_patterns if re.search(pattern, text))
    
    # Combined scores
    seeking_score = seeking_count + question_count + seeking_pattern_count
    giving_score = giving_count + giving_pattern_count
    
    # Post length adjustment - longer posts are more likely to be support-giving
    word_count = len(text.split())
    if word_count > 200:
        giving_score += 1
    
    # Classification logic
    if seeking_score > giving_score:
        return 'support_seeking'
    elif giving_score > seeking_score:
        return 'support_giving'
    else:
        # If tied, check if post starts with a question or seeking pattern
        first_50_words = ' '.join(text.split()[:50])
        if '?' in first_50_words or any(re.search(pattern, first_50_words) for pattern in seeking_patterns):
            return 'support_seeking'
        else:
            return 'other'

# Apply the classification
all_posts_df['post_intention'] = all_posts_df.apply(classify_post_intention, axis=1)

# 3. Cognitive Distortion Analysis
# --------------------------------
cognitive_distortion_patterns = {
    'all_or_nothing': [
        r'\balways\b', r'\bnever\b', r'\beveryone\b', r'\bnothing\b', 
        r'\bcomplete\w*\b', r'\btotal\w*\b', r'\bperfect\w*\b'
    ],
    'overgeneralization': [
        r'\beverything\b', r'\bnothing\b', r'\beverybody\b', r'\bnobody\b',
        r'\beverywhere\b', r'\bnowhere\b'
    ],
    'catastrophizing': [
        r'\bterrible\b', r'\bhorrible\b', r'\bcatastroph\w*\b', r'\bdisaster\b',
        r'\bawful\b', r'\bembarrass\w*\b', r'\bhumiliat\w*\b', r'\brunied\b'
    ],
    'emotional_reasoning': [
        r'\bfeel (?:like|that) [\w\s]+ (?:am|is|are|must|will|should)\b',
        r'\bknow (?:that|it\'s) [\w\s]+ (?:am|is|are|must|will|should)\b'
    ],
    'should_statements': [
        r'\bshould\b', r'\bmust\b', r'\bhave to\b', r'\bsupposed to\b',
        r'\bneed to\b', r'\bought to\b'
    ],
    'personalization': [
        r'\b(?:my|all my) fault\b', r'\bblame (?:me|myself)\b', 
        r'\bresponsible for\b', r'\bcaused by me\b'
    ],
    'mental_filtering': [
        r'\bno good\b', r'\bnever good enough\b', r'\balways bad\b', 
        r'\bnothing good\b', r'\bfail\w* at everything\b'
    ]
}

def identify_cognitive_distortions(text):
    if not isinstance(text, str):
        return {}
    
    text = text.lower()
    distortions = {}
    
    for distortion_type, patterns in cognitive_distortion_patterns.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            if found:
                matches.extend(found)
        
        distortions[distortion_type] = len(matches)
        
    # Calculate total distortions
    distortions['total_distortions'] = sum(distortions.values())
    
    return distortions

# Apply the distortion detection
all_posts_df['cognitive_distortions'] = all_posts_df['original_text'].apply(identify_cognitive_distortions)

# Extract individual distortion counts
for distortion_type in cognitive_distortion_patterns.keys():
    all_posts_df[f'{distortion_type}_count'] = all_posts_df['cognitive_distortions'].apply(
        lambda x: x.get(distortion_type, 0)
    )

all_posts_df['total_distortions'] = all_posts_df['cognitive_distortions'].apply(
    lambda x: x.get('total_distortions', 0)
)

# 4. Mental Health Severity Scoring
# --------------------------------
def calculate_mh_severity_score(row):
    # Base components for severity score
    depression_component = row['depression_count'] * 0.8
    anxiety_component = row['anxiety_count'] * 0.7
    trauma_component = row['trauma_count'] * 1.0
    distortion_component = row['total_distortions'] * 0.5
    # Check if adjusted sentiment score was properly created
    if 'adjusted_sentiment_score' not in all_posts_df.columns:
        print("WARNING: 'adjusted_sentiment_score' column was not created properly.")
        print("Available columns:", all_posts_df.columns.tolist())
        
        # Create the column explicitly if missing
        print("Creating 'adjusted_sentiment_score' as a copy of 'sentiment_score'...")
        if 'sentiment_score' in all_posts_df.columns:
            all_posts_df['adjusted_sentiment_score'] = all_posts_df['sentiment_score']
        else:
            print("ERROR: 'sentiment_score' also not found. Using zeros for sentiment scores.")
            all_posts_df['adjusted_sentiment_score'] = 0.0
            
        # Check if 'adjusted_sentiment' column exists, and create it if missing
        if 'adjusted_sentiment' not in all_posts_df.columns:
            all_posts_df['adjusted_sentiment'] = all_posts_df['adjusted_sentiment_score'].apply(categorize_sentiment)
            
        print("Created missing sentiment columns.")
    # Adjust based on sentiment score (more negative = more severe)
        # Check which sentiment score column exists and use it
    if 'adjusted_sentiment_score' in row:
        sentiment_score = row['adjusted_sentiment_score']
    elif 'sentiment_score' in row:
        sentiment_score = row['sentiment_score']
    else:
        # Default value if no sentiment score is available
        sentiment_score = 0
    
    # Adjust based on sentiment score (more negative = more severe)
    # Convert sentiment score from [-1,1] range to [0,2] range for severity calculation
    sentiment_component = (1 - sentiment_score) * 2
    
    # Calculate weighted score
    severity_score = (
        depression_component + 
        anxiety_component + 
        trauma_component + 
        distortion_component + 
        sentiment_component
    ) / 5
    
    # Normalize to 0-10 scale
    normalized_score = min(max(severity_score, 0), 10)
    
    # Categorize severity
    if normalized_score < 3:
        severity_category = 'mild'
    elif normalized_score < 6:
        severity_category = 'moderate'
    else:
        severity_category = 'severe'
        
    return {'score': normalized_score, 'category': severity_category}

# Apply severity scoring
all_posts_df['severity_analysis'] = all_posts_df.apply(calculate_mh_severity_score, axis=1)
all_posts_df['severity_score'] = all_posts_df['severity_analysis'].apply(lambda x: x['score'])
all_posts_df['severity_category'] = all_posts_df['severity_analysis'].apply(lambda x: x['category'])

# 5. Visualizations and Analysis for Mental Health Indicators
# ----------------------------------------------------------

# Analysis 1: Mental Health Indicator Distribution
print("Generating mental health indicator distribution analysis...")

# Calculate average category counts
mh_categories_avg = all_posts_df[[f'{cat}_count' for cat in mental_health_categories.keys()]].mean()

plt.figure(figsize=(12, 6))
mh_categories_avg.plot(kind='bar', color='purple')
plt.title('Average Mental Health Indicator Frequency per Post')
plt.xlabel('Mental Health Category')
plt.ylabel('Average Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/mental_health_indicator_distribution.png')
plt.close()

# Analysis 2: Correlation between severity and engagement
print("Analyzing correlation between mental health severity and engagement...")
# Add this code right before the "Analyzing correlation between mental health severity and engagement..." section

# Re-define engagement metrics if they're not already defined
# This ensures the variable exists for the upcoming analysis
print("Checking for engagement metrics...")
engagement_metrics = []
if 'score' in all_posts_df.columns:
    engagement_metrics.append('score')
if 'ups' in all_posts_df.columns:
    engagement_metrics.append('ups')
if 'num_comments' in all_posts_df.columns:
    engagement_metrics.append('num_comments')
if 'comments' in all_posts_df.columns:
    engagement_metrics.append('comments')

print(f"Found {len(engagement_metrics)} engagement metrics: {engagement_metrics}")

# Continue with the correlation analysis
print("Analyzing correlation between mental health severity and engagement...")

if engagement_metrics:
    # Calculate correlations
    correlation_results = {}
    for metric in engagement_metrics:
        try:
            corr, p_value = pearsonr(all_posts_df['severity_score'], all_posts_df[metric])
            correlation_results[metric] = {'correlation': corr, 'p_value': p_value}
        except Exception as e:
            print(f"Error calculating correlation for {metric}: {e}")
            correlation_results[metric] = {'correlation': None, 'p_value': None}
    
    # Create correlation dataframe
    corr_df = pd.DataFrame(correlation_results).T
    
    # Save correlation results
    corr_df.to_csv('results/severity_engagement_correlation.csv')
    
    # Visualize relationship
    try:
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(engagement_metrics, 1):
            if len(engagement_metrics) > 1:
                plt.subplot(len(engagement_metrics), 1, i)
            sns.scatterplot(x='severity_score', y=metric, hue='severity_category', data=all_posts_df)
            plt.title(f'Relationship Between Mental Health Severity and {metric}')
            plt.xlabel('Mental Health Severity Score (0-10)')
            plt.ylabel(metric.capitalize())
        
        plt.tight_layout()
        plt.savefig('results/severity_engagement_relationship.png')
        plt.close()
        print("Generated severity vs engagement visualization")
    except Exception as e:
        print(f"Error generating engagement visualization: {e}")
else:
    print("No engagement metrics available in the dataset. Skipping correlation analysis.")
if engagement_metrics:
    # Calculate correlations
    correlation_results = {}
    for metric in engagement_metrics:
        corr, p_value = pearsonr(all_posts_df['severity_score'], all_posts_df[metric])
        correlation_results[metric] = {'correlation': corr, 'p_value': p_value}
    
    # Create correlation dataframe
    corr_df = pd.DataFrame(correlation_results).T
    
    # Save correlation results
    corr_df.to_csv('results/severity_engagement_correlation.csv')
    
    # Visualize relationship
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(engagement_metrics, 1):
        plt.subplot(len(engagement_metrics), 1, i)
        sns.scatterplot(x='severity_score', y=metric, hue='severity_category', data=all_posts_df)
        plt.title(f'Relationship Between Mental Health Severity and {metric}')
        plt.xlabel('Mental Health Severity Score (0-10)')
        plt.ylabel(metric.capitalize())
    
    plt.tight_layout()
    plt.savefig('results/severity_engagement_relationship.png')
    plt.close()

# Analysis 3: Cognitive distortions by sentiment category
print("Analyzing cognitive distortions by sentiment category...")
if 'final_sentiment' in all_posts_df.columns:
    distortion_by_sentiment = all_posts_df.groupby('final_sentiment')[
        [f'{d}_count' for d in cognitive_distortion_patterns.keys() if f'{d}_count' in all_posts_df.columns]
    ].mean()
else:
    print("Error: 'final_sentiment' column is missing from DataFrame")

# Average distortions by sentiment
distortion_by_sentiment = all_posts_df.groupby('final_sentiment')[[f'{d}_count' for d in cognitive_distortion_patterns.keys()]].mean()

# Visualize distortions by sentiment
plt.figure(figsize=(14, 8))
distortion_by_sentiment.plot(kind='bar')
plt.title('Average Cognitive Distortion Frequency by Sentiment Category')
plt.xlabel('Sentiment')
plt.ylabel('Average Frequency')
plt.legend(title='Distortion Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/cognitive_distortions_by_sentiment.png')
plt.close()

# Analysis 4: Support-seeking vs support-giving sentiment comparison
print("Comparing sentiment between support-seeking and support-giving posts...")

# Calculate sentiment distribution by post intention
intention_sentiment = pd.crosstab(
    all_posts_df['post_intention'], 
    all_posts_df['final_sentiment'],
    normalize='index'
) * 100

# Visualize intention vs sentiment
plt.figure(figsize=(10, 6))
intention_sentiment.plot(kind='bar', stacked=True, colormap='RdYlGn')
plt.title('Sentiment Distribution by Post Intention')
plt.xlabel('Post Intention')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('results/post_intention_sentiment_distribution.png')
plt.close()

# Analysis 5: Mental health severity by subreddit
print("Analyzing mental health severity patterns by subreddit...")

# Add this code right before the "Analyzing mental health severity patterns by subreddit..." section

# Re-define subreddit_col if it's not already defined
print("Checking for subreddit column...")
if 'subreddit_col' not in locals() or not subreddit_col:
    # Choose the appropriate column name for subreddit
    if 'subreddit' in all_posts_df.columns:
        subreddit_col = 'subreddit'
    elif 'source' in all_posts_df.columns:
        subreddit_col = 'source'
    else:
        # Look for any column that might contain subreddit information
        potential_cols = [col for col in all_posts_df.columns if any(s in col.lower() for s in ['sub', 'source', 'community', 'forum'])]
        if potential_cols:
            subreddit_col = potential_cols[0]
            print(f"Using '{subreddit_col}' as the subreddit column")
        else:
            # Create a default if no suitable column is found
            print("No suitable subreddit column found. Creating a placeholder.")
            all_posts_df['subreddit_placeholder'] = 'unknown'
            subreddit_col = 'subreddit_placeholder'

print(f"Using '{subreddit_col}' for subreddit analysis")

# Check if there are multiple subreddits to analyze
subreddit_count = all_posts_df[subreddit_col].nunique()
print(f"Found {subreddit_count} unique subreddit values")

# Only proceed with visualization if there are multiple subreddits
print("Analyzing mental health severity patterns by subreddit...")
print("Checking for subreddit column...")
if subreddit_count > 1:
    # Calculate average severity by subreddit
    severity_by_subreddit = all_posts_df.groupby(subreddit_col)[['severity_score']].mean()
    severity_by_subreddit['severity_std'] = all_posts_df.groupby(subreddit_col)[['severity_score']].std()

    # Plot severity by subreddit
    plt.figure(figsize=(12, 6))
    ax = severity_by_subreddit['severity_score'].plot(kind='bar', yerr=severity_by_subreddit['severity_std'], 
                                                    capsize=4, color='darkred', alpha=0.7)
    plt.title('Mental Health Severity Score by Subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('Average Severity Score (0-10)')
    plt.xticks(rotation=45, ha='right')

    # Add category thresholds
    plt.axhline(y=3, linestyle='--', color='green', alpha=0.7, label='Mild/Moderate Threshold')
    plt.axhline(y=6, linestyle='--', color='red', alpha=0.7, label='Moderate/Severe Threshold')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/mental_health_severity_by_subreddit.png')
    plt.close()
    print("Generated severity by subreddit visualization")
else:
    print(f"Only found {subreddit_count} subreddit. Skipping subreddit comparison visualization.")
    # Create a simplified analysis instead
    avg_severity = all_posts_df['severity_score'].mean()
    std_severity = all_posts_df['severity_score'].std()
    print(f"Overall severity score: {avg_severity:.2f} ± {std_severity:.2f}")
    
    # Save a simple summary instead
    with open('results/severity_summary.txt', 'w') as f:
        f.write(f"Overall Mental Health Severity Analysis\n")
        f.write(f"=====================================\n")
        f.write(f"Average severity score: {avg_severity:.2f}\n")
        f.write(f"Standard deviation: {std_severity:.2f}\n")
        f.write(f"Severity distribution:\n")
        
        # Calculate distribution of severity categories
        severity_dist = all_posts_df['severity_category'].value_counts(normalize=True) * 100
        for category, percentage in severity_dist.items():
            f.write(f"  - {category}: {percentage:.1f}%\n")

# Save enhanced analysis results
print("Saving enhanced mental health analysis results...")
all_posts_df.to_csv('results/enhanced_mental_health_analysis.csv', index=False)


# Final summary
print("\nVADER Analysis complete! Results saved to the 'results' directory.")
print("Files generated:")
print("- vader_classified_mental_health_posts.csv: All posts with VADER sentiment classifications")
print("- sentiment_adjustment_comparison.csv: Comparison of original vs. mental health adjusted sentiment")
print("- vader_sentiment_by_subreddit.png: Visualization of sentiment distribution by subreddit")
print("- vader_sentiment_trends_over_time.png: Visualization of sentiment changes over time")
print("- vader_topic_modeling_results.txt: Top topics identified for each sentiment category")
print("- vader_engagement_by_sentiment.png & .csv: User engagement analysis by sentiment")
print("- vader_sentiment_intensity_distribution.png: Analysis of sentiment intensity levels")

print("\nNext steps could include:")
print("1. Analyzing specific mental health indicators in text")
print("2. Comparing sentiment across different time periods or events")
print("3. Building a classifier to identify support-seeking vs. support-giving posts")
print("4. Creating a dashboard for visualizing mental health discourse patterns")