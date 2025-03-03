import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import re
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Mental Health Discourse Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A;
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #3B82F6;
    }
    .caption {
        font-size: 0.9rem;
        font-style: italic;
        color: #64748B;
    }
    .highlight {
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions

@st.cache_data
def load_data():
    """Load the analyzed data"""
    try:
        # Try to load the enhanced analysis results first
        df = pd.read_csv('results/enhanced_mental_health_analysis.csv')
        st.sidebar.success("Successfully loaded enhanced analysis data")
    except FileNotFoundError:
        try:
            # Fall back to the VADER classified posts
            df = pd.read_csv('results/vader_classified_mental_health_posts.csv')
            st.sidebar.success("Successfully loaded VADER classified data")
        except FileNotFoundError:
            # Use some sample data if no files are found
            st.sidebar.warning("No analysis data found. Using sample data for demonstration.")
            df = create_sample_data()
    
    return df

def create_sample_data(n=100):
    """Create sample data for demonstration when actual data isn't available"""
    subreddits = ['depression', 'anxiety', 'mentalhealth', 'bipolar']
    
    df = pd.DataFrame({
        'subreddit': np.random.choice(subreddits, n),
        'created_utc': np.random.randint(1640995200, 1709251200, n),  # 2022-2023 timestamps
        'original_text': ['Sample post text ' + str(i) for i in range(n)],
        'processed_text': ['sample post text ' + str(i) for i in range(n)],
        'sentiment_score': np.random.uniform(-1, 1, n),
        'predicted_sentiment': np.random.choice(['positive', 'negative', 'neutral'], n),
        'adjusted_sentiment_score': np.random.uniform(-1, 1, n),
        'adjusted_sentiment': np.random.choice(['positive', 'negative', 'neutral'], n),
        'final_sentiment': np.random.choice(['positive', 'negative', 'neutral'], n),
    })
    
    # Convert timestamps to datetime
    df['post_date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month_year'] = df['post_date'].dt.strftime('%Y-%m')
    
    # Add mental health specific columns if they don't exist
    mh_categories = ['anxiety', 'depression', 'trauma', 'recovery_indicators', 
                     'support_seeking', 'support_giving', 'cognitive_distortions']
    
    for category in mh_categories:
        df[f'{category}_count'] = np.random.randint(0, 5, n)
    
    df['severity_score'] = np.random.uniform(0, 10, n)
    df['severity_category'] = np.where(df['severity_score'] < 3, 'mild', 
                                      np.where(df['severity_score'] < 6, 'moderate', 'severe'))
    
    df['post_intention'] = np.random.choice(['support_seeking', 'support_giving', 'other'], n)
    
    # Add engagement metrics
    df['score'] = np.random.randint(1, 100, n)
    df['ups'] = df['score']
    df['num_comments'] = np.random.randint(0, 30, n)
    df['word_count'] = np.random.randint(50, 500, n)
    
    return df

def plot_sentiment_distribution(df):
    """Plot the sentiment distribution by subreddit"""
    subreddit_col = 'subreddit' if 'subreddit' in df.columns else 'source'
    
    # Group by subreddit and sentiment
    sentiment_by_subreddit = df.groupby([subreddit_col, 'final_sentiment']).size().unstack(fill_value=0)
    sentiment_pct = sentiment_by_subreddit.div(sentiment_by_subreddit.sum(axis=1), axis=0) * 100
    
    # Create a more interactive plot with plotly
    data = []
    categories = sentiment_pct.columns.tolist()
    
    for category in categories:
        data.append(go.Bar(
            name=category,
            x=sentiment_pct.index,
            y=sentiment_pct[category],
            marker_color='green' if category == 'positive' else 'red' if category == 'negative' else 'gray'
        ))
    
    fig = go.Figure(data=data)
    fig.update_layout(
        title='Sentiment Distribution by Mental Health Subreddit',
        xaxis_title='Subreddit',
        yaxis_title='Percentage',
        barmode='stack',
        height=500,
        legend_title='Sentiment'
    )
    
    return fig

def plot_sentiment_over_time(df):
    """Plot sentiment trends over time"""
    subreddit_col = 'subreddit' if 'subreddit' in df.columns else 'source'
    
    # Ensure month_year is properly formatted and sorted
    if 'month_year' not in df.columns:
        if 'post_date' in df.columns:
            df['month_year'] = pd.to_datetime(df['post_date']).dt.strftime('%Y-%m')
        else:
            return None  # No time data available
    
    # Group by month, subreddit, and sentiment
    sentiment_counts = df.groupby(['month_year', df['final_sentiment']]).size().unstack(fill_value=0)
    
    # Convert to percentage
    sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    
    # Create plot
    fig = go.Figure()
    
    if 'positive' in sentiment_pct.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_pct.index,
            y=sentiment_pct['positive'],
            mode='lines+markers',
            name='Positive',
            line=dict(color='green', width=2)
        ))
    
    if 'negative' in sentiment_pct.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_pct.index,
            y=sentiment_pct['negative'],
            mode='lines+markers',
            name='Negative',
            line=dict(color='red', width=2)
        ))
    
    if 'neutral' in sentiment_pct.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_pct.index,
            y=sentiment_pct['neutral'],
            mode='lines+markers',
            name='Neutral',
            line=dict(color='gray', width=2)
        ))
    
    fig.update_layout(
        title='Sentiment Trends Over Time',
        xaxis_title='Month',
        yaxis_title='Percentage',
        legend_title='Sentiment',
        height=500
    )
    
    return fig

def plot_mental_health_indicators(df):
    """Plot the distribution of mental health indicators"""
    mh_categories = ['anxiety', 'depression', 'trauma', 'cognitive_distortions', 
                    'recovery_indicators', 'support_seeking', 'support_giving']
    
    # Check which categories exist in the dataframe
    available_categories = [cat for cat in mh_categories if f'{cat}_count' in df.columns]
    
    if not available_categories:
        return None  # No mental health indicator data available
    
    # Calculate average occurrences
    avg_counts = [df[f'{cat}_count'].mean() for cat in available_categories]
    
    # Create the plot
    fig = go.Figure(data=[
        go.Bar(
            x=available_categories,
            y=avg_counts,
            marker_color='purple'
        )
    ])
    
    fig.update_layout(
        title='Average Mental Health Indicator Frequency per Post',
        xaxis_title='Mental Health Category',
        yaxis_title='Average Frequency',
        height=500
    )
    
    return fig

def plot_severity_by_subreddit(df):
    """Plot mental health severity scores by subreddit"""
    subreddit_col = 'subreddit' if 'subreddit' in df.columns else 'source'
    
    if 'severity_score' not in df.columns:
        return None  # No severity data available
    
    # Calculate average severity by subreddit
    severity_by_subreddit = df.groupby(subreddit_col)[['severity_score']].agg(['mean', 'std']).reset_index()
    severity_by_subreddit.columns = [subreddit_col, 'mean', 'std']
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=severity_by_subreddit[subreddit_col],
        y=severity_by_subreddit['mean'],
        error_y=dict(
            type='data',
            array=severity_by_subreddit['std'],
            visible=True
        ),
        marker_color='darkred'
    ))
    
    # Add threshold lines
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=3,
        x1=len(severity_by_subreddit) - 0.5,
        y1=3,
        line=dict(
            color="green",
            width=2,
            dash="dash",
        ),
        name="Mild/Moderate"
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=6,
        x1=len(severity_by_subreddit) - 0.5,
        y1=6,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        ),
        name="Moderate/Severe"
    )
    
    fig.update_layout(
        title='Mental Health Severity Score by Subreddit',
        xaxis_title='Subreddit',
        yaxis_title='Average Severity Score (0-10)',
        height=500,
        showlegend=False
    )
    
    return fig

def plot_sentiment_intention_relationship(df):
    """Plot the relationship between post intention and sentiment"""
    if 'post_intention' not in df.columns or 'final_sentiment' not in df.columns:
        return None  # Required data not available
    
    # Create crosstab
    intention_sentiment = pd.crosstab(
        df['post_intention'], 
        df['final_sentiment'],
        normalize='index'
    ) * 100
    
    # Create the plot
    fig = go.Figure()
    
    for sentiment in intention_sentiment.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=intention_sentiment.index,
            y=intention_sentiment[sentiment],
            marker_color='green' if sentiment == 'positive' else 'red' if sentiment == 'negative' else 'gray'
        ))
    
    fig.update_layout(
        title='Sentiment Distribution by Post Intention',
        xaxis_title='Post Intention',
        yaxis_title='Percentage',
        barmode='stack',
        height=500,
        legend_title='Sentiment'
    )
    
    return fig

def plot_intensity_distribution(df):
    """Plot the distribution of sentiment intensity"""
    if 'sentiment_intensity' not in df.columns and 'adjusted_sentiment_score' in df.columns:
        # Create sentiment intensity if it doesn't exist
        df['sentiment_intensity'] = df['adjusted_sentiment_score'].abs()
        df['intensity_level'] = pd.cut(
            df['sentiment_intensity'], 
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
    elif 'sentiment_intensity' not in df.columns and 'sentiment_score' in df.columns:
        # Alternative using regular sentiment score
        df['sentiment_intensity'] = df['sentiment_score'].abs()
        df['intensity_level'] = pd.cut(
            df['sentiment_intensity'], 
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
    elif 'intensity_level' not in df.columns:
        return None  # Required data not available
    
    # Count intensity levels by sentiment
    intensity_counts = pd.crosstab(df['intensity_level'], df['final_sentiment'])
    
    # Create the plot
    fig = go.Figure()
    
    for sentiment in intensity_counts.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=intensity_counts.index,
            y=intensity_counts[sentiment],
            marker_color='green' if sentiment == 'positive' else 'red' if sentiment == 'negative' else 'gray'
        ))
    
    fig.update_layout(
        title='Sentiment Intensity Distribution',
        xaxis_title='Sentiment Intensity',
        yaxis_title='Number of Posts',
        barmode='group',
        height=500,
        legend_title='Sentiment'
    )
    
    return fig

def plot_engagement_by_sentiment(df, metric):
    """Plot engagement metrics by sentiment"""
    if metric not in df.columns:
        return None  # Required metric not available
    
    # Calculate average metric by sentiment
    avg_by_sentiment = df.groupby('final_sentiment')[metric].mean().reset_index()
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=avg_by_sentiment['final_sentiment'],
        y=avg_by_sentiment[metric],
        marker_color=['green' if s == 'positive' else 'red' if s == 'negative' else 'gray' 
                     for s in avg_by_sentiment['final_sentiment']]
    ))
    
    fig.update_layout(
        title=f'Average {metric.replace("_", " ").title()} by Sentiment',
        xaxis_title='Sentiment',
        yaxis_title=f'Average {metric.replace("_", " ").title()}',
        height=500
    )
    
    return fig

def plot_engagement_by_severity(df, metric):
    """Plot engagement metrics by severity category"""
    if metric not in df.columns or 'severity_category' not in df.columns:
        return None  # Required data not available
    
    # Calculate average metric by severity
    avg_by_severity = df.groupby('severity_category')[metric].mean().reset_index()
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=avg_by_severity['severity_category'],
        y=avg_by_severity[metric],
        marker_color=['green' if s == 'mild' else 'orange' if s == 'moderate' else 'red' 
                     for s in avg_by_severity['severity_category']]
    ))
    
    fig.update_layout(
        title=f'Average {metric.replace("_", " ").title()} by Severity',
        xaxis_title='Severity Category',
        yaxis_title=f'Average {metric.replace("_", " ").title()}',
        height=500
    )
    
    return fig

def plot_word_count_distribution(df):
    """Plot the distribution of word counts"""
    if 'word_count' not in df.columns:
        return None  # Word count data not available
    
    # Create histogram
    fig = px.histogram(
        df,
        x='word_count',
        nbins=30,
        color_discrete_sequence=['blue']
    )
    
    fig.update_layout(
        title='Post Length Distribution',
        xaxis_title='Word Count',
        yaxis_title='Number of Posts',
        height=500
    )
    
    # Add median line
    median_word_count = df['word_count'].median()
    fig.add_vline(x=median_word_count, line_dash="dash", line_color="red", 
                 annotation_text=f"Median: {median_word_count:.0f} words")
    
    return fig

def display_example_posts(df, sentiment_type, n=3):
    """Display example posts for a given sentiment type"""
    if sentiment_type not in df['final_sentiment'].unique():
        return []
    
    # Get sample posts of the specified sentiment
    sample_posts = df[df['final_sentiment'] == sentiment_type].sample(min(n, sum(df['final_sentiment'] == sentiment_type)))
    
    # Return the original text of these posts
    return sample_posts['original_text'].tolist()

def display_post_examples(df):
    """Display example posts based on various criteria"""
    st.markdown('<h2 class="subheader">Post Examples Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore actual post examples filtered by different criteria. This can help validate analysis results
    and understand the human context behind the metrics.
    """)
    
    # Create filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_sentiment = st.selectbox(
            "Filter by Sentiment",
            options=['All'] + df['final_sentiment'].unique().tolist(),
            index=0
        )
    
    with col2:
        if 'severity_category' in df.columns:
            selected_severity = st.selectbox(
                "Filter by Severity",
                options=['All'] + df['severity_category'].unique().tolist(),
                index=0
            )
        else:
            selected_severity = 'All'
    
    with col3:
        if 'post_intention' in df.columns:
            selected_intention = st.selectbox(
                "Filter by Post Intention",
                options=['All'] + df['post_intention'].unique().tolist(),
                index=0
            )
        else:
            selected_intention = 'All'
    
    # Additional mental health category filter
    mh_categories = ['anxiety', 'depression', 'trauma', 'cognitive_distortions',
                     'recovery_indicators', 'support_seeking', 'support_giving']
    selected_category = st.selectbox(
        "Filter by Mental Health Category",
        options=['All'] + mh_categories,
        index=0
    )
    
    # Minimum word count filter
    min_words = st.slider(
        "Minimum Word Count",
        min_value=0,
        max_value=500,
        value=50,
        step=10
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['final_sentiment'] == selected_sentiment]
    if selected_severity != 'All':
        filtered_df = filtered_df[filtered_df['severity_category'] == selected_severity]
    if selected_intention != 'All':
        filtered_df = filtered_df[filtered_df['post_intention'] == selected_intention]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df[f'{selected_category}_count'] > 0]
    
    # Apply word count filter
    filtered_df = filtered_df[filtered_df['word_count'] >= min_words]
    
    # Display results
    if len(filtered_df) == 0:
        st.warning("No posts match the selected criteria")
        return
    
    st.markdown(f"**Found {len(filtered_df)} matching posts**")
    
    # Show sample posts
    sample_size = st.slider(
        "Number of examples to show",
        min_value=1,
        max_value=min(10, len(filtered_df)),
        value=3,
        step=1
    )
    
    sample_posts = filtered_df.sample(sample_size)[['original_text', 'final_sentiment', 
                                                  'severity_category', 'post_intention']]
    
    for idx, post in sample_posts.iterrows():
        with st.expander(f"Post ({post['final_sentiment'].title()} - {post.get('severity_category', 'N/A')} - {post.get('post_intention', 'N/A')}"):
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                {post['original_text']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", post['final_sentiment'].title())
            with col2:
                if 'severity_category' in post:
                    st.metric("Severity", post['severity_category'].title())
            with col3:
                if 'post_intention' in post:
                    st.metric("Intention", post['post_intention'].replace('_', ' ').title())

def display_about():
    """Display project information"""
    st.markdown('<h2 class="subheader">About the Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Mental Health Discourse Analysis Dashboard**
    
    This project analyzes mental health discussions from Reddit using advanced NLP techniques to understand:
    - Sentiment patterns in mental health conversations
    - Prevalence of different mental health conditions
    - Support-seeking vs support-giving behaviors
    - Relationships between discourse characteristics and user engagement
    
    **Methodology:**
    1. **Data Collection:** Reddit posts from mental health-related subreddits
    2. **Text Processing:** Cleaning, lemmatization, and stopword removal
    3. **Sentiment Analysis:** VADER with mental health-specific adjustments
    4. **Mental Health Detection:** Custom lexicon-based indicator counting
    5. **Severity Scoring:** Composite metric combining multiple factors
    6. **Visual Analytics:** Interactive dashboard for exploring results
    
    **Key Features:**
    - Real-time filtering of analysis results
    - Contextual post examples with metadata
    - Temporal trend analysis
    - Engagement pattern visualization
    
    **Data Sources:**
    - Reddit API via Pushshift.io
    - Mental Health lexicon from peer-reviewed research
    - Custom cognitive distortion patterns from CBT literature
    """)
    
    st.markdown("---")
    st.markdown("**Disclaimer:** This is a research prototype. Always consult mental health professionals for clinical assessments.")

def display_user_engagement(df):
    """Display user engagement analysis"""
    st.markdown('<h2 class="subheader">User Engagement Analysis</h2>', unsafe_allow_html=True)
    
    # Check for engagement metrics
    engagement_metrics = []
    for metric in ['score', 'num_comments', 'ups']:
        if metric in df.columns:
            engagement_metrics.append(metric)
    
    if not engagement_metrics:
        st.warning("No engagement metrics available in the dataset")
        return
        
    # Display engagement plots
    for metric in engagement_metrics:
        fig = plot_engagement_by_sentiment(df, metric)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">Mental Health Discourse Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard presents an analysis of mental health discussions using VADER sentiment analysis
    and specialized mental health indicator detection.
    """)
    
    # Load the data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Dashboard Overview", "Sentiment Analysis", "Mental Health Indicators", 
         "User Engagement", "Post Examples", "About the Project"]
    )
    
    # If data is available, show filter options
    if 'subreddit' in df.columns or 'source' in df.columns:
        subreddit_col = 'subreddit' if 'subreddit' in df.columns else 'source'
        all_subreddits = sorted(df[subreddit_col].unique().tolist())
        
        selected_subreddits = st.sidebar.multiselect(
            "Filter by subreddit",
            all_subreddits,
            default=all_subreddits[:3] if len(all_subreddits) > 3 else all_subreddits
        )
        
        if selected_subreddits:
            df_filtered = df[df[subreddit_col].isin(selected_subreddits)]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Display different pages based on selection
    if page == "Dashboard Overview":
        display_dashboard(df_filtered)
        
    elif page == "Sentiment Analysis":
        display_sentiment_analysis(df_filtered)
        
    elif page == "Mental Health Indicators":
        display_mental_health_indicators(df_filtered)
        
    elif page == "User Engagement":
        display_user_engagement(df_filtered)
        
    elif page == "Post Examples":
        display_post_examples(df_filtered)
        
    elif page == "About the Project":
        display_about()

def display_dashboard(df):
    """Display the main dashboard overview"""
    st.markdown('<h2 class="subheader">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Posts", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        positive_pct = sum(df['final_sentiment'] == 'positive') / len(df) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        negative_pct = sum(df['final_sentiment'] == 'negative') / len(df) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        if 'severity_score' in df.columns:
            avg_severity = df['severity_score'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Severity Score", f"{avg_severity:.2f}/10")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            neutral_pct = sum(df['final_sentiment'] == 'neutral') / len(df) * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Neutral Sentiment", f"{neutral_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_dist_fig = plot_sentiment_distribution(df)
        if sentiment_dist_fig:
            st.plotly_chart(sentiment_dist_fig, use_container_width=True)
        else:
            st.warning("Sentiment distribution data not available")
    
    with col2:
        sentiment_time_fig = plot_sentiment_over_time(df)
        if sentiment_time_fig:
            st.plotly_chart(sentiment_time_fig, use_container_width=True)
        else:
            st.warning("Temporal sentiment data not available")
    
    # Second row of visualizations
    if 'severity_score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            mh_indicators_fig = plot_mental_health_indicators(df)
            if mh_indicators_fig:
                st.plotly_chart(mh_indicators_fig, use_container_width=True)
            else:
                st.warning("Mental health indicator data not available")
        
        with col2:
            severity_fig = plot_severity_by_subreddit(df)
            if severity_fig:
                st.plotly_chart(severity_fig, use_container_width=True)
            else:
                st.warning("Severity score data not available")

def display_sentiment_analysis(df):
    """Display detailed sentiment analysis"""
    st.markdown('<h2 class="subheader">Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This section explores the sentiment patterns in mental health discussions. The VADER sentiment analyzer
    has been used with specialized adjustments for mental health terminology.
    """)
    
    # Sentiment statistics
    st.markdown("### Sentiment Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_counts = df['final_sentiment'].value_counts()
        st.write("Sentiment Distribution")
        sentiment_pie = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        )
        st.plotly_chart(sentiment_pie, use_container_width=True)
    
    with col2:
        if 'adjusted_sentiment_score' in df.columns:
            st.write("Sentiment Score Distribution")
            sentiment_hist = px.histogram(
                df, 
                x='adjusted_sentiment_score',
                nbins=20,
                color_discrete_sequence=['blue']
            )
            sentiment_hist.update_layout(
                xaxis_title='Sentiment Score (-1 to 1)',
                yaxis_title='Number of Posts'
            )
            st.plotly_chart(sentiment_hist, use_container_width=True)
        elif 'sentiment_score' in df.columns:
            st.write("Sentiment Score Distribution")
            sentiment_hist = px.histogram(
                df, 
                x='sentiment_score',
                nbins=20,
                color_discrete_sequence=['blue']
            )
            sentiment_hist.update_layout(
                xaxis_title='Sentiment Score (-1 to 1)',
                yaxis_title='Number of Posts'
            )
            st.plotly_chart(sentiment_hist, use_container_width=True)
    
    with col3:
        # Display average sentiment by subreddit
        subreddit_col = 'subreddit' if 'subreddit' in df.columns else 'source'
        score_col = 'adjusted_sentiment_score' if 'adjusted_sentiment_score' in df.columns else 'sentiment_score'
        
        if score_col in df.columns:
            st.write(f"Average Sentiment by {subreddit_col.title()}")
            avg_sentiment = df.groupby(subreddit_col)[score_col].mean().reset_index()
            avg_sentiment.columns = [subreddit_col, 'avg_score']
            
            # Color threshold for sentiment scores
            avg_sentiment['color'] = avg_sentiment['avg_score'].apply(
                lambda x: 'green' if x > 0.05 else 'red' if x < -0.05 else 'gray'
            )
            
            sentiment_bar = px.bar(
                avg_sentiment,
                x=subreddit_col,
                y='avg_score',
                color='color',
                color_discrete_map={'green': 'green', 'red': 'red', 'gray': 'gray'}
            )
            sentiment_bar.update_layout(
                xaxis_title=subreddit_col.title(),
                yaxis_title='Average Sentiment Score',
                showlegend=False
            )
            st.plotly_chart(sentiment_bar, use_container_width=True)
    
    # Sentiment intensity analysis
    st.markdown("### Sentiment Intensity Analysis")
    intensity_fig = plot_intensity_distribution(df)
    if intensity_fig:
        st.plotly_chart(intensity_fig, use_container_width=True)
    else:
        st.warning("Sentiment intensity data not available")
    
    # Sentiment over time
    st.markdown("### Sentiment Trends Over Time")
    time_fig = plot_sentiment_over_time(df)
    if time_fig:
        st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.warning("Temporal sentiment data not available")

def display_mental_health_indicators(df):
    """Display mental health indicators analysis"""
    st.markdown('<h2 class="subheader">Mental Health Indicators</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This section examines specific mental health indicators and patterns detected in the discourse.
    """)
    
    # Check if we have mental health category data
    mh_categories = ['anxiety', 'depression', 'trauma', 'cognitive_distortions', 
                    'recovery_indicators', 'support_seeking', 'support_giving']
    available_categories = [cat for cat in mh_categories if f'{cat}_count' in df.columns]
    
    if not available_categories:
        st.warning("Mental health indicator data not available")
        return
    
    # Mental health indicators overview
    st.markdown("### Mental Health Indicators Overview")
    indicators_fig = plot_mental_health_indicators(df)
    if indicators_fig:
        st.plotly_chart(indicators_fig, use_container_width=True)
    
    # Mental health severity analysis
    if 'severity_score' in df.columns and 'severity_category' in df.columns:
        st.markdown("### Mental Health Severity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity score distribution
            severity_hist = px.histogram(
                df,
                x='severity_score',
                nbins=20,
                color_discrete_sequence=['darkred']
            )
            severity_hist.update_layout(
                title='Severity Score Distribution',
                xaxis_title='Severity Score (0-10)',
                yaxis_title='Number of Posts'
            )
            severity_hist.add_vline(x=3, line_dash="dash", line_color="green")
            severity_hist.add_vline(x=6, line_dash="dash", line_color="red")
            st.plotly_chart(severity_hist, use_container_width=True)
        
        with col2:
            # Severity category distribution
            severity_counts = df['severity_category'].value_counts()
            severity_pie = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                color=severity_counts.index,
                color_discrete_map={'mild': 'green', 'moderate': 'orange', 'severe': 'red'}
            )
            severity_pie.update_layout(title='Severity Category Distribution')
            st.plotly_chart(severity_pie, use_container_width=True)
        
        # Severity by subreddit
        st.markdown("### Severity by Subreddit")
        severity_sub_fig = plot_severity_by_subreddit(df)
        if severity_sub_fig:
            st.plotly_chart(severity_sub_fig, use_container_width=True)
    
    # Post intention analysis
    if 'post_intention' in df.columns:
        st.markdown("### Post Intention Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Post intention distribution
            intention_counts = df['post_intention'].value_counts()
            intention_pie = px.pie(
                values=intention_counts.values,
                names=intention_counts.index,
                color=intention_counts.index,
                color_discrete_map={
                    'support_seeking': 'blue', 
                    'support_giving': 'green', 
                    'other': 'gray'
                }
            )
            intention_pie.update_layout(title='Post Intention Distribution')
            st.plotly_chart(intention_pie, use_container_width=True)
        
        with col2:
            # Sentiment distribution by intention
            intention_sentiment = pd.crosstab(
                df['post_intention'], 
                df['final_sentiment'],
                normalize='index'
            ) * 100
            
            fig = px.bar(
                intention_sentiment,
                x=intention_sentiment.index,
                y=intention_sentiment.columns,
                title='Sentiment Distribution by Post Intention',
                labels={'x': 'Post Intention', 'y': 'Percentage'},
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            )
            fig.update_layout(barmode='stack')
            st.plotly_chart(fig, use_container_width=True)

# Run the main function
if __name__ == "__main__":
    main()