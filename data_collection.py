import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import List
import time

# Load environment variables
load_dotenv()

class RedditDataCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
    
    def collect_posts(self, subreddits: List[str], post_limit: int = 500) -> pd.DataFrame:
        """
        Collect posts from specified subreddits
        """
        all_posts = []
        
        for subreddit_name in subreddits:
            print(f"Collecting posts from r/{subreddit_name}...")
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Collect posts from different categories
            for category in ['hot', 'new', 'top']:
                posts = getattr(subreddit, category)(limit=post_limit)
                
                for post in posts:
                    post_data = {
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'post_id': post.id,
                        'category': category,
                        'upvote_ratio': post.upvote_ratio
                    }
                    all_posts.append(post_data)
                
                # Respect Reddit's API rate limits
                time.sleep(2)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_posts)
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f'data/raw/reddit_posts_{timestamp}.csv', index=False)
        
        print(f"\nCollection completed! Total posts collected: {len(df)}")
        return df

if __name__ == "__main__":
    # Initialize collector
    collector = RedditDataCollector()
    
    # Define target subreddits
    target_subreddits = [
        'mentalhealth',
        'anxiety',
        'depression'
    ]
    
    # Collect data
    df = collector.collect_posts(target_subreddits)
    
    # Display basic info about collected data
    print("\nData Collection Summary:")
    print("-" * 50)
    print(f"Total number of posts: {len(df)}")
    print("\nPosts per subreddit:")
    print(df['subreddit'].value_counts())
    print("\nDate range:")
    print(f"Earliest post: {df['created_utc'].min()}")
    print(f"Latest post: {df['created_utc'].max()}")