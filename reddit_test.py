import praw
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_reddit_connection():
    try:
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Test by getting 5 hot posts from r/mentalhealth
        subreddit = reddit.subreddit('mentalhealth')
        print("Successfully connected to Reddit API!")
        print("\nTest fetching 5 posts from r/mentalhealth:")
        print("-" * 50)
        
        for post in subreddit.hot(limit=5):
            print(f"Title: {post.title[:100]}...")
            print(f"Score: {post.score}")
            print("-" * 50)
            
        return True
        
    except Exception as e:
        print(f"Error connecting to Reddit API: {str(e)}")
        return False

if __name__ == "__main__":
    test_reddit_connection()