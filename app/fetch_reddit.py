import pandas as pd
from datetime import datetime  
import praw
from prawcore.exceptions import RequestException, ResponseException, ServerError, Forbidden
import os

def fetch_reddit_posts(stock_name):
    try: 
        reddit = praw.Reddit(
            client_id=os.environ.get("REDDIT_CLIENT_ID"),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
            username=os.environ.get("REDDIT_USERNAME"),
            password=os.environ.get("REDDIT_PASSWORD"),
            user_agent=os.environ.get("REDDIT_USER_AGENT")
        )
        subreddit = reddit.subreddit("all")
        posts = subreddit.search(stock_name, limit=20, time_filter="day")
        print("Searching Reddit for:", stock_name)

        data = []
        for post in posts:
            timestamp = datetime.fromtimestamp(post.created_utc)
            data.append({"text": post.title + " " + post.selftext,
                         "timestamp": timestamp})

        return pd.DataFrame(data)

    except Exception as e:
            print(f"Reddit fetch error: {type(e).__name__}: {e}")
            return pd.DataFrame() 