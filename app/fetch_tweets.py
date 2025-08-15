import tweepy
import pandas as pd
import datetime
import os

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(keyword):
    print("Fetching tweets for:", keyword)
    try:
        query = f"{keyword} lang:en -is:retweet"
        tweets = client.search_recent_tweets(query=query, max_results=20, tweet_fields=["text", "created_at"])

        if tweets and tweets.data:
            texts = [tweet.text for tweet in tweets.data]
            timestamps = [tweet.created_at for tweet in tweets.data]
            return pd.DataFrame({
                "text": texts,
                "timestamp": pd.to_datetime(timestamps, errors='coerce')
            })
        else:
            print("No tweets found.")
            return pd.DataFrame(columns=["text", "timestamp"])
    except Exception as e:
        print("Error fetching tweets:", e)
        return pd.DataFrame(columns=["text", "timestamp"])