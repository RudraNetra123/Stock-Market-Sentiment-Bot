import requests
import pandas as pd
import datetime
import os

API_KEY = os.environ.get("MARKETAUX_API_KEY")
BASE_URL = "https://api.marketaux.com/v1/news/all"

def fetch_marketaux_news(symbol: str, limit: int = 30) -> pd.DataFrame:
    date = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
    
    params = {
        "symbols": symbol.upper(),
        "published_after": date,
        "language": "en",
        "api_token": API_KEY,
        "limit": limit
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])[:limit]

        texts = []
        timestamps = []

        for item in data:
            texts.append(f"{item['title']} {item.get('description', '')}")
            timestamps.append(item.get("published_at", None))

        df = pd.DataFrame({
            "text": texts,
            "timestamp": pd.to_datetime(timestamps, errors='coerce')
        })
        return df

    except Exception as e:
        print(f"[Marketaux ERROR] {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["text"])
