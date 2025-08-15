import datetime
import requests
import pandas as pd
import os
API_KEY = os.environ.get("FINNHUB_API_KEY")
BASE = "https://finnhub.io/api/v1/company-news"

def fetch_news(symbol: str, lookback_days: int = 7,limit:int = 50) -> pd.DataFrame:
    """To fetch rfecent news articles for the given stock."""
    today = datetime.date.today()
    _from = (today - datetime.timedelta(days=lookback_days)).isoformat()
    _to = today.isoformat()

    url = f"{BASE}?symbol={symbol.upper()}&from={_from}&to={_to}&token={API_KEY}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()[:50] 

        texts = []
        timestamps = []

        for d in data:
            if 'headline' in d:
                texts.append(f"{d['headline']} {d.get('summary', '')}")
                timestamps.append(datetime.datetime.fromtimestamp(d['datetime']))

        return pd.DataFrame({
            "text": texts,
            "timestamp": pd.to_datetime(timestamps, errors='coerce')
        })

    except Exception as e:
        print(f"[ERROR] Fetching news failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["text"])
