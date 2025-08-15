from dotenv import load_dotenv
load_dotenv() 
from flask import Flask, request, render_template
from .fetch_tweets import fetch_tweets
from .fetch_reddit import fetch_reddit_posts
from .analyze_sent import clean_text, get_sentiment
from .fetch_news import fetch_news
from .fetch_marketaux import fetch_marketaux_news
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import time
import yfinance as yf
from .price_data import get_hourly_price_data
from datetime import datetime
import torch
from model.lstm_model import LSTMRegressor
app = Flask(__name__)

# Loading the LSTM model
lstm_model = LSTMRegressor()
model_path = os.path.join('model', 'saved_model.pth')
if os.path.exists(model_path):
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()
else:
    lstm_model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    counts = {}
    chart_path = None
    stock_name = ""
    interpretation = None
    sentiment_score = None
    market_status = None 
    trend_chart_path = None
    chart_path = None
    prediction_text = None 
    history_file = 'sentiment_price_history.csv'

    if request.method == 'POST':
        stock_name = request.form['stock_name']
        print("User typed:", stock_name)
        try:
            stock_data = yf.Ticker(stock_name)
            live_price = stock_data.info.get("currentPrice")  
            price_change = stock_data.info.get("regularMarketChangePercent")
            market_status = f"${live_price:.2f} ({price_change:.2f}%)" if live_price else "N/A"
        except Exception as e:
            live_price = None
            market_status = "Unable to fetch live price"
        
        tweets_df = fetch_tweets(stock_name)
        reddit_posts = fetch_reddit_posts(stock_name)
        news_df = fetch_news(stock_name)
        marketaux_df = fetch_marketaux_news(stock_name, limit=20)

        reddit_df = pd.DataFrame(reddit_posts, columns=['text'])
        combined_df = pd.concat([tweets_df[['text']], reddit_df, news_df, marketaux_df], ignore_index=True)

        def sentiment_to_score(sent): 
            return 1 if sent == "Positive" else -1 if sent == "Negative" else 0

        if not combined_df.empty:
            combined_df['cleaned'] = combined_df['text'].apply(clean_text)
            combined_df['sentiment'] = combined_df['cleaned'].apply(get_sentiment)
            combined_df['score'] = combined_df['sentiment'].apply(sentiment_to_score)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
            combined_df = combined_df.dropna(subset=['timestamp'])
            combined_df['hour'] = combined_df['timestamp'].dt.floor('H')

            hourly_score = combined_df.groupby('hour')['score'].sum().reset_index()
            hourly_score = hourly_score.sort_values("hour").reset_index(drop=True)
            hourly_score['smoothed_score'] = hourly_score['score'].rolling(window=3, center=True, min_periods=1).mean()
            hourly_score["delta_score"] = hourly_score["smoothed_score"].diff()
            hourly_score = hourly_score.dropna(subset=["delta_score"])

            counts = combined_df['sentiment'].value_counts().to_dict()
            positive = counts.get("Positive", 0)
            negative = counts.get("Negative", 0)
            neutral = counts.get("Neutral", 0)
            sentiment_score = positive - negative

            if sentiment_score > 3:
                interpretation = "Market sentiment appears **Bullish** 📈"
            elif sentiment_score < -3:
                interpretation = "Market sentiment appears **Bearish** 📉"
            else:
                interpretation = "Market sentiment is **Neutral or Mixed** 🤔"

            color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
            colors = [color_map.get(label, 'blue') for label in counts.keys()]
            plt.figure(figsize=(6, 4))
            plt.bar(counts.keys(), counts.values(), color=colors)
            plt.xlabel("Sentiment")
            plt.ylabel("Post Count")
            plt.title(f"Overall Sentiment for '{stock_name.upper()}'")
            plt.tight_layout()
            filename = f"sentiment_chart_{stock_name.upper()}_{int(time.time())}.png"
            save_path = os.path.join('app', 'static', filename)
            plt.savefig(save_path)
            plt.close()
            chart_path = f"/static/{filename}"

            plt.figure(figsize=(8, 4))
            plt.plot(hourly_score['hour'], hourly_score['smoothed_score'], marker='o', linestyle='-')
            plt.xlabel("Time (Hourly)")
            plt.ylabel("Sentiment Score")
            plt.title(f"Sentiment Trend for '{stock_name.upper()}' Over Time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            trend_filename = f"sentiment_trend_{stock_name.upper()}_{int(time.time())}.png"
            trend_path = os.path.join('app', 'static', trend_filename)
            plt.savefig(trend_path)
            plt.close()
            trend_chart_path = f"/static/{trend_filename}"

            price_df = get_hourly_price_data(stock_name)
            hourly_score['hour'] = pd.to_datetime(hourly_score['hour']).dt.round('H').dt.tz_localize(None)
            price_df['Datetime'] = pd.to_datetime(price_df['Datetime']).dt.round('H').dt.tz_localize(None)
            price_df.rename(columns={'Datetime': 'hour'}, inplace=True)

            if not hourly_score.empty and not price_df.empty:
                latest_delta = hourly_score['delta_score'].iloc[-1]

                if len(price_df) >= 2:
                    last_close = price_df['Close'].iloc[-1]
                    prev_close = price_df['Close'].iloc[-2]
                    pct_move = ((last_close - prev_close) / prev_close) * 100

                    with open(history_file, 'a') as f:
                        f.write(f"{latest_delta},{pct_move}\n")

                    if lstm_model is not None and not hourly_score.empty:
                        seq_deltas = hourly_score['delta_score'].dropna().tail(2).values
                        if len(seq_deltas) == 2:
                            seq_tensor = torch.tensor(seq_deltas, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                            with torch.no_grad():
                                pred_pct = lstm_model(seq_tensor).item()
                                prediction_text = f"Predicted next-hour price change (LSTM): {pred_pct:.2f}%"
                        else:
                            prediction_text = "Not enough sentiment history for LSTM prediction."
                    else:
                        prediction_text = "LSTM model not loaded or no data."

                else:
                    prediction_text = "Not enough price data to compute change."

            else:
                prediction_text = "Try a more active stock."

    return render_template(
       "index.html",
        counts=counts,
        chart_path=chart_path,
        stock_name=stock_name,
        interpretation=interpretation,
        sentiment_score=sentiment_score,
        market_status=market_status,
        trend_chart_path=trend_chart_path,
        prediction_text=prediction_text
    )

if __name__ == "__main__":
    app.run(debug=True)
