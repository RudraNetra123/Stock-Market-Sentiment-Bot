# **Stock Market Sentiment Bot**

A Python-based bot that gathers sentiment from financial news, Twitter, and Reddit, analyzes it using NLP techniques, predicts sentiment using an LSTM model, and visualizes how sentiment correlates with real stock prices.

## Features

- Fetches live data from News APIs, Twitter, and Reddit  
- Performs sentiment analysis on text data in real time  
- Uses an LSTM neural network to predict stock sentiment  
- Integrates stock price data using Finnhub API  
- Visualizes sentiment trends vs. actual price movements  
- Comes with a Flask web app dashboard for interactive use  

**Installation:**
```
  git clone https://github.com/RudraNetra123/Stock-Market-Sentiment-Bot.git
  cd Stock-Market-Sentiment-Bot
  python -m venv venv
  source venv/bin/activate   # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
```

**Setup:**
Create a .env file in the project root and add your API keys:
```
  NEWS_API_KEY=your_key_here
  TWITTER_BEARER_TOKEN=your_key_here
  REDDIT_CLIENT_ID=your_key_here
  REDDIT_CLIENT_SECRET=your_key_here
  FINNHUB_API_KEY=your_key_here
```
**Usage:**
Run the Flask app:
```
  python -m app.main
  Open your browser at:
  http://127.0.0.1:5000
```
**Model:**
The LSTM model is trained on labeled stock sentiment datasets. You can retrain or fine-tune it using the model/train.py script.

**TODO(Future Work):**
```
  Add support for more stocks & tickers
  Improve model accuracy with more data
  Schedule daily automated sentiment reports
  Deploy to Heroku / AWS / GCP
```
**Disclaimer:**
This project is for educational purposes only and should not be used as financial advice.
