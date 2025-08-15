import yfinance as yf
import pandas as pd

def get_hourly_price_data(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    df = ticker.history(period="2d", interval="1h")
    df.reset_index(inplace=True)
    return df[["Datetime", "Close"]]
