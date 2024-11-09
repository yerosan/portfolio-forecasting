# src/data/data_loader.py

import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = None

    def fetch_data(self, start_date="2015-01-01", end_date="2023-10-31"):
        """Fetch historical data from Yahoo Finance for the given tickers."""
        self.data = {}
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}...")
            self.data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        return self.data
