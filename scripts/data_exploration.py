# src/data/data_exploration.py
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

class DataExplorer:
    @staticmethod
    def basic_statistics(data):
        """Print basic statistics for each asset."""
        for ticker, df in data.items():
            print(f"\nBasic Statistics for {ticker}:")
            print(df.describe())

    @staticmethod
    def plot_closing_price(data):
        """Plot the closing prices of each asset."""
        for ticker, df in data.items():
            df['Close'].plot(title=f"{ticker} Closing Price Over Time", figsize=(10, 6))
            plt.xlabel("Date")
            plt.ylabel("Closing Price")
            plt.show()
    
    

    @staticmethod
    def calculate_daily_returns(data):
        """Calculate daily returns as a percentage change."""
        daily_returns = {}
        for ticker, df in data.items():
            daily_returns[ticker] = df['Close'].pct_change().dropna()
        return daily_returns

    @staticmethod
    def plot_daily_volatility(daily_returns):
        """Plot daily volatility for each ticker."""
        for ticker, returns in daily_returns.items():
            returns.plot(title=f"{ticker} Daily Returns", figsize=(10, 6))
            plt.xlabel("Date")
            plt.ylabel("Daily Return")
            plt.show()

    @staticmethod
    def calculate_rolling_volatility(data, window=30):
        """Calculate and plot rolling mean and volatility (std deviation)."""
        for ticker, df in data.items():
            rolling_mean = df['Close'].rolling(window=window).mean()
            rolling_std = df['Close'].rolling(window=window).std()
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['Close'], label='Closing Price')
            plt.plot(rolling_mean, label=f'{window}-Day Rolling Mean')
            plt.plot(rolling_std, label=f'{window}-Day Rolling Std (Volatility)')
            plt.title(f"{ticker} - Rolling Volatility and Mean")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.show()

    @staticmethod
    def detect_outliers(data, threshold=3):
        """
        Detects outliers based on Z-score method for each ticker's daily returns.
        Prints out the dates and values of detected outliers.
        
        Parameters:
        - data (dict): Dictionary of DataFrames with each containing 'Close' prices.
        - threshold (float): The Z-score threshold to identify outliers.
        """
        outliers = {}

        for ticker, df in data.items():
            # Ensure 'Close' column exists and has data
            if 'Close' not in df.columns:
                print(f"Data for {ticker} does not contain 'Close' prices. Skipping.")
                continue

            # Calculate daily returns and drop NaN values
            daily_returns = df['Close'].pct_change().dropna()
            
            # Calculate Z-scores for daily returns
            z_scores = (daily_returns - daily_returns.mean()) / daily_returns.std()

            # Identify outliers based on the threshold
            ticker_outliers = daily_returns[z_scores.abs() > threshold]
            
            # Store outliers in the dictionary
            outliers[ticker] = ticker_outliers
            print(f"\nOutliers for {ticker}:")
            print(ticker_outliers)

        return outliers
    
    @staticmethod
    def decompose_time_series(data, model='additive'):
        """Decompose the time series into trend, seasonal, and residual components."""
        for ticker, df in data.items():
            decomposition = seasonal_decompose(df['Close'], model=model, period=30)
            decomposition.plot()
            plt.suptitle(f"{ticker} - Decomposition of Time Series")
            plt.show()