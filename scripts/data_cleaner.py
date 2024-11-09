# src/data/data_cleaner.py

import pandas as pd

class DataCleaner:
    @staticmethod
    def clean_data(data):
        """Clean the provided data by handling missing values and checking data types."""
        cleaned_data = {}
        for ticker, df in data.items():
            df = df.dropna()  # Drop rows with any missing values
            df = df.astype({'Close': 'float64'})  # Ensure 'Close' is of type float
            cleaned_data[ticker] = df
        return cleaned_data
