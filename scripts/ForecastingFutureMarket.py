import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
class ForecastingFutureMarketTrends:
    def __init__(self, model, historical_data, forecast_periods=365):
        self.model = model  # Trained LSTM model
        self.historical_data = historical_data  # Historical Tesla prices
        self.forecast_periods = forecast_periods  # Days to forecast
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Scale historical data
        self.scaled_data = self.scaler.fit_transform(historical_data.values.reshape(-1, 1))

    def generate_forecast(self):
        # Prepare data for LSTM forecasting
        forecast_input = self.scaled_data[-60:]  # Start with the last observed data (adjust as needed)
        forecast_output = []

        for _ in range(self.forecast_periods):
            forecast_input_reshaped = forecast_input.reshape(1, forecast_input.shape[0], 1)
            predicted_scaled = self.model.predict(forecast_input_reshaped)
            forecast_output.append(predicted_scaled[0, 0])
            
            # Update input sequence for the next prediction
            forecast_input = np.append(forecast_input[1:], predicted_scaled, axis=0)
        
        # Scale forecasted values back to the original range
        forecast_output = np.array(forecast_output).reshape(-1, 1)
        forecast_original = self.scaler.inverse_transform(forecast_output)

        # Plot historical data and forecast
        plt.figure(figsize=(12, 6))
        plt.plot(self.historical_data.index, self.historical_data, label="Historical Price", color="blue")
        plt.plot(pd.date_range(self.historical_data.index[-1], periods=self.forecast_periods, freq="D"),
                 forecast_original, label="Forecasted Price", color="orange")
        plt.title("TSLA Forecasted Prices (LSTM Model)")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
        
        return forecast_original
