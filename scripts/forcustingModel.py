import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class ForecastingModel:
    def __init__(self, data, test_ratio=0.2, window_size=60):
        self.data = data['Close']  # Use the closing price data
        self.train_size = int(len(self.data) * (1 - test_ratio))
        self.train = self.data[:self.train_size]
        self.test = self.data[self.train_size:]
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def evaluate_forecast(self, y_true, y_pred):
        actual_data = self.test.iloc[:len(y_pred)]  # Slice to match forecast length
        mae = mean_absolute_error(actual_data, y_pred)
        rmse = np.sqrt(mean_squared_error(actual_data, y_pred))
        # mape = np.mean(np.abs((actual_data - y_pred) / actual_data)) * 100
        return {'MAE': mae, 'RMSE': rmse}

    def plot_forecast(self,forecast, model_name):
        # Ensure forecast has the same length as the test set
        actual_data = self.test.iloc[:len(forecast)]  # Slice to match forecast length

        plt.figure(figsize=(10, 6))
        plt.plot(actual_data.index, actual_data, label='Actual Price', color="blue")
        plt.plot(actual_data.index, forecast, label=f'{model_name} Forecast', color="orange")
        plt.legend()
        plt.title(f'{model_name} Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.show()

    def prepare_lstm_data(self):
        # Scale the data for LSTM
        train_scaled = self.scaler.fit_transform(self.train.values.reshape(-1, 1))
        test_scaled = self.scaler.transform(self.test.values.reshape(-1, 1))

        # Prepare LSTM data format (X, y pairs)
        def create_lstm_data(data, window_size):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_lstm_data(train_scaled, self.window_size)
        X_test, y_test = create_lstm_data(test_scaled, self.window_size)
        

        # Reshape data for LSTM [samples, time steps, features]
        return X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test

    def lstm_forecasting(self):
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_lstm_data()

        # Define the LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        # Compile and fit the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions and inverse scale
        forecast_scaled = model.predict(X_test)
        forecast = self.scaler.inverse_transform(forecast_scaled)
        return forecast.flatten(),y_test, model
