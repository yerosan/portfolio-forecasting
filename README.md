# Time Series Forecasting for Portfolio Management

This project demonstrates time series forecasting to optimize an investment portfolio. It leverages historical data from Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) to make data-driven portfolio allocation decisions.

## Project Structure
- **notebooks/**: Jupyter notebooks for detailed analysis (e.g., `EDA.ipynb`).
- **scripts/**: Contains modules for data processing:
  - `data_loader.py`: Loads data.
  - `data_cleaner.py`: Cleans and prepares data.
  - `data_exploration.py`: Performs exploratory data analysis.
- **src/**: Core source code for the project.
- **tests/**: Contains unit tests.
- **week11/**: Weekly progress folder.

## Progress
**Completed:**
- **Data Preprocessing**: Loaded, cleaned, and explored TSLA, BND, and SPY data.
- **Exploratory Analysis**: Visualized trends, rolling means, and volatility.

**Upcoming:**
- **Initial Forecasting**: Built preliminary ARIMA, SARIMA, and LSTM models for TSLA.
- **Model Refinement**: Optimize models and extend to BND and SPY.
- **Portfolio Optimization**: Balance risk and return using forecasted data.

## Getting Started
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

- *Explore notebooks*: See the notebooks/ directory for in-depth analysis.