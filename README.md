# Time Series Forecasting for Portfolio Management

This project demonstrates time series forecasting to optimize an investment portfolio. It leverages historical data from Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) to make data-driven portfolio allocation decisions.

## Project Structure
``` bush
├── .vscode/                      # VSCode settings
├── .github/workflows/            # CI/CD for unit tests
├── notebooks/                    # Jupyter notebooks for EDA and modeling
│   ├── __init__.py               # Initialization file
│   └── EDA.ipynb                 # EDA notebook
├── scripts/                      # Scripted pipelines for reproducible runs
│   ├── __init__.py               # Initialization file
│   ├── data_cleaner.py           # Data cleaning script
│   ├── data_exploration.py       # Data exploration script
│   └── data_loader.py            # Data loading script
├── src/                          # Main codebase
│   └── __init__.py               # Initialization file
├── tests/                        # Unit tests for project modules
├── week11/                       # porject env
├── .gitignore                    # Git ignore file
├── README.md                     # Project overview
└── requirements.txt              # Dependencies
```
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