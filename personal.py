import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Function to fetch stock and commodity data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Function to calculate daily returns
def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe_ratio

# Function to visualize portfolio composition
def plot_portfolio_composition(weights, tickers):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    ax.set_title("Portfolio Composition")
    st.pyplot(fig)

# Monte Carlo simulation for portfolio forecasting
def monte_carlo_simulation(returns, weights, days=63, simulations=500):
    portfolio_mean = np.sum(returns.mean() * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    simulated_end_values = []

    for _ in range(simulations):
        daily_returns = np.random.normal(portfolio_mean / 252, portfolio_volatility / np.sqrt(252), days)
        portfolio_path = np.cumprod(1 + daily_returns) - 1
        simulated_end_values.append(portfolio_path[-1])
    
    return simulated_end_values

# Visualization for portfolio forecast
def plot_forecast(portfolio_cum_returns, forecast_values):
    plt.figure(figsize=(10, 6))
    days = np.arange(len(portfolio_cum_returns))
    forecast_days = np.arange(len(portfolio_cum_returns), len(portfolio_cum_returns) + 63)
    
    # Plot historical portfolio performance
    plt.plot(days, portfolio_cum_returns, label="Historical Performance", color="blue")
    
    # Monte Carlo simulation forecast
    lower_bound = np.percentile(forecast_values, 5)
    upper_bound = np.percentile(forecast_values, 95)
    median_forecast = np.percentile(forecast_values, 50)
    
    plt.fill_between(forecast_days, lower_bound, upper_bound, color="orange", alpha=0.3, label="90% Confidence Interval")
    plt.axhline(median_forecast, linestyle="--", color="red", label="Median Forecast")
    
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value Change (%)")
    plt.title("Portfolio Performance with Forecast")
    plt.legend()
    st.pyplot(plt)

# Streamlit dashboard
st.title("Expanded Stock and Commodity Portfolio Analysis with Forecast")
st.sidebar.header("Portfolio Settings")

# Inputs
default_tickers = "AAPL, MSFT, TSLA, GLD, SLV"
tickers = st.sidebar.text_input(
    "Enter stock and commodity tickers (comma-separated)",
    default_tickers,
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

tickers_list = tickers.split(", ")
weights_input = st.sidebar.text_input(
    "Enter portfolio weights (comma-separated)",
    ",".join(["{:.2f}".format(1/len(tickers_list))] * len(tickers_list))
)
weights = np.array([float(x) for x in weights_input.split(",")])

# Ensure weights sum to 1
if not np.isclose(weights.sum(), 1):
    st.sidebar.error("Weights must sum to 1.")
    st.stop()

# Fetch and display data
if st.sidebar.button("Analyze Portfolio"):
    # Data retrieval
    stock_data = fetch_data(tickers_list, start_date, end_date)
    st.subheader("Stock and Commodity Price Data")
    st.write(stock_data)

    # Calculate returns and portfolio metrics
    stock_returns = calculate_returns(stock_data)
    port_return, port_std_dev, port_sharpe = calculate_portfolio_metrics(stock_returns, weights)

    # Display metrics
    st.subheader("Portfolio Metrics")
    st.write(f"**Expected Annual Return**: {port_return:.2%}")
    st.write(f"**Portfolio Standard Deviation (Risk)**: {port_std_dev:.2%}")
    st.write(f"**Sharpe Ratio**: {port_sharpe:.2f}")

    # Visualization
    st.subheader("Portfolio Composition")
    plot_portfolio_composition(weights, tickers_list)

    st.subheader("Stock and Commodity Correlation Heatmap")
    corr = stock_returns.corr()
    st.write(corr.style.background_gradient(cmap='coolwarm'))

    st.subheader("Portfolio Performance")
    portfolio_daily_returns = (stock_returns @ weights)
    portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod() - 1
    st.line_chart(portfolio_cum_returns * 100)

    # Forecasting
    st.subheader("Portfolio Forecast")
    simulated_end_values = monte_carlo_simulation(stock_returns, weights)
    plot_forecast(portfolio_cum_returns.values, simulated_end_values)

# Footer
st.markdown("""
---
*Powered by Streamlit and yFinance.*
""")
