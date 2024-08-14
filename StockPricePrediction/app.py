import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.sidebar.title("Stock Price Prediction App 📈")
st.sidebar.subheader("About the Developer")
st.sidebar.markdown(
    "Developed by [Tajeddine Bourhim ](https://tajeddine-portfolio.netlify.app/)."
)
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/scorpionTaj)"
)
st.sidebar.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/tajeddine-bourhim/)"
)

st.sidebar.subheader("📚 About This App")
st.sidebar.markdown(
    "This is a stock price prediction app that uses a Long Short-Term Memory (LSTM) neural network to predict the closing price of a stock. The app uses the Yahoo Finance API to fetch the stock data."
)

stock = st.text_input("Enter the stock symbol (e.g. AAPL):")

if stock:
    end = st.date_input("End Date", datetime.now())
    start = st.date_input("Start Date", datetime(end.year - 20, end.month, end.day))

    apple_data = yf.download(stock, start, end)

    if not apple_data.empty:
        model = load_model("Latest_stock_price_prediction.keras")
        st.subheader("📊 Stock Data")
        st.write(apple_data)
    else:
        st.error("No data found for the given stock symbol.")
else:
    st.warning("Please enter a stock symbol.")

split_len = int(len(apple_data) * 0.8)
x_test = pd.DataFrame(apple_data.Close[split_len:])


def graph_plotting(figsize, values, data, extra_data=0, extra_dataset=None):
    """
    Function to plot graphs with given parameters.
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(values, "Red")
    plt.plot(data.Close, "b")
    if extra_data:
        plt.plot(extra_dataset)
    return fig


try:
    # Display the original close price and moving average for 250 days
    st.subheader("📈 Original Close Price and 250-Day Moving Average")
    apple_data["MA_for_250_days"] = apple_data.Close.rolling(250).mean()
    st.pyplot(graph_plotting((15, 6), apple_data["MA_for_250_days"], apple_data, 0))

    # Display the original close price and moving average for 200 days
    st.subheader("📈 Original Close Price and 200-Day Moving Average")
    apple_data["MA_for_200_days"] = apple_data.Close.rolling(200).mean()
    st.pyplot(graph_plotting((15, 6), apple_data["MA_for_200_days"], apple_data, 0))

    # Display the original close price and moving average for 100 days
    st.subheader("📈 Original Close Price and 100-Day Moving Average")
    apple_data["MA_for_100_days"] = apple_data.Close.rolling(100).mean()
    st.pyplot(graph_plotting((15, 6), apple_data["MA_for_100_days"], apple_data, 0))

    # Display the original close price and moving average for 100 days and 250 days
    st.subheader("📈 Original Close Price and 100-Day and 250-Day Moving Average")
    st.pyplot(
        graph_plotting(
            (15, 6),
            apple_data["MA_for_100_days"],
            apple_data,
            1,
            apple_data["MA_for_250_days"],
        )
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[["Close"]])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100 : i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)

    # Inverse transform the predictions and test data to original scale
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Create a DataFrame to hold the original test data and predictions
    ploting_data = pd.DataFrame(
        {
            "original_test_data": inv_y_test.reshape(-1),
            "predictions": inv_pre.reshape(-1),
        },
        index=apple_data.index[split_len + 100 :],
    )

    # Display the original values vs predicted values
    st.subheader("🔍 Original values vs Predicted values")
    st.write(ploting_data)

    # Plot the original close price vs predicted close price
    st.subheader("📉 Original Close Price vs Predicted Close price")
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([apple_data.Close[: split_len + 100], ploting_data], axis=0))
    plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)

    # Calculate and display performance metrics
    mse = mean_squared_error(inv_y_test, inv_pre)
    r2 = r2_score(inv_y_test, inv_pre)
    st.subheader("📊 Performance Metrics")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

except Exception as e:
    st.error(f"An error occurred: {e}")
