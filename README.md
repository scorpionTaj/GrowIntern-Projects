# Apple Stock Price Prediction

## Overview
This project uses an LSTM (Long Short-Term Memory) neural network to predict Apple stock prices over the last 20 years. The application is built using Streamlit and provides visualizations for stock data along with predictions and key performance metrics.

## Technologies Used
- **Streamlit**: For building the interactive web app.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Keras**: For building and loading the LSTM model.
- **Scikit-learn**: For scaling and evaluation metrics.
- **Yahoo Finance API (yfinance)**: For fetching historical stock data.
- **Matplotlib**: For creating visualizations.
- **Datetime**: For managing date inputs.

## Features
- **Stock Data Visualization**: Display stock data including 100-day, 200-day, and 250-day moving averages.
- **LSTM Model Prediction**: Predict future stock prices based on historical data.
- **Performance Metrics**: Evaluate the model using metrics like Mean Squared Error (MSE) and R² Score.
- **Interactive UI**: Enter a stock symbol, choose a date range, and visualize predictions.

## Installation
1. Clone the repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.

## Usage
1. Enter the stock symbol (e.g., AAPL) in the input field.
2. Select the start and end dates for the data range.
3. View the stock data, moving averages, and model predictions.
4. Evaluate model performance using displayed metrics.

## Performance Metrics
- **Mean Squared Error (MSE)**
- **R-squared (R²) Score**

## Developer
- Developed by [Tajeddine Bourhim](https://tajeddine-portfolio.netlify.app/)

---

# Car Price Prediction

## Overview
This project predicts the price of a car based on various features such as mileage, year, engine size, and more. The prediction is performed using a Linear Regression model and the application is built using Streamlit.

## Technologies Used
- **Streamlit**: For building the interactive web app.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For preprocessing, building the model pipeline, and evaluation.
- **Pickle**: For loading the pre-trained model.

## Features
- **Car Price Prediction**: Predict car prices based on inputs like year, mileage, engine size, etc.
- **Interactive UI**: Input car features and get an immediate price prediction.
- **Performance Metrics**: Evaluate the model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

## Installation
1. Clone the repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.

## Usage
1. Enter the details of the car (year, mileage, engine size, etc.).
2. Click on "Predict Price" to get the estimated car price.
3. Explore the accuracy of predictions using the provided metrics.

## Performance Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²) Score**

## Developer
- Developed by [Tajeddine Bourhim](https://tajeddine-portfolio.netlify.app/)
