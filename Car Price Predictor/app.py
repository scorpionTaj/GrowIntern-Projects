import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load pre-trained model
with open("model.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Define the feature columns
feature_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize",
    "transmission",
    "fuelType",
    "Manufacturer",
]


def predict_price(
    year, mileage, tax, mpg, engineSize, transmission, fuelType, Manufacturer
):
    input_df = pd.DataFrame(
        [[year, mileage, tax, mpg, engineSize, transmission, fuelType, Manufacturer]],
        columns=feature_columns,
    )
    prediction = pipeline.predict(input_df)
    return prediction[0][0]


# Streamlit app layout
st.write("Enter the details of the car to predict its price:")

# Input fields
year = st.number_input("Year", min_value=1900, max_value=2100, value=2010)
mileage = st.number_input("Mileage", min_value=0, value=50000)
tax = st.number_input("Tax (£)", min_value=0, value=100)
mpg = st.number_input("MPG", min_value=0, value=50)
engineSize = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)
transmission = st.selectbox(
    "Transmission", options=["Automatic", "Semi-Auto", "Manual"]
)
fuelType = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric", "Hybrid"])
Manufacturer = st.selectbox(
    "Manufacturer",
    options=[
        "toyota",
        "hyundi",
        "ford",
        "BMW",
        "Audi",
        "merc",
        "volkswagen",
        "vauxhall",
    ],
)

# Button to predict
if st.button("🔮 Predict Price"):
    price = predict_price(
        year, mileage, tax, mpg, engineSize, transmission, fuelType, Manufacturer
    )
    st.write(f"The predicted price of the car is £{price:.2f}")

# Developer Info

st.sidebar.title("🚗 Car Price Predictor")
st.sidebar.subheader("About the Developer")
st.sidebar.markdown(
    "Developed by [Tajeddine Bourhim](https://tajeddine-portfolio.netlify.app/)."
)
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/scorpionTaj)"
)
st.sidebar.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/tajeddine-bourhim/)"
)
st.sidebar.subheader("📚 About This App")
st.sidebar.markdown(
    "This app uses a machine learning model to predict the price of a car based on various features."
)
st.sidebar.markdown(
    "Model trained using historical car price data and includes features like year, mileage, and more."
)
