import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(layout="wide")
st.title("🌫️ AQI Prediction from PM2.5 – Streamlit App")

uploaded_file = st.file_uploader("📂 Upload air quality CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Breakpoints and category
    breakpoints_pm25 = [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500)
    ]

    def calculate_aqi(conc, bps):
        for bp_lo, bp_hi, i_lo, i_hi in bps:
            if bp_lo <= conc <= bp_hi:
                return round(((i_hi - i_lo) / (bp_hi - bp_lo)) * (conc - bp_lo) + i_lo)
        return None

    def categorize_aqi(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"

    # AQI Computation
    df["AQI_PM2.5"] = df["PM2.5"].apply(lambda x: calculate_aqi(x, breakpoints_pm25))
    df["AQI_Category"] = df["AQI_PM2.5"].apply(categorize_aqi)

    # Feature engineering
    df["pollutionRatio"] = df["Benzene"] / df["Toluene"]
    df["NO*WS"] = df["NO"] * df["WS"]
    df.dropna(inplace=True)

    st.subheader("🔍 Cleaned Data Preview")
    st.dataframe(df.head())

    # Select features and target
    selected_features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temp', 'Humidity', 'WS', 'Benzene', 'Toluene', 'NO', 'pollutionRatio', 'NO*WS']
    target = 'AQI_PM2.5'
    X = df[selected_features]
    y = df[target]

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 📦 XGBoost Model
    st.subheader("🌲 XGBoost Regression")
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    xgb_rmse = mean_squared_error(y_test, xgb_preds, squared=False)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    st.markdown(f"**RMSE**: {xgb_rmse:.2f}, **MAE**: {xgb_mae:.2f}, **R²**: {xgb_r2:.4f}")

    # 🧠 Neural Net Model
    st.subheader("🧠 Neural Network Regression")

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1
