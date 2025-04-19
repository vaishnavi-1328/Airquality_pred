import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px

# Title
st.title("Air Quality Index (AQI) Analysis App")

# Load dataset
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # Feature Engineering
    st.subheader("Feature Engineering")
    df["pollutionRatio"] = df["Benzene"] / df["Toluene"]
    df["NO*WS"] = df["NO"] * df["WS"]

    # AQI Calculation Logic
    breakpoints_pm25 = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.4, 301, 500)
    ]

    def calculate_aqi(concentration, breakpoints):
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return round(((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo)
        return None

    def categorize_aqi(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    df["AQI_PM2.5"] = df["PM2.5"].apply(lambda x: calculate_aqi(x, breakpoints_pm25))
    df["AQI_Category"] = df["AQI_PM2.5"].apply(categorize_aqi)

    st.write("AQI Calculation Completed")
    st.dataframe(df[["PM2.5", "AQI_PM2.5", "AQI_Category"]].head())

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis (EDA)")
    selected_features = st.multiselect("Select features for bivariate plots:", df.columns.tolist())
    if len(selected_features) >= 2:
        fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color="AQI_Category")
        st.plotly_chart(fig)

    # Feature Importance using Random Forest
    st.subheader("Feature Importance (Random Forest)")
    df_clean = df.dropna()
    X = df_clean.select_dtypes(include=np.number).drop(columns=["AQI_PM2.5"])
    y = df_clean["AQI_PM2.5"]
    model = RandomForestRegressor()
    model.fit(X, y)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df)
    st.bar_chart(importance_df.set_index("Feature"))

    # PCA
    st.subheader("Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["AQI_Category"] = df_clean["AQI_Category"].values
    fig2 = px.scatter(pca_df, x="PC1", y="PC2", color="AQI_Category")
    st.plotly_chart(fig2)

    # Model Comparison
    st.subheader("Model Comparison")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"### {name} Results")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"R²: {r2_score(y_test, y_pred):.2f}")
        fig_res = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual AQI", "y": "Predicted AQI"}, title=f"{name} Residual Plot")
        st.plotly_chart(fig_res)
