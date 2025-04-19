import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px
import base64
import io
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(layout="wide")
st.title("Air Quality Index (AQI) Dashboard")
st.markdown("---")

tabs = st.tabs(["Home", "EDA", "Feature Importance", "PCA", "Model Metrics", "Model Comparison"])
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["pollutionRatio"] = df["Benzene"] / df["Toluene"]
    df["NO*WS"] = df["NO"] * df["WS"]

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

    def remove_outliers_iqr(data):
        numeric_data = data.select_dtypes(include=[np.number])
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        filter_mask = ~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
        return data[filter_mask]

    df_clean = remove_outliers_iqr(df).dropna(subset=["AQI_PM2.5"])
    X = df_clean.select_dtypes(include=np.number).drop(columns=["AQI_PM2.5"])
    y = df_clean["AQI_PM2.5"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # MODEL METRICS TAB
    with tabs[4]:
        st.subheader("Model Performance Metrics")
        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression()
        }
        preds = {}
        for name, model in models.items():
            st.markdown(f"### {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds[name] = (y_test, y_pred)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**R2 Score:** {r2:.2f}")

            # Classification via AQI category
            pred_cat = pd.cut(y_pred, bins=[0, 50, 100, 150, 200, 300, 500],
                              labels=["Good", "Moderate", "Unhealthy for Sensitive Groups",
                                      "Unhealthy", "Very Unhealthy", "Hazardous"])
            true_cat = pd.cut(y_test, bins=[0, 50, 100, 150, 200, 300, 500],
                              labels=["Good", "Moderate", "Unhealthy for Sensitive Groups",
                                      "Unhealthy", "Very Unhealthy", "Hazardous"])

            conf_df = pd.crosstab(true_cat, pred_cat, rownames=["Actual"], colnames=["Predicted"])
            st.write("**Confusion Matrix (categorized AQI):**")
            st.dataframe(conf_df)

        st.subheader("Download Cleaned Data")
        csv = df_clean.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Dataset</a>'
        st.markdown(href, unsafe_allow_html=True)
