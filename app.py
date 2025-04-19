import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import plotly.express as px

st.set_page_config(page_title="AQI Modeling App", layout="wide")
st.title("Air Quality Index (AQI) Web App with Modeling")
st.markdown("Upload your air quality dataset, calculate AQI, and run machine learning models to predict AQI categories.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset Preview")
    st.write(df.head())

    # AQI Breakpoints for PM2.5
    breakpoints_pm25 = [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500)
    ]

    def calculate_aqi(pm25):
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints_pm25:
            if bp_lo <= pm25 <= bp_hi:
                return round(((i_hi - i_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + i_lo)
        return None

    def categorize_aqi(aqi):
        if aqi is None:
            return "Unknown"
        elif aqi <= 50:
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

    df['AQI_PM2.5'] = df['PM2.5'].apply(calculate_aqi)
    df['AQI_Category'] = df['AQI_PM2.5'].apply(categorize_aqi)

    st.subheader("Feature Engineering")
    df['pollutionRatio'] = df['Benzene'] / (df['Toluene'] + 1e-5)
    df['NO*WS'] = df['NO'] * df['WS']
    st.write(df[['pollutionRatio', 'NO*WS']].head())

    df_clean = df.dropna()
    X = df_clean.drop(columns=['AQI_PM2.5', 'AQI_Category'])
    y = df_clean['AQI_Category']

    st.subheader("Outlier Summary")
    st.write(X.describe())

    st.subheader("Normalization & PCA")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['AQI_Category'] = y.values

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='AQI_Category', title="PCA Visualization")
    st.plotly_chart(fig)

    st.subheader("Train-Test Split and Model Training")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    tabs = st.tabs(["Naive Bayes", "Decision Tree", "Random Forest", "XGBoost"])

    with tabs[0]:
        st.markdown("### Naive Bayes")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred_nb))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_nb))

    with tabs[1]:
        st.markdown("### Decision Tree")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_dt))

    with tabs[2]:
        st.markdown("### Random Forest")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_rf))

        st.markdown("#### Feature Importance")
        importance = pd.Series(rf.feature_importances_, index=df_clean.drop(columns=['AQI_PM2.5','AQI_Category']).columns)
        st.bar_chart(importance.sort_values(ascending=False))

    with tabs[3]:
        st.markdown("### XGBoost")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred_xgb))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_xgb))
else:
    st.info("Please upload a CSV file with air quality features including 'PM2.5', 'Benzene', 'Toluene', 'NO', and 'WS'.")
