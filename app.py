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
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import plotly.express as px

st.set_page_config(page_title="AQI ML Web App", layout="wide")
st.title("Air Quality Index (AQI) - Full Machine Learning Pipeline")

uploaded_file = st.sidebar.file_uploader("📥 Upload a CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    tabs = st.tabs(["📊 Overview", "⚙️ Preprocessing", "🔍 PCA", "🤖 Models", "📈 Results"])

    with tabs[0]:
        st.header("📊 Project Overview")
        st.write("This app calculates AQI from PM2.5, performs feature engineering, visualizes PCA, and trains ML models.")
        st.subheader("Raw Dataset Preview")
        st.dataframe(df.head())

    # AQI Calculation
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

    with tabs[1]:
        st.header("⚙️ Feature Engineering and Normalization")

        # Only compute if columns exist
        if 'Benzene' in df.columns and 'Toluene' in df.columns:
            df['pollutionRatio'] = df['Benzene'] / (df['Toluene'] + 1e-5)
        if 'NO' in df.columns and 'WS' in df.columns:
            df['NO*WS'] = df['NO'] * df['WS']
        if 'SO2' in df.columns and 'TEMP' in df.columns:
            df['SO2*TEMP'] = df['SO2'] * df['TEMP']
        if 'O3' in df.columns and 'RH' in df.columns:
            df['O3*RH'] = df['O3'] * df['RH']
        if 'NO2' in df.columns and 'CO' in df.columns:
            df['NO2/CO'] = df['NO2'] / (df['CO'] + 1e-5)
        if 'NOx' in df.columns and 'NO' in df.columns:
            df['NOx/NO'] = df['NOx'] / (df['NO'] + 1e-5)

        df_clean = df.dropna()
        st.success("Feature engineering completed on available columns.")
        st.dataframe(df_clean.describe())

    # Define target and features
    y = df_clean['AQI_Category']
    X = df_clean.drop(columns=['AQI_PM2.5', 'AQI_Category'])

    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with tabs[2]:
        st.header("🔍 PCA Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['AQI_Category'] = y.values
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='AQI_Category', title="PCA - 2D Projection")
        st.plotly_chart(fig)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    with tabs[3]:
        st.header("🤖 Model Training")
        models = {
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }

        model_scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            model_scores[name] = acc
            st.subheader(f"🔹 {name} - Accuracy: {acc:.2f}")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(ax=ax)
            st.pyplot(fig)

    with tabs[4]:
        st.header("📈 Results Summary")
        st.subheader("📊 Model Accuracy Comparison")
        st.bar_chart(pd.Series(model_scores).sort_values(ascending=False))
        st.subheader("🌟 Feature Importances (Random Forest)")
        rf = models['Random Forest']
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        st.bar_chart(importance.sort_values(ascending=False))

else:
    st.warning("Please upload a CSV file with PM2.5 and related air quality columns to begin.")
