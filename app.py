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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import plotly.express as px
from scipy.stats import zscore

st.set_page_config(page_title="AQI ML App", layout="wide")
st.title("Air Quality Index (AQI) ML App with Full EDA, Models, and Residuals")

uploaded_file = st.sidebar.file_uploader("\U0001F4C5 Upload a CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    tabs = st.tabs(["\U0001F4CA Overview", "\U0001F4C8 EDA", "\u2699\ufe0f Preprocessing", "\U0001F50D PCA", "\U0001F916 Models", "\U0001F4C9 Results"])

    with tabs[0]:
        st.header("\U0001F4CA Project Overview")
        st.write("This app calculates AQI from PM2.5, performs EDA, PCA, trains ML models, and visualizes residuals.")
        st.dataframe(df.head())

    with tabs[1]:
        st.header("\U0001F4C8 EDA: Univariate, Bivariate, Multivariate")
        numeric_cols = df.select_dtypes(include=np.number).columns

        st.subheader("1️⃣ Univariate Histograms (Grid View)")
        valid_cols = [col for col in numeric_cols if df[col].notna().sum() > 0]
        n_cols = 4
        n_rows = (len(valid_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for idx, col in enumerate(valid_cols):
            sns.histplot(df[col].dropna(), kde=True, bins=50, ax=axes[idx])
            axes[idx].set_title(f"Distribution of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
        for i in range(len(valid_cols), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("2️⃣ Bivariate: Scatter with PM2.5")
        for col in numeric_cols:
            if col != "PM2.5":
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[col], y=df["PM2.5"], ax=ax)
                ax.set_title(f"PM2.5 vs {col}")
                st.pyplot(fig)

        st.subheader("3️⃣ Multivariate: Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        if len(numeric_cols) <= 6:
            st.subheader("Pairplot")
            fig = sns.pairplot(df[numeric_cols])
            st.pyplot(fig)

    breakpoints_pm25 = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500)]

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

    with tabs[2]:
        st.header("\u2699\ufe0f Preprocessing")
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

        df = df.dropna()
        z = np.abs(zscore(df.select_dtypes(include=np.number)))
        df = df[(z < 3).all(axis=1)]
        st.success(f"Remaining after outlier removal: {len(df)} rows")

    y = df['AQI_Category']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = df.drop(columns=['AQI_PM2.5', 'AQI_Category'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with tabs[3]:
        st.header("\U0001F50D PCA")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['AQI_Category'] = y.values
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='AQI_Category', title="PCA 2D")
        st.plotly_chart(fig)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
    residuals_dict = {}
    predictions_dict = {}

    with tabs[4]:
        st.header("\U0001F916 Models")
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
            predictions_dict[name] = y_pred
            acc = accuracy_score(y_test, y_pred)
            model_scores[name] = acc
            st.subheader(f"{name} Accuracy: {acc:.2f}")
            st.text(classification_report(y_test, y_pred, target_names=le.classes_))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
            disp.plot(ax=ax, cmap='viridis')
            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            plt.title(f"{name} Confusion Matrix")
            st.pyplot(fig)

            if name in ["Decision Tree", "Random Forest", "XGBoost"]:
                residuals_dict[name] = y_test - y_pred

    with tabs[5]:
        st.header("\U0001F4C9 Results Summary")
        st.subheader("Accuracy Comparison")
        st.bar_chart(pd.Series(model_scores))

        st.subheader("Feature Importances (Random Forest)")
        rf = models['Random Forest']
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        st.bar_chart(importance.sort_values(ascending=False))

        st.subheader("Residual Plots")
        for model_name, residuals in residuals_dict.items():
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, bins=20, ax=ax)
            ax.set_title(f"{model_name} Residuals")
            ax.set_xlabel("Residual = True - Predicted")
            st.pyplot(fig)

else:
    st.warning("Please upload a CSV file with PM2.5 and related air quality columns.")
