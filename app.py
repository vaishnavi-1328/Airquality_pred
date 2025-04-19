import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px
import base64
from matplotlib.backends.backend_pdf import PdfPages
import io

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

    # HOME
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    # EDA
    with tabs[1]:
        st.subheader("EDA - Histogram Grid")
        valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
        n_cols = 5
        n_rows = int(np.ceil(len(valid_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for idx, col in enumerate(valid_cols):
            sns.histplot(X[col].dropna(), kde=True, bins=50, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}')
        for i in range(len(valid_cols), len(axes)):
            fig.delaxes(axes[i])
        st.pyplot(fig)

        st.subheader("Bivariate Scatter Plots")
        fig_bi, axes_bi = plt.subplots(2, 5, figsize=(30, 10))
        for idx, col in enumerate(valid_cols[:10]):
            sns.scatterplot(x=col, y="AQI_PM2.5", data=df_clean, ax=axes_bi.flatten()[idx])
            axes_bi.flatten()[idx].set_title(f'{col} vs AQI_PM2.5')
        st.pyplot(fig_bi)

        st.subheader("Boxplots by AQI Category")
        fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6))
        sns.boxplot(x="AQI_Category", y="PM2.5", data=df, ax=axes2[0])
        sns.boxplot(x="AQI_Category", y="Benzene", data=df, ax=axes2[1])
        sns.boxplot(x="AQI_Category", y="Toluene", data=df, ax=axes2[2])
        st.pyplot(fig2)

        st.subheader("Pairplot of Important Features")
        pairplot_data = df[["PM2.5", "SO2", "NO2", "WS", "AQI_PM2.5", "AQI_Category"]]
        pairplot_fig = sns.pairplot(pairplot_data, hue="AQI_Category")
        st.pyplot(pairplot_fig)

        st.subheader("Correlation Heatmap")
        numeric_cols = df_clean.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("Download All EDA Plots as PDF")
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig)
            pdf.savefig(fig_bi)
            pdf.savefig(fig2)
            pdf.savefig(pairplot_fig.fig)
            pdf.savefig(fig_corr)
        st.download_button("Download EDA Plots PDF", data=pdf_buffer.getvalue(), file_name="eda_plots.pdf", mime="application/pdf")

    # MODEL TABS...
    # Your Feature Importance, PCA, Model Metrics, and Model Comparison tabs follow this

    # Add residual histograms in last tab
    with tabs[5]:
        st.subheader("Residual Histograms (All Models)")
        all_models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression()
        }
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        fig_resid_all, axes_resid_all = plt.subplots(1, len(all_models), figsize=(6 * len(all_models), 5))
        if len(all_models) == 1:
            axes_resid_all = [axes_resid_all]
        for i, (name, model) in enumerate(all_models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            sns.histplot(residuals, bins=30, kde=True, ax=axes_resid_all[i])
            axes_resid_all[i].set_title(f'{name} Residuals')
        plt.tight_layout()
        st.pyplot(fig_resid_all)
