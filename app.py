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

    # HOME TAB
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    # EDA TAB
    with tabs[1]:
        st.subheader("Histogram Grid")
        valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
        fig, axes = plt.subplots(5, 5, figsize=(30, 20))
        axes = axes.flatten()
        for idx, col in enumerate(valid_cols):
            sns.histplot(X[col], kde=True, ax=axes[idx])
            axes[idx].set_title(col)
        for idx in range(len(valid_cols), 25):
            fig.delaxes(axes[idx])
        st.pyplot(fig)

        st.subheader("Bivariate Scatter Plots")
        fig_bi, axes_bi = plt.subplots(2, 5, figsize=(30, 10))
        for idx, col in enumerate(valid_cols[:10]):
            sns.scatterplot(x=col, y="AQI_PM2.5", data=df_clean, ax=axes_bi.flatten()[idx])
            axes_bi.flatten()[idx].set_title(f"{col} vs AQI_PM2.5")
        st.pyplot(fig_bi)

        st.subheader("Boxplots by AQI Category")
        fig_box, axes_box = plt.subplots(1, 3, figsize=(24, 6))
        sns.boxplot(x="AQI_Category", y="PM2.5", data=df, ax=axes_box[0])
        sns.boxplot(x="AQI_Category", y="Benzene", data=df, ax=axes_box[1])
        sns.boxplot(x="AQI_Category", y="Toluene", data=df, ax=axes_box[2])
        st.pyplot(fig_box)

        st.subheader("Pairplot of Key Features")
        pairplot_fig = sns.pairplot(df[["PM2.5", "SO2", "NO2", "WS", "AQI_PM2.5", "AQI_Category"]], hue="AQI_Category")
        st.pyplot(pairplot_fig)

        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_clean.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("Download All EDA Plots as PDF")
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig)
            pdf.savefig(fig_bi)
            pdf.savefig(fig_box)
            pdf.savefig(pairplot_fig.fig)
            pdf.savefig(fig_corr)
        st.download_button("Download EDA Plots PDF", data=pdf_buffer.getvalue(), file_name="eda_plots.pdf", mime="application/pdf")

    # FEATURE IMPORTANCE TAB
    with tabs[2]:
        st.subheader("Feature Importance using Random Forest")
        model = RandomForestRegressor()
        model.fit(X, y)
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)
        st.bar_chart(importance_df.set_index("Feature"))

    # PCA TAB
    with tabs[3]:
        st.subheader("Principal Component Analysis (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["AQI_Category"] = df_clean["AQI_Category"].values
        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="AQI_Category", title="PCA: AQI Categories in 2D Space")
        st.plotly_chart(fig_pca)

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
        results = []
        preds = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds[name] = (y_test, y_pred)
            results.append({
                "Model": name,
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            })
        results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
        st.dataframe(results_df)

        st.subheader("Download Cleaned Data")
        csv = df_clean.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Dataset</a>'
        st.markdown(href, unsafe_allow_html=True)

    # MODEL COMPARISON TAB
    with tabs[5]:
        st.subheader("Residual Histograms (All Models)")
        fig_resid_all, axes_resid_all = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            sns.histplot(residuals, bins=30, kde=True, ax=axes_resid_all[i])
            axes_resid_all[i].set_title(f"{name} Residuals")
        st.pyplot(fig_resid_all)
