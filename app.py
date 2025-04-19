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

st.set_page_config(layout="wide")
st.title("Air Quality Index (AQI) Dashboard")
st.markdown("---")

# Horizontal tab navigation
tabs = st.tabs(["Home", "EDA", "Feature Importance", "PCA", "Model Metrics", "Model Comparison"])

# Sidebar for uploading dataset
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

    df_clean = remove_outliers_iqr(df)
    df_clean = df_clean.dropna(subset=["AQI_PM2.5"])
    X = df_clean.select_dtypes(include=np.number).drop(columns=["AQI_PM2.5"])
    y = df_clean["AQI_PM2.5"]

    # HOME
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    # EDA
    with tabs[1]:
        st.subheader("Distribution of Variables")
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

        st.subheader("Boxplots by AQI Category")
        fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6))
        sns.boxplot(x="AQI_Category", y="PM2.5", data=df, ax=axes2[0])
        sns.boxplot(x="AQI_Category", y="Benzene", data=df, ax=axes2[1])
        sns.boxplot(x="AQI_Category", y="Toluene", data=df, ax=axes2[2])
        st.pyplot(fig2)

        st.subheader("Pairplot of Important Features")
        selected_features = ["PM2.5", "SO2", "NO2", "WS", "AQI_PM2.5"]
        pairplot_data = df[selected_features + ["AQI_Category"]]
        pairplot_fig = sns.pairplot(pairplot_data, hue="AQI_Category")
        st.pyplot(pairplot_fig)

    # FEATURE IMPORTANCE
    with tabs[2]:
        st.subheader("Feature Importance using Random Forest")
        model = RandomForestRegressor()
        model.fit(X, y)
        importance = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)
        st.bar_chart(importance_df.set_index("Feature"))

    # PCA
    with tabs[3]:
        st.subheader("PCA - Principal Component Analysis")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["AQI_Category"] = df_clean["AQI_Category"].values
        fig2 = px.scatter(pca_df, x="PC1", y="PC2", color="AQI_Category")
        st.plotly_chart(fig2)

    # MODEL METRICS TAB
    with tabs[4]:
        st.subheader("Model Performance Metrics")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression()
        }

        results = []
        preds = {}
        full_comparison_df = pd.DataFrame()

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds[name] = (y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
            temp_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred,
                "Model": name,
                "AQI_Category": pd.cut(y_test, bins=[0, 50, 100, 150, 200, 300, 500],
                    labels=["Good", "Moderate", "Unhealthy for Sensitive Groups",
                            "Unhealthy", "Very Unhealthy", "Hazardous"])
            })
            full_comparison_df = pd.concat([full_comparison_df, temp_df], ignore_index=True)

        results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
        st.dataframe(results_df)

        st.subheader("Download Cleaned Data")
        csv = df_clean.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download Cleaned Dataset</a>'
        st.markdown(href, unsafe_allow_html=True)

    # MODEL COMPARISON TAB
    with tabs[5]:
        st.subheader("Combined Model Comparison")
        if not full_comparison_df.empty:
            fig_all = px.scatter(full_comparison_df, x="Actual", y="Predicted", color="Model", symbol="AQI_Category",
                                 title="Actual vs Predicted AQI - All Models")
            st.plotly_chart(fig_all)

else:
    st.warning("Please upload a CSV file using the sidebar.")
