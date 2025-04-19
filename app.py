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

# Sidebar configuration
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Feature Importance", "PCA", "Model Comparison"])

# Load dataset
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

    df_clean = df.dropna()
    X = df_clean.select_dtypes(include=np.number).drop(columns=["AQI_PM2.5"])
    y = df_clean["AQI_PM2.5"]

    # Home Page
    if page == "Home":
        st.title("Air Quality Index (AQI) Dashboard")
        st.write("Welcome to the AQI Analysis Tool. Upload your data using the sidebar.")
        st.subheader("Preview of Data")
        st.dataframe(df.head())

    # EDA Page
    elif page == "EDA":
        st.title("Exploratory Data Analysis")

        st.subheader("Distribution Plots")
        valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
        n_cols = 5
        n_rows = int(np.ceil(len(valid_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(valid_cols):
            sns.histplot(X[col].dropna(), kde=True, bins=50, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

        for i in range(len(valid_cols), len(axes)):
            fig.delaxes(axes[i])

        st.pyplot(fig)

        st.subheader("Boxplots by AQI Category")
        fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6))
        sns.boxplot(x="AQI_Category", y="PM2.5", data=df, ax=axes2[0])
        axes2[0].set_title("PM2.5 by AQI Category")
        sns.boxplot(x="AQI_Category", y="Benzene", data=df, ax=axes2[1])
        axes2[1].set_title("Benzene by AQI Category")
        sns.boxplot(x="AQI_Category", y="Toluene", data=df, ax=axes2[2])
        axes2[2].set_title("Toluene by AQI Category")
        st.pyplot(fig2)

        st.subheader("Pairplot")
        selected_features = ["PM2.5", "SO2", "NO2", "WS", "AQI_PM2.5"]
        pairplot_fig = sns.pairplot(df[selected_features], hue="AQI_Category")
        st.pyplot(pairplot_fig)

    # Feature Importance Page
    elif page == "Feature Importance":
        st.title("Feature Importance (Random Forest)")
        model = RandomForestRegressor()
        model.fit(X, y)
        importance = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)
        st.bar_chart(importance_df.set_index("Feature"))

    # PCA Page
    elif page == "PCA":
        st.title("Principal Component Analysis")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["AQI_Category"] = df_clean["AQI_Category"].values
        fig2 = px.scatter(pca_df, x="PC1", y="PC2", color="AQI_Category")
        st.plotly_chart(fig2)

    # Model Comparison Page
    elif page == "Model Comparison":
        st.title("Model Comparison")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression()
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

else:
    st.warning("Please upload a CSV file using the sidebar.")
