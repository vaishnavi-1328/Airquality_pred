import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title='AQI PM2.5 Analysis App', layout='wide')
st.title("🌫️ Air Quality Index (PM2.5) Analysis")

# Upload CSV
data = st.file_uploader("Upload your CSV file", type=['csv'])

if data is not None:
    df = pd.read_csv(data)
    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    # EDA
    st.subheader("🔍 Exploratory Data Analysis")
    st.write(df.describe())
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # Outlier Detection (IQR Method)
    st.subheader("🚨 Outlier Detection and Removal")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    st.write(f"Original shape: {df.shape}, After outlier removal: {df_clean.shape}")

    # Normalization
    st.subheader("⚖️ Feature Normalization")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean.dropna())
    df_scaled = pd.DataFrame(df_scaled, columns=df_clean.columns)
    st.write(df_scaled.head())

    # PCA
    st.subheader("🧬 Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
    fig_pca, ax_pca = plt.subplots()
    ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    ax_pca.set_xlabel('PCA 1')
    ax_pca.set_ylabel('PCA 2')
    ax_pca.set_title('PCA - 2D Projection')
    st.pyplot(fig_pca)

    # Bivariate Analysis
    st.subheader("📊 Bivariate Analysis")
    selected_features = st.multiselect("Select features to plot vs PM2.5", df.columns.tolist())
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_clean[feature], y=df_clean['PM2.5'], ax=ax)
        ax.set_title(f'{feature} vs PM2.5')
        st.pyplot(fig)

    # Multivariate Heatmap
    st.subheader("🧩 Multivariate Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # Regression Models
    st.subheader("🤖 Model Training and Comparison")
    X = df_clean.drop(columns=['PM2.5'])
    y = df_clean['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'XGBoost': XGBRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**{name}**")
        st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

        fig_res, ax_res = plt.subplots()
        sns.residplot(x=y_test, y=y_pred, lowess=True, ax=ax_res, line_kws={'color': 'red'})
        ax_res.set_title(f"Residual Plot - {name}")
        ax_res.set_xlabel("Actual PM2.5")
        ax_res.set_ylabel("Residuals")
        st.pyplot(fig_res)

else:
    st.info("Please upload a CSV file to begin analysis.")
