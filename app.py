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

st.set_page_config(page_title='AQI PM2.5 Analysis', layout='wide')

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Preview", "Preprocessing", "EDA", "Modeling & Results"])

if page == "Upload & Preview":
    st.title("📄 Upload and Preview Dataset")
    data = st.file_uploader("Upload your CSV file", type=['csv'])
    if data is not None:
        df = pd.read_csv(data)
        st.session_state['df'] = df
        st.dataframe(df.head())

elif page == "Preprocessing":
    st.title("🛠️ Preprocessing")
    if 'df' in st.session_state:
        df = st.session_state['df']
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.session_state['df_clean'] = df_clean

        st.write(f"Original shape: {df.shape}, After outlier removal: {df_clean.shape}")

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_clean.dropna())
        df_scaled = pd.DataFrame(df_scaled, columns=df_clean.columns)
        st.session_state['df_scaled'] = df_scaled
        st.dataframe(df_scaled.head())

        st.subheader("📌 Feature Importance")
        X_feat = df_clean.drop(columns=['PM2.5'])
        y_feat = df_clean['PM2.5']
        rf_feat = RandomForestRegressor()
        rf_feat.fit(X_feat, y_feat)
        importances = rf_feat.feature_importances_
        feat_imp = pd.Series(importances, index=X_feat.columns).sort_values(ascending=False)
        fig_fi, ax_fi = plt.subplots()
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_fi)
        ax_fi.set_title("Feature Importance - Random Forest")
        st.pyplot(fig_fi)
    else:
        st.warning("Please upload and preview the dataset first.")

elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if 'df_clean' in st.session_state:
        df_clean = st.session_state['df_clean']

        st.subheader("Descriptive Statistics")
        st.write(df_clean.describe())

        st.subheader("Missing Values")
        st.write(df_clean.isnull().sum())

        st.subheader("📊 Bivariate Analysis")
        selected_features = st.multiselect("Select features to plot vs PM2.5", df_clean.columns.tolist())
        for feature in selected_features:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df_clean[feature], y=df_clean['PM2.5'], ax=ax)
            ax.set_title(f'{feature} vs PM2.5')
            st.pyplot(fig)

        st.subheader("🧩 Multivariate Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("🧬 Principal Component Analysis (PCA)")
        df_scaled = st.session_state['df_scaled']
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)
        st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
        fig_pca, ax_pca = plt.subplots()
        ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
        ax_pca.set_xlabel('PCA 1')
        ax_pca.set_ylabel('PCA 2')
        ax_pca.set_title('PCA - 2D Projection')
        st.pyplot(fig_pca)
    else:
        st.warning("Please go to Preprocessing to prepare data.")

elif page == "Modeling & Results":
    st.title("🤖 Modeling and Results")
    if 'df_clean' in st.session_state:
        df_clean = st.session_state['df_clean']
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

            st.subheader(f"📈 {name}")
            st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

            fig_res, ax_res = plt.subplots()
            sns.residplot(x=y_test, y=y_pred, lowess=True, ax=ax_res, line_kws={'color': 'red'})
            ax_res.set_title(f"Residual Plot - {name}")
            ax_res.set_xlabel("Actual PM2.5")
            ax_res.set_ylabel("Residuals")
            st.pyplot(fig_res)
    else:
        st.warning("Please go to Preprocessing to prepare data.")
