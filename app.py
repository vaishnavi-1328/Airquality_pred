import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")
st.title("🌫️ Air Quality Index (AQI) Streamlit Dashboard")

# File upload
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("🔎 Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    object_cols = df.select_dtypes(include='object').columns.tolist()

    # Outlier Removal
    st.subheader("🚫 Outlier Removal")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    st.write("✅ Outliers removed using IQR method.")

    # Univariate Analysis
    st.subheader("📊 Univariate Analysis")
    selected_col = st.selectbox("Select a variable", numeric_cols)
    fig_uni, ax_uni = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax_uni)
    st.pyplot(fig_uni)

    # Pairplot (Bivariate)
    st.subheader("📈 Bivariate Analysis (Pairplot)")
    pair_cols = st.multiselect("Choose variables for pairplot", numeric_cols, default=numeric_cols[:5])
    if len(pair_cols) >= 2:
        pairplot_fig = sns.pairplot(df[pair_cols])
        st.pyplot(pairplot_fig)
    else:
        st.warning("Please select at least two variables.")

    # Boxplots by Category
    st.subheader("📦 Multivariate Analysis: Boxplots by AQI Category")
    if object_cols:
        cat_col = st.selectbox("Select AQI Category column", object_cols)
        box_cols = st.multiselect("Select numeric features for boxplots", numeric_cols, default=numeric_cols[:3])
        for bcol in box_cols:
            fig_box, ax_box = plt.subplots()
            sns.boxplot(data=df, x=cat_col, y=bcol, ax=ax_box)
            ax_box.set_title(f"{bcol} by {cat_col}")
            ax_box.tick_params(axis='x', rotation=45)
            st.pyplot(fig_box)
    else:
        st.warning("No categorical column found for AQI categories.")

    # PCA
    st.subheader("🌐 PCA - Principal Component Analysis")
    if len(numeric_cols) >= 2:
        scaled = StandardScaler().fit_transform(df[numeric_cols])
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", title="PCA - 2D Projection")
        st.plotly_chart(fig_pca)
    else:
        st.warning("Need at least 2 numeric columns for PCA.")

    # Modeling
    st.subheader("🧠 Model Training and Evaluation")
    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Feature Variables", [col for col in df.columns if col != target], default=numeric_cols)

    if len(features) > 0:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }

        for name, model in models.items():
            st.subheader(f"🔍 {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))

            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Residual Plot (for numeric targets)
            if np.issubdtype(y.dtype, np.number):
                residuals = y_test - y_pred
                fig_resid, ax_resid = plt.subplots()
                ax_resid.scatter(y_pred, residuals)
                ax_resid.axhline(0, color='red', linestyle='--')
                ax_resid.set_title(f"{name} - Residual Plot")
                ax_resid.set_xlabel("Predicted")
                ax_resid.set_ylabel("Residuals")
                st.pyplot(fig_resid)

else:
    st.warning("📁 Please upload a CSV file to begin.")
