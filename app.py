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
        st.write("This web application calculates AQI from PM2.5, performs feature engineering, visualizes PCA, and trains multiple ML models.")
        st.subheader("Raw Dataset Preview")
        st.dataframe(df.head())

    breakpoints_pm25 = [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500)
    ]

    def calculate_aqi(pm25):
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints_pm25:
            if bp_lo <= pm25 <= bp_hi:
