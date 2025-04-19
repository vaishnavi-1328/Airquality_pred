import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Air Quality Index (AQI) Web App")
st.markdown("Upload a CSV file containing PM2.5 data. This app will calculate and categorize the AQI.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

breakpoints_pm25 = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 500.4, 301, 500)
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

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'PM2.5' in df.columns:
        df["AQI"] = df["PM2.5"].apply(calculate_aqi)
        df["Category"] = df["AQI"].apply(categorize_aqi)

        st.subheader("Preview of Data")
        st.write(df.head())

        st.subheader("AQI Categories")
        st.bar_chart(df["Category"].value_counts())

        st.subheader("AQI Histogram")
        fig, ax = plt.subplots()
        ax.hist(df["AQI"].dropna(), bins=30, color="orange", edgecolor="black")
        ax.set_xlabel("AQI")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Statistics")
        st.write(df.describe())
    else:
        st.error("Your CSV must have a column named 'PM2.5'")
else:
    st.info("Please upload your CSV file to begin.")
