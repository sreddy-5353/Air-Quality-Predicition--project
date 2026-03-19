import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load air quality dataset. Falls back to synthetic data if no file found."""
    for path in ["data/air_quality.csv", "air_quality.csv", "data.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df

    # Generate realistic synthetic dataset
    np.random.seed(42)
    n = 2000
    pm25 = np.random.gamma(2, 18, n)
    pm10 = pm25 * np.random.uniform(1.2, 2.0, n) + np.random.normal(0, 5, n)
    co   = np.random.exponential(1.2, n)
    no2  = np.random.gamma(1.5, 18, n)
    so2  = np.random.gamma(1.2, 8, n)
    o3   = np.random.gamma(2, 15, n)
    temp = np.random.normal(28, 7, n)
    hum  = np.random.beta(5, 3, n) * 100
    wind = np.random.exponential(10, n)

    # AQI formula (approximate)
    aqi = (pm25 * 1.5 + pm10 * 0.6 + no2 * 0.4 + so2 * 0.3 +
           co * 8 + o3 * 0.5 - wind * 0.5 + hum * 0.1 +
           np.random.normal(0, 5, n))
    aqi = np.clip(aqi, 0, 500)

    df = pd.DataFrame({
        "PM2.5": pm25.round(2),
        "PM10": pm10.round(2),
        "CO": co.round(3),
        "NO2": no2.round(2),
        "SO2": so2.round(2),
        "O3": o3.round(2),
        "Temperature": temp.round(1),
        "Humidity": hum.round(1),
        "Wind_Speed": wind.round(1),
        "AQI": aqi.round(1)
    })
    return df
