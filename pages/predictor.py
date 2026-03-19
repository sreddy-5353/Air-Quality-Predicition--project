import streamlit as st
import numpy as np
import os
import pickle

def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "#22c55e", "Air quality is satisfactory. No health risk.", "😊"
    elif aqi <= 100:
        return "Moderate", "#eab308", "Acceptable air quality. Minor risk for sensitive groups.", "🙂"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#f97316", "Sensitive individuals should limit outdoor exposure.", "😐"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "Everyone may experience health effects.", "😷"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7", "Health alert — everyone may be seriously affected.", "🤢"
    else:
        return "Hazardous", "#7f1d1d", "Emergency conditions. Avoid all outdoor activity.", "☠️"

def show():
    st.markdown("## 🔮 Live AQI Predictor")
    st.markdown("Enter pollutant readings to predict the Air Quality Index instantly.")

    # Check if model exists
    model_path = "saved_model/model.pkl"
    if not os.path.exists(model_path):
        st.info("🧠 No trained model found. Using a **demo estimator** for preview.")
        use_demo = True
    else:
        use_demo = False

    st.markdown("### 📡 Enter Sensor Readings")
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=35.0, step=0.5)
        pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=600.0, value=55.0, step=0.5)
        co = st.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=1.2, step=0.1)
    with col2:
        no2 = st.number_input("NO₂ (ppb)", min_value=0.0, max_value=300.0, value=25.0, step=0.5)
        so2 = st.number_input("SO₂ (ppb)", min_value=0.0, max_value=200.0, value=10.0, step=0.5)
        o3 = st.number_input("O₃ (ppb)", min_value=0.0, max_value=300.0, value=30.0, step=0.5)
    with col3:
        temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=28.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)

    st.markdown("---")

    if st.button("⚡ Predict AQI", use_container_width=True):
        input_vals = np.array([[pm25, pm10, co, no2, so2, o3, temperature, humidity, wind_speed]])

        if use_demo:
            # Simple heuristic demo model
            aqi = (pm25 * 1.5 + pm10 * 0.6 + no2 * 0.4 + so2 * 0.3 +
                   co * 8 + o3 * 0.5 - wind_speed * 0.5 + humidity * 0.1)
            aqi = float(np.clip(aqi, 0, 500))
        else:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            scaler = saved["scaler"]
            features = saved["features"]
            feat_map = {"PM2.5": pm25, "PM10": pm10, "CO": co, "NO2": no2,
                        "SO2": so2, "O3": o3, "Temperature": temperature,
                        "Humidity": humidity, "Wind_Speed": wind_speed}
            inp = np.array([[feat_map.get(f, 0) for f in features]])
            inp_sc = scaler.transform(inp)
            aqi = float(model.predict(inp_sc)[0][0])

        label, color, message, emoji = get_aqi_info(aqi)

        # Result card
        st.markdown(f"""
        <div style="background: {color}18; border: 2px solid {color};
                    border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;">
            <div style="font-size: 4rem; font-weight: 800; color: {color};">{aqi:.1f}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin-top: 0.3rem;">
                {emoji} {label}
            </div>
            <div style="font-size: 15px; color: #374151; margin-top: 0.8rem;">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge-style bar
        st.markdown("**AQI Scale Position**")
        pct = min(aqi / 500, 1.0)
        st.progress(pct)
        scale_cols = st.columns(6)
        for col, (band, bc) in zip(scale_cols, [("Good\n0-50","#22c55e"),("Moderate\n51-100","#eab308"),
            ("Sensitive\n101-150","#f97316"),("Unhealthy\n151-200","#ef4444"),
            ("Very\n201-300","#a855f7"),("Hazardous\n300+","#7f1d1d")]):
            with col:
                st.markdown(f"<div style='background:{bc}30; border-radius:6px; padding:6px; text-align:center; font-size:11px; font-weight:600; color:{bc};'>{band}</div>", unsafe_allow_html=True)

        # Health tips
        st.markdown("### 💡 Health Recommendations")
        if aqi <= 100:
            tips = ["✅ Safe to go outside", "✅ Normal physical activity is fine", "✅ Windows can be kept open"]
        elif aqi <= 200:
            tips = ["⚠️ Sensitive groups should limit outdoor time", "⚠️ Consider wearing a mask outdoors", "⚠️ Monitor air quality updates"]
        else:
            tips = ["🚨 Avoid all outdoor physical activity", "🚨 Keep windows and doors closed", "🚨 Use air purifiers indoors", "🚨 Seek medical advice if symptoms occur"]
        for tip in tips:
            st.markdown(tip)
