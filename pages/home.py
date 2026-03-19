import streamlit as st

def show():
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f1923 0%, #1e3a8a 60%, #2563eb 100%);
                padding: 3rem 2.5rem; border-radius: 20px; margin-bottom: 2rem; color: white;">
        <div style="font-size: 13px; font-weight: 600; letter-spacing: 3px; color: #93c5fd; margin-bottom: 0.5rem;">
            DEEP LEARNING · ENVIRONMENTAL AI
        </div>
        <h1 style="font-size: 2.8rem; font-weight: 800; margin: 0 0 1rem; color: white; line-height: 1.2;">
            Predicting Air Quality<br>with Neural Networks
        </h1>
        <p style="font-size: 1.1rem; color: #bfdbfe; max-width: 600px; line-height: 1.7; margin-bottom: 1.5rem;">
            A deep learning approach to predict the Air Quality Index (AQI) from
            environmental sensor readings — PM2.5, CO₂, NO₂, temperature, humidity & more.
        </p>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.15); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 500;">🧠 Neural Network (MLP)</span>
            <span style="background: rgba(255,255,255,0.15); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 500;">📡 Real-time Prediction</span>
            <span style="background: rgba(255,255,255,0.15); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 500;">📊 Interactive EDA</span>
            <span style="background: rgba(255,255,255,0.15); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 500;">⚡ TensorFlow / Keras</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AQI Scale
    st.markdown("### 🎨 AQI Index Scale")
    aqi_levels = [
        ("0–50",   "Good",                "#22c55e", "Air quality is satisfactory. No risk."),
        ("51–100", "Moderate",            "#eab308", "Acceptable quality. Minor concern for sensitive people."),
        ("101–150","Unhealthy (Sensitive)","#f97316", "Sensitive groups may be affected."),
        ("151–200","Unhealthy",           "#ef4444", "Everyone may begin to feel effects."),
        ("201–300","Very Unhealthy",      "#a855f7", "Health alert. Everyone may be seriously affected."),
        ("301+",   "Hazardous",           "#7f1d1d", "Emergency conditions. Entire population impacted."),
    ]
    cols = st.columns(6)
    for col, (aqi, label, color, tip) in zip(cols, aqi_levels):
        with col:
            st.markdown(f"""
            <div style="background:{color}18; border-left: 4px solid {color};
                        border-radius: 8px; padding: 0.8rem; text-align: center;">
                <div style="font-size: 1rem; font-weight: 700; color: {color};">{aqi}</div>
                <div style="font-size: 12px; font-weight: 600; color: #374151; margin-top: 2px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(tip)

    st.markdown("---")

    # Project overview cards
    st.markdown("### 🗺️ Project Workflow")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📊", "Data Explorer", "Load and visualize air quality dataset. Understand distributions, correlations, and seasonal patterns."),
        ("🧠", "Model Training", "Build and train a multi-layer neural network with configurable architecture and hyperparameters."),
        ("🔮", "Live Predictor", "Input pollutant values and instantly predict AQI with a confidence score."),
        ("📈", "Evaluation", "Visualize loss curves, RMSE, MAE, R² score and prediction vs actual plots."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div style="background:white; border: 1px solid #e2eaf5; border-radius: 14px;
                        padding: 1.2rem; height: 180px; box-shadow: 0 2px 8px rgba(30,60,120,0.06);">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: 700; font-size: 15px; margin: 0.4rem 0 0.5rem; color: #1e3a8a;">{title}</div>
                <div style="font-size: 13px; color: #6b7280; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    cols = st.columns(5)
    techs = [
        ("", "Python 3.10"),
        ("", "TensorFlow 2.x"),
        ("", "Pandas / NumPy"),
        ("", "Plotly / Seaborn"),
        ("", "Streamlit"),
    ]
    for col, (icon, tech) in zip(cols, techs):
        with col:
            st.markdown(f"""
            <div style="text-align:center; background:white; border:1px solid #e2eaf5;
                        border-radius:10px; padding:0.8rem;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-size:13px; font-weight:600; color:#374151; margin-top:4px;">{tech}</div>
            </div>
            """, unsafe_allow_html=True)



