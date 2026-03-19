
import streamlit as st

# Page config
st.set_page_config(
    page_title="Air Quality Predictor | Neural Networks",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #1a2a3a 100%);
}
[data-testid="stSidebar"] * {
    color: #e0eaf5 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
}

/* Main background */
[data-testid="stAppViewContainer"] {
    background: #f5f8fc;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2eaf5;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(30,60,120,0.06);
}

/* Headers */
h1, h2, h3 { 
    font-family: 'Space Grotesk', sans-serif; 
    font-weight: 700; 
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}

/* Code font */
code { 
    font-family: 'JetBrains Mono', monospace; 
}

/* Info boxes */
.aqi-box {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🌫️ Air Quality Prediction with Neural Network")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Data Explorer", "🧠 Model Training", "🔮 Live Predictor", "📈 Model Evaluation"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**About this Project**")
    st.markdown(
        "A neural network trained on environmental sensor data to predict AQI levels in real-time.",
        help="Built with TensorFlow/Keras + Streamlit"
    )

# Route to pages (FIXED)
if page == "🏠 Home":
    from pages.home import show as home_page
    home_page()

elif page == "📊 Data Explorer":
    from pages.data_explorer import show as data_page
    data_page()

elif page == "🧠 Model Training":
    from pages.model_training import show as training_page
    training_page()

elif page == "🔮 Live Predictor":
    from pages.predictor import show as predictor_page
    predictor_page()

elif page == "📈 Model Evaluation":
    from pages.evaluation import show as eval_page
    eval_page()
