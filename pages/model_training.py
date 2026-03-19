import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import load_data
from utils.model import build_model, train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def show():
    st.markdown("## 🧠 Model Training")
    st.markdown("Configure and train your neural network on the air quality dataset.")

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ["AQI", "AQI_Category", "Date", "date"]]

    st.markdown("### ⚙️ Hyperparameter Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Architecture**")
        hidden_layers = st.slider("Hidden Layers", 1, 5, 3)
        neurons = st.select_slider("Neurons per Layer", options=[32, 64, 128, 256, 512], value=128)
        activation = st.selectbox("Activation Function", ["relu", "tanh", "selu", "elu"])
    with col2:
        st.markdown("**Training**")
        epochs = st.slider("Epochs", 10, 200, 50, step=10)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
    with col3:
        st.markdown("**Data Split**")
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
        optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"])

    # Feature selection
    st.markdown("### 🎯 Feature Selection")
    selected_features = st.multiselect("Select input features", feature_cols, default=feature_cols)

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        return

    # Model summary
    with st.expander("📐 Model Architecture Preview", expanded=False):
        arch_lines = ["**Input Layer** → " + str(len(selected_features)) + " features"]
        for i in range(hidden_layers):
            arch_lines.append(f"**Dense Layer {i+1}** → {neurons} neurons, {activation} + Dropout({dropout})")
        arch_lines.append("**Output Layer** → 1 neuron (AQI prediction)")
        for line in arch_lines:
            st.markdown(f"- {line}")

    st.markdown("---")

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Preparing data..."):
            X = df[selected_features].values
            y = df["AQI"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

        progress = st.progress(0)
        status = st.empty()
        chart_placeholder = st.empty()

        status.info("Building model architecture...")
        model = build_model(len(selected_features), hidden_layers, neurons, activation, dropout, learning_rate, optimizer)

        history_data = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        status.info("Training in progress...")
        batch_epochs = min(10, epochs)
        steps = epochs // batch_epochs

        for step in range(steps):
            hist = train_model(model, X_train_sc, y_train, X_test_sc, y_test, batch_epochs, batch_size)
            for k in history_data:
                history_data[k].extend(hist.history.get(k, []))

            progress.progress((step + 1) / steps)

            # Live loss chart
            fig = go.Figure()
            ep = list(range(1, len(history_data["loss"]) + 1))
            fig.add_trace(go.Scatter(x=ep, y=history_data["loss"], name="Train Loss",
                                     line=dict(color="#2563eb", width=2)))
            fig.add_trace(go.Scatter(x=ep, y=history_data["val_loss"], name="Val Loss",
                                     line=dict(color="#f97316", width=2, dash="dash")))
            fig.update_layout(title="Live Training — Loss Curve", xaxis_title="Epoch",
                               yaxis_title="MSE Loss", plot_bgcolor="white", paper_bgcolor="white",
                               height=300, margin=dict(t=40, b=30))
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        status.success("✅ Training complete!")

        # Evaluation on test set
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        preds = model.predict(X_test_sc).flatten()
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        st.markdown("### 📊 Training Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("R² Score", f"{r2:.4f}")
        m4.metric("Test Samples", f"{len(y_test):,}")

        # Pred vs Actual chart
        fig2 = go.Figure()
        idx = np.argsort(y_test)[:200]
        fig2.add_trace(go.Scatter(x=list(range(len(idx))), y=y_test[idx],
                                  mode="lines", name="Actual", line=dict(color="#1e3a8a", width=2)))
        fig2.add_trace(go.Scatter(x=list(range(len(idx))), y=preds[idx],
                                  mode="lines", name="Predicted", line=dict(color="#ef4444", width=2, dash="dot")))
        fig2.update_layout(title="Predicted vs Actual AQI (first 200 sorted test samples)",
                           plot_bgcolor="white", paper_bgcolor="white", height=320)
        st.plotly_chart(fig2, use_container_width=True)

        # Save model
        save_model(model, scaler, selected_features)
        st.success("💾 Model saved! Go to **Live Predictor** to test it.")
