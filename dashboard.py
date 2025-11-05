import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from pathlib import Path
import plotly.graph_objects as go

# ================================================================
# APP CONFIGURATION
# ================================================================
st.set_page_config(page_title="🔋 Battery RUL Predictor", layout="wide")

st.title("🔋  Battery RUL Prediction Dashboard")
st.markdown("""
Upload your **battery sensor dataset (CSV)** to predict **Remaining Useful Life (RUL)** using your trained LSTM model.
""")

# ================================================================
# LOAD TRAINED MODEL + SCALERS
# ================================================================
MODEL_PATH = Path("outputs/models/fixed_model.keras")
SCALER_PATH = Path("outputs/models/scalers.pkl")

@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    return model, scalers

try:
    model, scalers = load_model_and_scalers()
    st.sidebar.success("✅ Model & Scalers Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model or scalers: {e}")
    st.stop()

scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]
selected_features = scalers["feature_names"]
SEQ_LEN = scalers["sequence_length"]

# ================================================================
# USER INPUT
# ================================================================
uploaded_file = st.file_uploader("📂 Upload a Battery Data File (CSV)", type=["csv"])
user_name = st.text_input("👤 Enter Your Name", placeholder="e.g., Aseem")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

    # Check if all required features are present
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        st.error(f"⚠️ Missing required columns: {missing}")
        st.stop()

    st.markdown(f"### 🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ============================================================
    # PREPROCESS AND PREDICT
    # ============================================================
    try:
        X = df[selected_features].fillna(0).values
        # Create overlapping sequences
        sequences = []
        for i in range(0, len(X) - SEQ_LEN):
            sequences.append(X[i:i+SEQ_LEN])
        X_seq = np.array(sequences)

        # Scale
        n_samp, n_steps, n_feat = X_seq.shape
        X_scaled = scaler_X.transform(X_seq.reshape(-1, n_feat)).reshape(n_samp, n_steps, n_feat)

        # Predict RUL
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # ============================================================
        # VISUALIZATION
        # ============================================================
        st.markdown("---")
        st.markdown(f"### 📈 Predicted Remaining Useful Life (RUL) Trend — for {user_name or 'User'}")
        cycles = np.arange(len(y_pred))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=y_pred,
            mode="lines+markers",
            line=dict(color="#16a34a", width=2),
            name="Predicted RUL"
        ))
        fig.update_layout(
            xaxis_title="Cycle Index",
            yaxis_title="Predicted RUL (cycles)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # SUMMARY STATISTICS
        # ============================================================
        avg_rul = np.mean(y_pred)
        min_rul, max_rul = np.min(y_pred), np.max(y_pred)

        st.markdown("### 🧾 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("🔋 Average RUL", f"{avg_rul:.1f} cycles")
        col2.metric("⚠️ Min RUL", f"{min_rul:.1f} cycles")
        col3.metric("🚀 Max RUL", f"{max_rul:.1f} cycles")

        st.markdown("---")
        st.markdown("### 💡 Insights & Recommendations")

        if avg_rul > 500:
            st.success("✅ Excellent battery health — RUL is high, no maintenance needed.")
        elif avg_rul > 200:
            st.warning("⚠️ Battery showing moderate wear — schedule maintenance soon.")
        else:
            st.error("🔴 Battery nearing end of life — replacement or diagnostics recommended.")

        # ============================================================
        # DOWNLOAD RESULTS
        # ============================================================
        report_text = f"""
        Battery Health Report for {user_name or 'User'}
        ---------------------------------------------
        Average RUL: {avg_rul:.2f} cycles
        Min RUL: {min_rul:.2f} cycles
        Max RUL: {max_rul:.2f} cycles
        Sequence Length: {SEQ_LEN}
        Model: LSTM
        """
        st.download_button("📄 Download RUL Report", data=report_text, file_name="battery_rul_report.txt")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

else:
    st.info("📤 Please upload a CSV file to begin the analysis.")
