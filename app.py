import streamlit as st
import joblib
import pandas as pd

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="IoT Vulnerability Detection",
    page_icon="🔐",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================

model = joblib.load("pipeline_terbaik.pkl")

# ======================
# SIDEBAR
# ======================

st.sidebar.title("🔐 IoT Security")

st.sidebar.markdown("## Project Information")

st.sidebar.markdown("### Dataset")
st.sidebar.write("IoT Vulnerability Dataset")

st.sidebar.markdown("### Feature Selection")
st.sidebar.write("Embedded Method (SelectFromModel)")

st.sidebar.markdown("### Classifier")
st.sidebar.write("Random Forest")

st.sidebar.markdown("### Cross Validation")
st.sidebar.write("Stratified K-Fold (5 Fold)")

st.sidebar.markdown("---")

st.sidebar.markdown("## Model Performance")
st.sidebar.metric("Accuracy", "90.03%")
st.sidebar.metric("F1 Score", "86.71%")

# ======================
# MAIN PAGE
# ======================

st.title("🔐 IoT Vulnerability Detection System")

st.write("""
Sistem ini digunakan untuk mendeteksi jenis serangan pada trafik IoT menggunakan:
""")

st.markdown("""
- StandardScaler
- Embedded Feature Selection
- Random Forest Classifier
""")

st.success("Pipeline berhasil dimuat (pipeline_terbaik.pkl)")

# ======================
# INFO BOX
# ======================

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 Balanced Dataset")

with col2:
    st.info("🌲 Random Forest")

with col3:
    st.info("⚙️ Embedded Feature Selection")

st.divider()

# ======================
# PREDIKSI DATASET CSV
# ======================

st.divider()

st.header("📁 Upload Dataset CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV untuk diprediksi",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Preview Dataset:")
    st.dataframe(data.head())

    try:
        prediksi = model.predict(data)

        data["Prediction"] = prediksi

        st.success("Prediksi berhasil dilakukan!")

        st.write("Hasil Prediksi:")
        st.dataframe(data.head())

    except Exception as e:
        st.error(f"Terjadi error: {e}")

# ======================
# UPLOAD CSV
# ======================

st.divider()

st.header("📁 Upload Dataset CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV untuk diprediksi",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview Dataset:")
    st.dataframe(data.head())

    prediksi = model.predict(data)

    data["Prediction"] = prediksi

    st.write("Hasil Prediksi:")
    st.dataframe(data.head())