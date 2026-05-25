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

# ======================
# INPUT MANUAL
# ======================

st.divider()

st.header("🧠 Input Manual Prediksi")

feature1 = st.number_input(
    "Feature 1",
    value=0.0,
    key="f1"
)

feature2 = st.number_input(
    "Feature 2",
    value=0.0,
    key="f2"
)

if st.button("Prediksi", key="btn_prediksi"):

    try:
        data_baru = [[feature1, feature2]]

        hasil = model.predict(data_baru)

        st.success("Prediksi berhasil!")

        st.write("Hasil Prediksi:")
        st.write(hasil[0])

    except Exception as e:
        st.error(f"Terjadi error: {e}")

# ======================
# UPLOAD CSV
# ======================

st.divider()

st.header("📁 Upload Dataset CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV untuk diprediksi",
    type=["csv"],
    key="upload_csv"
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("Preview Dataset:")
        st.dataframe(data.head())

        prediksi = model.predict(data)

        data["Prediction"] = prediksi

        st.success("Prediksi dataset berhasil!")

        st.write("Hasil Prediksi:")
        st.dataframe(data.head())

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi error saat prediksi dataset: {e}")