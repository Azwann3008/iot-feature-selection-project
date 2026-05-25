import streamlit as st
import joblib

# Load model
model = joblib.load("pipeline_terbaik.pkl")

st.title("IoT Vulnerability Detection")

st.success("Aplikasi Streamlit berhasil dijalankan!")

# Input
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

# Prediksi
if st.button("Prediksi"):

    data_baru = [[feature1, feature2]]

    hasil = model.predict(data_baru)

    st.success(f"Hasil Prediksi: {hasil[0]}")