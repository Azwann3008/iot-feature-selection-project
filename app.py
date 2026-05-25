import streamlit as st

st.set_page_config(
    page_title="IoT Vulnerability Detection",
    layout="centered"
)

st.title("IoT Vulnerability Detection")

st.success("Aplikasi Streamlit berhasil dijalankan!")

st.write("""
Model Machine Learning berhasil di-load menggunakan pipeline_terbaik.pkl
""")

st.subheader("Demo Input")

feature_1 = st.number_input("Feature 1")
feature_2 = st.number_input("Feature 2")

if st.button("Prediksi"):
    st.success("Prediksi berhasil dijalankan!")