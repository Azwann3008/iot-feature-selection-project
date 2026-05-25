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

@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("pipeline_terbaik.pkl")
        return pipeline, None
    except FileNotFoundError:
        return None, "File 'pipeline_terbaik.pkl' tidak ditemukan."
    except Exception as e:
        return None, str(e)

model, load_error = load_pipeline()

# ======================
# SIDEBAR
# ======================

with st.sidebar:
    st.markdown("## 🔐 IoT Security")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("IoT Vulnerability Dataset (Preprocessed Balanced)")
    st.markdown("**Feature Selection**")
    st.caption("Embedded Method — SelectFromModel")
    st.markdown("**Classifier**")
    st.caption("Random Forest Classifier")
    st.markdown("**Cross Validation**")
    st.caption("Stratified K-Fold (5 Fold)")
    st.markdown("---")
    st.markdown("**📈 Model Performance**")
    st.metric("Accuracy", "90.03%")
    st.metric("F1 Score", "86.71%")
    st.markdown("---")
    st.caption("Mid Semester — Machine Learning II")

# ======================
# HEADER
# ======================

st.title("🔐 IoT Vulnerability Detection")
st.caption("Sistem deteksi serangan trafik IoT menggunakan Machine Learning Pipeline")

if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()
else:
    st.success("✅ pipeline_terbaik.pkl berhasil dimuat — Pipeline: Scaler → Feature Selection → Model")

st.divider()

# ======================
# 3 CARD KLIKABLE
# ======================

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("📊 Balanced Dataset"):
        st.markdown("**IoT Vulnerability Dataset**")
        st.markdown("- Sumber: Mendeley Data (Preprocessed)")
        st.markdown("- Kelas: Normal + berbagai serangan IoT")
        st.markdown("- Dataset telah di-*balance* agar tiap kelas proporsional")
        st.markdown("- Format: CSV, siap pakai tanpa preprocessing tambahan")

with col2:
    with st.expander("🌲 Random Forest"):
        st.markdown("**Random Forest Classifier**")
        st.markdown("- Algoritma ensemble berbasis Decision Tree")
        st.markdown("- Tahan terhadap overfitting")
        st.markdown("- Mendukung `predict_proba` untuk probabilitas kelas")
        st.markdown("- Dioptimasi via GridSearchCV / RandomizedSearchCV")
        st.markdown("- Cross Validation: **Stratified K-Fold (5 Fold)**")

with col3:
    with st.expander("⚙️ Embedded Feature Selection"):
        st.markdown("**SelectFromModel (Embedded Method)**")
        st.markdown("- Seleksi fitur berdasarkan *feature importance* dari Random Forest")
        st.markdown("- Terintegrasi di dalam Pipeline → **No Data Leakage**")
        st.markdown("- Fitur dengan importance di bawah threshold otomatis dibuang")
        st.markdown("- Urutan Pipeline: `Scaler → SelectFromModel → Random Forest`")

st.divider()

# ======================
# TAB: INPUT MANUAL & CSV
# ======================

tab1, tab2 = st.tabs(["🧠 Input Manual", "📁 Upload CSV"])

# ---- TAB 1: INPUT MANUAL (hanya 2 fitur) ----
with tab1:
    st.markdown("### Input Nilai Fitur")
    st.caption(
        "Masukkan nilai dua fitur utama jaringan IoT. "
        "Data akan diproses otomatis melalui Pipeline: Scaler → Feature Selection → Model."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        src_bytes = st.number_input(
            label="src_bytes  (Jumlah byte dari sumber)",
            value=0.0,
            format="%.4f",
            key="fi_src_bytes"
        )
    with col_b:
        dst_bytes = st.number_input(
            label="dst_bytes  (Jumlah byte ke tujuan)",
            value=0.0,
            format="%.4f",
            key="fi_dst_bytes"
        )

    st.markdown("")

    if st.button("🔍 Prediksi", key="btn_manual", use_container_width=True):
        try:
            data_input = pd.DataFrame([{
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes
            }])
            hasil = model.predict(data_input)
            st.success(f"🎯 Hasil Prediksi: **{hasil[0]}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(data_input)
                proba_df = pd.DataFrame(
                    proba,
                    columns=[str(c) for c in model.classes_]
                ).round(4)
                st.markdown("**Probabilitas per Kelas:**")
                st.dataframe(proba_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.caption("Pastikan nama kolom sesuai dengan fitur yang digunakan saat training.")

# ---- TAB 2: UPLOAD CSV ----
with tab2:
    st.markdown("### Upload Dataset CSV")
    st.caption(
        "Upload file CSV dengan kolom fitur yang **sama persis** "
        "seperti saat training (tanpa kolom label/target)."
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        key="upload_csv"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown(f"**Preview:** {len(data)} baris · {len(data.columns)} kolom")
            st.dataframe(data.head(), use_container_width=True)

            if st.button("🔍 Prediksi Dataset", key="btn_csv", use_container_width=True):
                with st.spinner("Memproses melalui Pipeline..."):
                    prediksi = model.predict(data)
                    data_hasil = data.copy()
                    data_hasil["Prediction"] = prediksi

                    if hasattr(model, "predict_proba"):
                        proba_batch = model.predict_proba(data)
                        for i, c in enumerate(model.classes_):
                            data_hasil[f"Prob_{c}"] = proba_batch[:, i].round(4)

                st.success(f"✅ {len(data_hasil)} baris selesai diprediksi.")
                st.dataframe(data_hasil.head(10), use_container_width=True)

                st.markdown("**Distribusi Prediksi:**")
                dist = data_hasil["Prediction"].value_counts().reset_index()
                dist.columns = ["Kelas", "Jumlah"]
                dist["Persentase"] = (dist["Jumlah"] / len(data_hasil) * 100).round(2).astype(str) + "%"
                st.dataframe(dist, use_container_width=True)

                csv_out = data_hasil.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil Prediksi (.csv)",
                    data=csv_out,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    key="btn_download"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.caption("Pastikan file CSV valid dan kolomnya sesuai fitur training.")

# ======================
# FOOTER
# ======================

st.divider()
st.caption("Mid Semester · Machine Learning II · IoT Vulnerability Detection · No Data Leakage")
