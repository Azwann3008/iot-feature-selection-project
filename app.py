import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="IoT Vulnerability Detection",
    page_icon="🔐",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>
    /* Font & base */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d1f0d 100%);
        min-height: 100vh;
    }

    /* Main title */
    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0,255,136,0.4);
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #8899aa;
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.05em;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: rgba(0,255,136,0.5);
    }
    .metric-label {
        color: #8899aa;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #00ff88;
    }

    /* Section header */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #00ccff;
        border-left: 3px solid #00ccff;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Status badge */
    .badge-success {
        display: inline-block;
        background: rgba(0,255,136,0.12);
        border: 1px solid rgba(0,255,136,0.4);
        border-radius: 20px;
        color: #00ff88;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 1.5rem;
    }
    .badge-error {
        display: inline-block;
        background: rgba(255,80,80,0.12);
        border: 1px solid rgba(255,80,80,0.4);
        border-radius: 20px;
        color: #ff5050;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 1.5rem;
    }

    /* Result box */
    .result-box {
        background: rgba(0,255,136,0.07);
        border: 1px solid rgba(0,255,136,0.35);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .result-label {
        color: #8899aa;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .result-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: #00ff88;
        margin-top: 0.3rem;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 2rem 0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0,0,0,0.4);
        border-right: 1px solid rgba(0,255,136,0.15);
    }
    [data-testid="stSidebar"] * {
        color: #ccddee !important;
    }

    /* Input labels */
    label {
        color: #aabbcc !important;
        font-size: 0.88rem !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00ccff);
        color: #0a0a1a;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        transition: opacity 0.2s, transform 0.1s;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.85;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Download button */
    .stDownloadButton > button {
        background: rgba(0,204,255,0.12);
        color: #00ccff !important;
        border: 1px solid rgba(0,204,255,0.35);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        border-radius: 8px;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background: rgba(0,204,255,0.22);
    }

    /* Info boxes */
    .stAlert {
        background: rgba(0,204,255,0.07) !important;
        border: 1px solid rgba(0,204,255,0.25) !important;
        border-radius: 10px !important;
        color: #aabbcc !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Number input */
    .stNumberInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #eef2f7 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stNumberInput input:focus {
        border-color: rgba(0,255,136,0.5) !important;
        box-shadow: 0 0 0 2px rgba(0,255,136,0.1) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(0,204,255,0.3);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #aabbcc !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ======================
# LOAD MODEL (Pipeline)
# ======================

@st.cache_resource
def load_pipeline():
    """
    Memuat objek Pipeline secara utuh dari file .pkl.
    Pipeline mencakup: StandardScaler → Feature Selection → Model.
    Dengan memuat Pipeline utuh, input baru akan otomatis melewati
    preprocessing dan feature selection sebelum diprediksi.
    """
    try:
        pipeline = joblib.load("pipeline_terbaik.pkl")
        return pipeline, None
    except FileNotFoundError:
        return None, "File 'pipeline_terbaik.pkl' tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py."
    except Exception as e:
        return None, str(e)

model, load_error = load_pipeline()


# ======================
# SIDEBAR
# ======================

with st.sidebar:
    st.markdown("### 🔐 IoT Security")
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
    st.markdown("**📈 Model Performance (CV)**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", "90.03%")
    with col_b:
        st.metric("F1 Score", "86.71%")

    st.markdown("---")
    st.caption("Mid Semester — Machine Learning II")


# ======================
# MAIN HEADER
# ======================

st.markdown('<div class="main-title">🔐 IoT Vulnerability Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Sistem Deteksi Serangan Trafik IoT · Pipeline-based · No Data Leakage</div>', unsafe_allow_html=True)

# Model load status
if load_error:
    st.markdown(f'<div class="badge-error">⚠ Pipeline gagal dimuat: {load_error}</div>', unsafe_allow_html=True)
    st.stop()
else:
    st.markdown('<div class="badge-success">✓ pipeline_terbaik.pkl berhasil dimuat</div>', unsafe_allow_html=True)

# Info cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">Dataset</div><div class="metric-value" style="font-size:1rem">Balanced IoT</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Feature Selection</div><div class="metric-value" style="font-size:1rem">Embedded</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">90.03%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-value">86.71%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ======================
# CARA PENGGUNAAN
# ======================

with st.expander("ℹ️ Cara Penggunaan Aplikasi"):
    st.markdown("""
    **Input Manual** — Masukkan nilai setiap fitur secara manual, lalu tekan tombol **Prediksi**.  
    Data akan otomatis melewati *StandardScaler* dan *Feature Selection* yang sudah tertanam di dalam Pipeline.

    **Upload CSV** — Upload file CSV yang berisi baris data jaringan IoT.  
    Pastikan kolom CSV **sama persis** dengan fitur yang digunakan saat training (tanpa kolom label/target).  
    Hasil prediksi akan ditambahkan sebagai kolom baru dan dapat diunduh.

    > ⚠️ Pipeline memuat scaler dan feature selector secara otomatis — tidak perlu preprocessing manual di UI ini.
    """)


# ======================
# TAB NAVIGASI
# ======================

tab1, tab2 = st.tabs(["🧠 Input Manual", "📁 Upload CSV"])


# ---- TAB 1: INPUT MANUAL ----
with tab1:
    st.markdown('<div class="section-header">Input Nilai Fitur Secara Manual</div>', unsafe_allow_html=True)

    # Dapatkan jumlah fitur yang diharapkan oleh pipeline
    try:
        # Coba dapatkan nama fitur dari step pertama pipeline (scaler)
        n_features = model.named_steps['scaler'].n_features_in_
        feature_names = getattr(model.named_steps['scaler'], 'feature_names_in_', None)
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    except Exception:
        # Fallback: gunakan 2 fitur default jika pipeline belum difit
        n_features = 2
        feature_names = ["Feature_1", "Feature_2"]

    st.caption(f"Pipeline mengharapkan **{n_features} fitur** sebagai input.")

    # Buat input grid — 4 kolom per baris
    input_values = {}
    cols_per_row = 4
    feature_list = list(feature_names)

    for row_start in range(0, n_features, cols_per_row):
        row_features = feature_list[row_start: row_start + cols_per_row]
        cols = st.columns(len(row_features))
        for col, fname in zip(cols, row_features):
            with col:
                input_values[fname] = st.number_input(
                    label=fname,
                    value=0.0,
                    format="%.6f",
                    key=f"manual_{fname}"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Prediksi", key="btn_manual"):
        try:
            # Buat DataFrame dengan urutan kolom yang benar
            data_input = pd.DataFrame([input_values])

            # Prediksi menggunakan Pipeline utuh
            hasil = model.predict(data_input)
            proba = model.predict_proba(data_input) if hasattr(model, "predict_proba") else None

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Hasil Prediksi</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">🎯 {hasil[0]}</div>', unsafe_allow_html=True)

            if proba is not None:
                classes = model.classes_
                proba_df = pd.DataFrame(
                    proba, columns=[f"P({c})" for c in classes]
                ).round(4)
                st.markdown('<div class="result-label" style="margin-top:1rem">Probabilitas per Kelas</div>', unsafe_allow_html=True)
                st.dataframe(proba_df, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi: {e}")
            st.caption("Pastikan nilai input sesuai dengan format fitur yang digunakan saat training.")


# ---- TAB 2: UPLOAD CSV ----
with tab2:
    st.markdown('<div class="section-header">Prediksi Batch via File CSV</div>', unsafe_allow_html=True)

    st.caption(
        "Upload file CSV yang berisi data jaringan IoT. "
        "Kolom CSV harus sama persis dengan fitur training (tanpa kolom label/target)."
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        key="upload_csv",
        help="Format: CSV dengan header kolom sesuai fitur training"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            st.markdown(f"**Preview Dataset** ({len(data)} baris, {len(data.columns)} kolom)")
            st.dataframe(data.head(10), use_container_width=True)

            # Validasi jumlah kolom
            try:
                expected = model.named_steps['scaler'].n_features_in_
                if data.shape[1] != expected:
                    st.warning(
                        f"⚠️ Dataset memiliki {data.shape[1]} kolom, "
                        f"namun Pipeline mengharapkan {expected} fitur. "
                        "Pastikan kolom target/label sudah dihapus dari file CSV."
                    )
            except Exception:
                pass  # Skip validasi jika pipeline belum difit

            if st.button("🔍 Prediksi Dataset", key="btn_csv"):
                with st.spinner("Memproses data melalui Pipeline..."):
                    prediksi = model.predict(data)
                    data_hasil = data.copy()
                    data_hasil["Prediction"] = prediksi

                    if hasattr(model, "predict_proba"):
                        proba_batch = model.predict_proba(data)
                        classes = model.classes_
                        for i, c in enumerate(classes):
                            data_hasil[f"Prob_{c}"] = proba_batch[:, i].round(4)

                st.success(f"✅ Prediksi selesai — {len(data_hasil)} baris diproses.")

                st.markdown("**Hasil Prediksi (10 baris pertama)**")
                st.dataframe(data_hasil.head(10), use_container_width=True)

                # Ringkasan distribusi prediksi
                st.markdown("**Distribusi Prediksi**")
                dist = data_hasil["Prediction"].value_counts().reset_index()
                dist.columns = ["Kelas", "Jumlah"]
                dist["Persentase"] = (dist["Jumlah"] / len(data_hasil) * 100).round(2).astype(str) + "%"
                st.dataframe(dist, use_container_width=True)

                # Download
                csv_out = data_hasil.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil Prediksi (.csv)",
                    data=csv_out,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    key="btn_download"
                )

        except Exception as e:
            st.error(f"❌ Terjadi error saat memproses file: {e}")
            st.caption("Pastikan file berformat CSV valid dan kolom sesuai dengan fitur training.")


# ======================
# FOOTER
# ======================

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#445566; font-size:0.78rem; font-family:\'JetBrains Mono\', monospace;">'
    'Mid Semester · Machine Learning II · IoT Vulnerability Detection · Pipeline-based (No Data Leakage)'
    '</div>',
    unsafe_allow_html=True
)