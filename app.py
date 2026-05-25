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
# SEMUA KOLOM FITUR (87 kolom, sesuai training)
# ======================

ALL_FEATURES = [
    'dur', 'Protocol', 'Length', 'Source Host', 'Destination Host',
    'Sender IP address', 'Target IP address', 'Opcode', 'checksom(ICMP)',
    'Sequence Number (LE)', 'Sequence Number (BE)', 'File Data',
    'Content length', 'Request URI Query', 'Request Method',
    'Full Request URI', 'Request Version', 'Response', 'Ack No',
    'Ack No (RAW)', 'Checksum(TCP)', 'Connection Finish ',
    'Connection Reset', 'Connection Establish Request',
    'Connection Establish Ack', 'Source Port', 'Destination Port',
    'TCPFlags', 'Acknowledgment', 'TCP Segment Length', 'TCP Options',
    'TCP Payload', 'TCP Seq No', 'Src or Drc port', 'Stream index',
    'Time since previous frame', 'Query Name', 'DNS retransmission',
    'DNS query retransmission', 'DNS query retransmission in',
    'LG bit', 'IG bit', 'LG bit.1', 'Duplicate IP address configured',
    'Time to Live', 'Conversation completeness', 'Push', 'Content Type',
    'This is an ACK to the segment in frame', 'ECN-Echo', 'Mode',
    'Type', 'Type.1', 'Window', 'Echo data', 'Accept', 'Status Code',
    'Transaction ID', 'Handshake Type', 'Flags', 'Packet Type',
    'MSS Value', 'Message type', 'Timestamp value', 'TSecr',
    'TCP Option - SACK permitted', 'Response time', 'No response seen',
    'Kind', 'Duplicate ACK #', 'This frame is a (suspected) retransmission',
    'Previous segment(s) not captured (common at capture start)',
    'FTP Data', 'Bytes in flight', 'Request command', 'Request command.1',
    'BER Error: length is not valid', 'Length.1', 'Flags.1',
    'Packet Length (encrypted)', 'Direction',
    'TCP Option - Maximum segment size', 'Data', 'Checksum',
    'CDATA', 'Label', 'Attack_Category'
]

# 10 fitur paling penting untuk ditampilkan di input manual
TOP_FEATURES = [
    'dur', 'Length', 'Source Port', 'Destination Port',
    'Time to Live', 'TCP Segment Length', 'TCP Payload',
    'Bytes in flight', 'Acknowledgment', 'TCP Seq No'
]

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
        st.markdown("- Jumlah fitur: **87 fitur** jaringan IoT")
        st.markdown("- Kelas target: `Attack_sub_category`")
        st.markdown("- Kelas: Normal + berbagai jenis serangan IoT")
        st.markdown("- Dataset telah di-*balance* agar tiap kelas proporsional")

with col2:
    with st.expander("🌲 Random Forest"):
        st.markdown("**Random Forest Classifier**")
        st.markdown("- Algoritma ensemble berbasis Decision Tree")
        st.markdown("- `n_estimators=100`, `max_depth=10`")
        st.markdown("- Mendukung `predict_proba` untuk probabilitas kelas")
        st.markdown("- Dioptimasi via GridSearchCV / RandomizedSearchCV")
        st.markdown("- Cross Validation: **Stratified K-Fold (5 Fold)**")

with col3:
    with st.expander("⚙️ Embedded Feature Selection"):
        st.markdown("**SelectFromModel (Embedded Method)**")
        st.markdown("- Seleksi fitur dari 87 → 10 fitur terbaik")
        st.markdown("- Berdasarkan *feature importance* Random Forest")
        st.markdown("- Terintegrasi dalam Pipeline → **No Data Leakage**")
        st.markdown("- Urutan: `Scaler → SelectFromModel → Random Forest`")

st.divider()

# ======================
# TAB NAVIGASI
# ======================

tab1, tab2 = st.tabs(["🧠 Input Manual", "📁 Upload CSV"])

# ---- TAB 1: INPUT MANUAL ----
with tab1:
    st.markdown("### Input Nilai Fitur")
    st.caption(
        "Isi 10 fitur utama di bawah. Fitur lainnya otomatis bernilai 0. "
        "Data lengkap (87 fitur) dikirim ke Pipeline: Scaler → Feature Selection → Model."
    )

    input_values = {col: 0.0 for col in ALL_FEATURES}

    col_a, col_b = st.columns(2)
    for i, fname in enumerate(TOP_FEATURES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            input_values[fname] = st.number_input(
                label=fname,
                value=0.0,
                format="%.4f",
                key=f"fi_{fname}"
            )

    st.markdown("")

    if st.button("🔍 Prediksi", key="btn_manual", use_container_width=True):
        try:
            # Kirim semua 87 kolom, dengan nilai dari input atau 0
            data_input = pd.DataFrame([input_values])[ALL_FEATURES]
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

# ---- TAB 2: UPLOAD CSV ----
with tab2:
    st.markdown("### Upload Dataset CSV")
    st.caption(
        "Upload file CSV dengan **87 kolom fitur** yang sama persis seperti saat training "
        "(tanpa kolom `Attack_sub_category`)."
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

            # Validasi kolom
            missing_cols = [c for c in ALL_FEATURES if c not in data.columns]
            extra_cols   = [c for c in data.columns if c not in ALL_FEATURES]

            if missing_cols:
                st.warning(f"⚠️ Kolom berikut tidak ditemukan di CSV: {missing_cols[:5]}{'...' if len(missing_cols)>5 else ''}")
            if extra_cols:
                st.info(f"ℹ️ Kolom berikut akan diabaikan: {extra_cols[:5]}{'...' if len(extra_cols)>5 else ''}")

            if st.button("🔍 Prediksi Dataset", key="btn_csv", use_container_width=True):
                with st.spinner("Memproses melalui Pipeline..."):
                    # Pastikan urutan kolom benar, kolom yang hilang diisi 0
                    for col in ALL_FEATURES:
                        if col not in data.columns:
                            data[col] = 0.0
                    data_ordered = data[ALL_FEATURES]

                    prediksi = model.predict(data_ordered)
                    data_hasil = data_ordered.copy()
                    data_hasil["Prediction"] = prediksi

                    if hasattr(model, "predict_proba"):
                        proba_batch = model.predict_proba(data_ordered)
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

# ======================
# FOOTER
# ======================

st.divider()
st.caption("Mid Semester · Machine Learning II · IoT Vulnerability Detection · No Data Leakage")
