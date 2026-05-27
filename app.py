# ============================================================
# app.py — Tahap 10: Aplikasi Streamlit
# IoT Vulnerability Detection System
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import random

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="IoT Vulnerability Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Futuristic / Cyberpunk Terminal Aesthetic
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}
.stApp {
    background-color: #050a0e;
    background-image:
        linear-gradient(rgba(0,255,180,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,180,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Scanline overlay ── */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 999;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(0,8,16,0.95) !important;
    border-right: 1px solid rgba(0,255,180,0.2) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #7affda !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stSidebar"] .stMetric {
    background: rgba(0,255,180,0.05);
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 4px;
    padding: 8px;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #00ffb4 !important;
    font-size: 1.4rem !important;
}

/* ── Main title ── */
h1 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ffb4 !important;
    font-size: 2rem !important;
    letter-spacing: 0.1em !important;
    text-shadow: 0 0 30px rgba(0,255,180,0.5), 0 0 60px rgba(0,255,180,0.2);
}
h2, h3 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ccff !important;
    letter-spacing: 0.05em !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,255,180,0.2) !important;
    margin: 1.5rem 0 !important;
}

/* ── Success / Error / Info ── */
.stSuccess {
    background: rgba(0,255,180,0.08) !important;
    border: 1px solid rgba(0,255,180,0.4) !important;
    border-radius: 4px !important;
    color: #00ffb4 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stError {
    background: rgba(255,50,50,0.08) !important;
    border: 1px solid rgba(255,50,50,0.4) !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stWarning {
    background: rgba(255,200,0,0.06) !important;
    border: 1px solid rgba(255,200,0,0.3) !important;
    border-radius: 4px !important;
}
.stInfo {
    background: rgba(0,200,255,0.06) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    border-radius: 4px !important;
}

/* ── Number input ── */
.stNumberInput input {
    background: rgba(0,20,30,0.9) !important;
    border: 1px solid rgba(0,255,180,0.25) !important;
    border-radius: 4px !important;
    color: #00ffb4 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus {
    border-color: rgba(0,255,180,0.8) !important;
    box-shadow: 0 0 12px rgba(0,255,180,0.2) !important;
}
label {
    color: #7affda !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Button ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ffb4 !important;
    color: #00ffb4 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.15em !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    background: rgba(0,255,180,0.1) !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.3), inset 0 0 20px rgba(0,255,180,0.05) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,200,255,0.5) !important;
    color: #00ccff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 4px !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: rgba(0,200,255,0.08) !important;
    box-shadow: 0 0 15px rgba(0,200,255,0.2) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(0,15,25,0.8) !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
    border-radius: 4px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(0,255,180,0.35) !important;
}
.streamlit-expanderHeader p {
    font-family: 'Share Tech Mono', monospace !important;
    color: #7affda !important;
    letter-spacing: 0.08em !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Share Tech Mono', monospace !important;
    color: #7affda !important;
    letter-spacing: 0.1em !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00ffb4 !important;
    border-bottom: 2px solid #00ffb4 !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid rgba(0,255,180,0.15) !important;
    border-radius: 4px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(0,15,25,0.6) !important;
    border: 1px dashed rgba(0,200,255,0.3) !important;
    border-radius: 4px !important;
}

/* ── Caption / small text ── */
.stCaption, small {
    font-family: 'Share Tech Mono', monospace !important;
    color: rgba(122,255,218,0.5) !important;
    font-size: 0.78rem !important;
}

/* ── Subheader ── */
.stSubheader {
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Metric cards in sidebar ── */
[data-testid="metric-container"] {
    background: rgba(0,255,180,0.04) !important;
    border: 1px solid rgba(0,255,180,0.15) !important;
    border-radius: 4px !important;
    padding: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA
# ============================================================

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
    'Kind', 'Duplicate ACK #',
    'This frame is a (suspected) retransmission',
    'Previous segment(s) not captured (common at capture start)',
    'FTP Data', 'Bytes in flight', 'Request command', 'Request command.1',
    'BER Error: length is not valid', 'Length.1', 'Flags.1',
    'Packet Length (encrypted)', 'Direction',
    'TCP Option - Maximum segment size', 'Data', 'Checksum',
    'CDATA', 'Label', 'Attack_Category'
]

TOP_FEATURES = [
    'dur', 'Length', 'Source Port', 'Destination Port',
    'Time to Live', 'TCP Segment Length', 'TCP Payload',
    'Bytes in flight', 'Acknowledgment', 'TCP Seq No'
]

# Range realistis dari dataset asli (min, max, default_contoh)
FEATURE_RANGES = {
    'dur'               : (0.0,       7.97,       0.05),
    'Length'            : (42.0,      5858.0,     112.0),
    'Source Port'       : (0.0,       65535.0,    49152.0),
    'Destination Port'  : (0.0,       65534.0,    80.0),
    'Time to Live'      : (0.0,       90.0,       64.0),
    'TCP Segment Length': (0.0,       5792.0,     40.0),
    'TCP Payload'       : (0.0,       16866.0,    1024.0),
    'Bytes in flight'   : (0.0,       505440.0,   512.0),
    'Acknowledgment'    : (0.0,       3.0,        1.0),
    'TCP Seq No'        : (0.0,       4294884607, 1048576.0),
}

LABEL_MAP = {
    0: "ARPPoisoning", 1: "Backdoor",     2: "ICMPflood",
    3: "ICMPredirect", 4: "Normal",       5: "Password_crack",
    6: "Port_Scanning",7: "SQLInjection", 8: "SYN_FLOOD",
    9: "Smurf",        10:"UDP_flood",    11:"VUlnerability_Scan"
}

def random_sample():
    """Generate nilai acak realistis berdasarkan range dataset asli."""
    return {f: round(random.uniform(FEATURE_RANGES[f][0], FEATURE_RANGES[f][1]), 4)
            for f in TOP_FEATURES}

def normal_sample():
    """Contoh trafik Normal."""
    return {
        'dur': 0.05, 'Length': 74.0, 'Source Port': 49152.0,
        'Destination Port': 80.0, 'Time to Live': 64.0,
        'TCP Segment Length': 20.0, 'TCP Payload': 512.0,
        'Bytes in flight': 512.0, 'Acknowledgment': 1.0,
        'TCP Seq No': 1048576.0
    }

def attack_sample():
    """Contoh trafik SYN Flood."""
    return {
        'dur': 0.0, 'Length': 42.0, 'Source Port': 12345.0,
        'Destination Port': 80.0, 'Time to Live': 45.0,
        'TCP Segment Length': 0.0, 'TCP Payload': 0.0,
        'Bytes in flight': 0.0, 'Acknowledgment': 0.0,
        'TCP Seq No': 99999999.0
    }

# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("pipeline_terbaik.pkl")
        return pipeline, None
    except FileNotFoundError:
        return None, "File 'pipeline_terbaik.pkl' tidak ditemukan. Jalankan mlflow.py terlebih dahulu."
    except Exception as e:
        return None, str(e)

model, load_error = load_pipeline()

# ============================================================
# SESSION STATE untuk nilai input
# ============================================================

if "input_vals" not in st.session_state:
    st.session_state.input_vals = normal_sample()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## `// IoT SECURITY`")
    st.markdown("---")
    st.markdown("`DATASET`")
    st.caption("IoT Vulnerability (Preprocessed Balanced)")
    st.markdown("`FEATURE SELECTION`")
    st.caption("Embedded — SelectFromModel")
    st.markdown("`CLASSIFIER`")
    st.caption("Random Forest Classifier")
    st.markdown("`CROSS VALIDATION`")
    st.caption("Stratified K-Fold (5 Fold)")
    st.markdown("---")
    st.markdown("`// MODEL PERFORMANCE`")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("ACCURACY", "90.03%")
        st.metric("PRECISION", "87.45%")
    with col_s2:
        st.metric("F1 SCORE", "86.71%")
        st.metric("RECALL", "86.71%")
    st.markdown("---")
    st.markdown("`// PIPELINE`")
    st.caption("① StandardScaler")
    st.caption("② SelectFromModel")
    st.caption("③ RandomForest (n=100, d=10)")
    st.markdown("---")
    st.caption("Mid Semester — Machine Learning II")

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="
    border-left: 3px solid #00ffb4;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    background: rgba(0,255,180,0.03);
">
    <div style="
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        color: #00ffb4;
        text-shadow: 0 0 20px rgba(0,255,180,0.4);
        letter-spacing: 0.1em;
    ">🔐 IoT VULNERABILITY DETECTION</div>
    <div style="
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: rgba(122,255,218,0.5);
        letter-spacing: 0.15em;
        margin-top: 0.3rem;
    ">PIPELINE :: StandardScaler → SelectFromModel → RandomForestClassifier</div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠ SYSTEM ERROR :: {load_error}")
    st.stop()
else:
    st.success(f"✓ PIPELINE LOADED :: pipeline_terbaik.pkl — {len(ALL_FEATURES)} features → Scaler → FeatureSelection → Model")

st.divider()

# ============================================================
# 3 INFO CARDS
# ============================================================

st.markdown("### `// SYSTEM INFO`")
st.caption("Klik untuk expand detail.")

c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("📊 BALANCED DATASET", expanded=False):
        st.markdown("**IoT Vulnerability Dataset**")
        st.markdown("- Sumber: Mendeley Data")
        st.markdown("- Jumlah fitur: **87 kolom**")
        st.markdown("- Target: `Attack_sub_category`")
        st.markdown("- Kelas: **12 kelas** serangan & normal")
        st.markdown("- Distribusi: Balanced per kelas")
        st.markdown("- Split: 80% train / 20% test (stratified)")
        st.markdown("---")
        for k, v in LABEL_MAP.items():
            icon = "🟢" if v == "Normal" else "🔴"
            st.markdown(f"{icon} `{v}`")

with c2:
    with st.expander("🌲 RANDOM FOREST", expanded=False):
        st.markdown("**Random Forest Classifier**")
        st.markdown("- Ensemble berbasis Decision Tree")
        st.markdown("- Voting mayoritas → tahan overfitting")
        st.markdown("---")
        st.markdown("**Best Params (GridSearchCV Tahap 7):**")
        st.code("n_estimators      = 100\nmax_depth         = 10\nmin_samples_split = 2\nmin_samples_leaf  = 1\nrandom_state      = 42", language="python")
        st.markdown("---")
        st.markdown("**Validasi:**")
        st.markdown("- Stratified K-Fold (5 Fold)")
        st.markdown("- Best CV Accuracy: **99.09%**")

with c3:
    with st.expander("⚙️ EMBEDDED FEATURE SELECTION", expanded=False):
        st.markdown("**SelectFromModel**")
        st.markdown("- 87 fitur → **15 fitur terpilih**")
        st.markdown("- Seleksi via feature importance RF")
        st.markdown("- Di dalam Pipeline → **No Data Leakage**")
        st.markdown("---")
        st.markdown("**Perbandingan Metode:**")
        st.code("Filter  → SelectKBest (f_classif)\nWrapper → RFE\nEmbedded → SelectFromModel ✓", language="text")
        st.markdown("---")
        st.markdown("**Pipeline Order:**")
        st.code("Scaler → SelectFromModel → RandomForest", language="text")

st.divider()

# ============================================================
# KELAS & CARA PAKAI
# ============================================================

col_info1, col_info2 = st.columns(2)
with col_info1:
    with st.expander("🎯 KELAS YANG DIDETEKSI (12 KELAS)", expanded=False):
        k1, k2, k3 = st.columns(3)
        for i, (_, v) in enumerate(LABEL_MAP.items()):
            icon = "🟢" if v == "Normal" else "🔴"
            tipe = "Normal" if v == "Normal" else "Attack"
            if i % 3 == 0: k1.markdown(f"{icon} `{v}`")
            elif i % 3 == 1: k2.markdown(f"{icon} `{v}`")
            else: k3.markdown(f"{icon} `{v}`")

with col_info2:
    with st.expander("📖 CARA PENGGUNAAN", expanded=False):
        st.markdown("""
**Tab Input Manual:**
- Isi 10 fitur utama atau klik tombol sample
- Fitur lain otomatis bernilai 0
- Semua 87 kolom dikirim ke Pipeline
- Tekan **▶ ANALYZE** untuk prediksi

**Tab Upload CSV:**
- Upload CSV 87 kolom (tanpa `Attack_sub_category`)
- Kolom kurang otomatis diisi 0
- Tekan **▶ BATCH ANALYZE**

> Pipeline bekerja utuh — tidak ada preprocessing manual.
        """)

st.divider()

# ============================================================
# TAB PREDIKSI
# ============================================================

tab1, tab2 = st.tabs(["▶  INPUT MANUAL", "▶  UPLOAD CSV"])

# ----------------------------------------------------------
# TAB 1
# ----------------------------------------------------------
with tab1:
    st.markdown("### `// PACKET ANALYZER`")
    st.caption("Isi nilai fitur jaringan IoT. Gunakan tombol sample untuk nilai contoh.")

    # Tombol sample
    btn1, btn2, btn3, _ = st.columns([1, 1, 1, 3])
    with btn1:
        if st.button("⚡ RANDOM", key="btn_random"):
            st.session_state.input_vals = random_sample()
            st.rerun()
    with btn2:
        if st.button("🟢 NORMAL", key="btn_normal"):
            st.session_state.input_vals = normal_sample()
            st.rerun()
    with btn3:
        if st.button("🔴 ATTACK", key="btn_attack"):
            st.session_state.input_vals = attack_sample()
            st.rerun()

    st.markdown("")

    # Input grid 2 kolom
    input_values = {col: 0.0 for col in ALL_FEATURES}
    col_a, col_b = st.columns(2)

    for i, fname in enumerate(TOP_FEATURES):
        default_val = float(st.session_state.input_vals.get(fname, 0.0))
        col = col_a if i % 2 == 0 else col_b
        with col:
            input_values[fname] = st.number_input(
                label=fname,
                value=default_val,
                format="%.4f",
                key=f"fi_{fname}"
            )

    st.markdown("")

    if st.button("▶  ANALYZE PACKET", key="btn_manual", use_container_width=True):
        try:
            data_input = pd.DataFrame([input_values])[ALL_FEATURES]
            hasil_raw   = model.predict(data_input)[0]
            hasil_label = LABEL_MAP.get(int(hasil_raw), str(hasil_raw))

            if hasil_label == "Normal":
                st.success(f"🟢 RESULT :: {hasil_label} — Trafik normal, tidak terdeteksi ancaman.")
            else:
                st.error(f"🔴 THREAT DETECTED :: {hasil_label} — Serangan teridentifikasi!")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(data_input)
                proba_df = pd.DataFrame(
                    proba,
                    columns=[str(c) for c in model.classes_]
                ).round(4)
                with st.expander("📊 PROBABILITY DISTRIBUTION", expanded=True):
                    st.dataframe(proba_df, use_container_width=True)

        except Exception as e:
            st.error(f"⚠ ERROR :: {e}")
            st.caption("Pastikan nama kolom sesuai dengan fitur training.")

# ----------------------------------------------------------
# TAB 2
# ----------------------------------------------------------
with tab2:
    st.markdown("### `// BATCH ANALYZER`")
    st.caption("Upload CSV dengan 87 kolom fitur (tanpa kolom `Attack_sub_category`).")

    uploaded_file = st.file_uploader(
        "SELECT FILE",
        type=["csv"],
        key="upload_csv"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown(f"**FILE LOADED** :: {len(data)} rows · {len(data.columns)} columns")
            st.dataframe(data.head(), use_container_width=True)

            missing_cols = [c for c in ALL_FEATURES if c not in data.columns]
            extra_cols   = [c for c in data.columns if c not in ALL_FEATURES]

            if missing_cols:
                st.warning(f"⚠ {len(missing_cols)} kolom hilang → diisi 0 :: {missing_cols[:3]}{'...' if len(missing_cols)>3 else ''}")
            if extra_cols:
                st.info(f"ℹ {len(extra_cols)} kolom tidak dikenal → diabaikan")

            if st.button("▶  BATCH ANALYZE", key="btn_csv", use_container_width=True):
                with st.spinner("Processing pipeline :: Scaler → SelectFromModel → Model..."):
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

                st.success(f"✓ DONE :: {len(data_hasil)} packets analyzed.")

                st.markdown("**RESULT PREVIEW (top 10):**")
                st.dataframe(data_hasil.head(10), use_container_width=True)

                with st.expander("📊 PREDICTION DISTRIBUTION", expanded=True):
                    dist = data_hasil["Prediction"].value_counts().reset_index()
                    dist.columns = ["Class", "Count"]
                    dist["%"] = (dist["Count"] / len(data_hasil) * 100).round(2).astype(str) + "%"
                    dist["Type"] = dist["Class"].apply(
                        lambda x: "🟢 Normal" if str(x) in ["Normal","4"] else "🔴 Attack"
                    )
                    st.dataframe(dist, use_container_width=True)

                csv_out = data_hasil.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇  EXPORT RESULTS (.csv)",
                    data=csv_out,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    key="btn_download"
                )

        except Exception as e:
            st.error(f"⚠ ERROR :: {e}")
            st.caption("Pastikan file CSV valid dan kolomnya sesuai fitur training.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style="
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: rgba(122,255,218,0.3);
    text-align: center;
    letter-spacing: 0.15em;
">
MID SEMESTER · MACHINE LEARNING II · IOT VULNERABILITY DETECTION ·
PIPELINE: StandardScaler → SelectFromModel → RandomForestClassifier · NO DATA LEAKAGE
</div>
""", unsafe_allow_html=True)