# ============================================================
# app.py — Tahap 10: Aplikasi Streamlit
# IoT Vulnerability Detection System
# ============================================================
# Pipeline dimuat UTUH dari pipeline_terbaik.pkl.
# Semua input melewati: Scaler → SelectFromModel → Model
# Upload limit: 1 GB (via CLI flag)
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import os

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
# CSS — Futuristic + Smooth
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background-color: #050a0e;
    background-image:
        linear-gradient(rgba(0,255,180,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,180,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
}
.main .block-container { animation: fadeInUp 0.5s ease both; }
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0);    }
}
.stApp::before {
    content:""; position:fixed; top:0; left:0; width:100%; height:100%;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.05) 2px,rgba(0,0,0,0.05) 4px);
    pointer-events:none; z-index:999;
}

[data-testid="stSidebar"] {
    background: rgba(0,8,16,0.97) !important;
    border-right: 1px solid rgba(0,255,180,0.18) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color:#7affda !important; font-family:'Share Tech Mono',monospace !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color:#00ffb4 !important; font-size:1.25rem !important;
    text-shadow:0 0 10px rgba(0,255,180,0.35);
}
[data-testid="metric-container"] {
    background:rgba(0,255,180,0.03) !important;
    border:1px solid rgba(0,255,180,0.12) !important;
    border-radius:4px !important; padding:0.5rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="metric-container"]:hover { border-color:rgba(0,255,180,0.3) !important; }

h1 { font-family:'Share Tech Mono',monospace !important; color:#00ffb4 !important;
     text-shadow:0 0 20px rgba(0,255,180,0.35) !important; letter-spacing:0.08em !important; }
h2, h3 { font-family:'Share Tech Mono',monospace !important; color:#00ccff !important; letter-spacing:0.05em !important; }
hr { border:none !important; border-top:1px solid rgba(0,255,180,0.15) !important; margin:1.5rem 0 !important; }

.stSuccess > div {
    background:rgba(0,255,180,0.07) !important; border:1px solid rgba(0,255,180,0.35) !important;
    border-radius:4px !important; color:#00ffb4 !important;
    font-family:'Share Tech Mono',monospace !important; animation:slideIn 0.35s ease both;
}
.stError > div {
    background:rgba(255,50,50,0.07) !important; border:1px solid rgba(255,60,60,0.4) !important;
    border-radius:4px !important; font-family:'Share Tech Mono',monospace !important;
    animation:slideIn 0.35s ease both;
}
.stWarning > div { background:rgba(255,200,0,0.05) !important; border:1px solid rgba(255,200,0,0.28) !important; border-radius:4px !important; }
.stInfo > div    { background:rgba(0,200,255,0.05) !important; border:1px solid rgba(0,200,255,0.25) !important; border-radius:4px !important; }
@keyframes slideIn { from{opacity:0;transform:translateX(-8px);} to{opacity:1;transform:translateX(0);} }

.stNumberInput input {
    background:rgba(0,18,28,0.95) !important; border:1px solid rgba(0,255,180,0.2) !important;
    border-radius:4px !important; color:#00ffb4 !important;
    font-family:'Share Tech Mono',monospace !important; font-size:0.95rem !important;
    transition:border-color 0.25s ease, box-shadow 0.25s ease !important; caret-color:#00ffb4 !important;
}
.stNumberInput input:focus {
    border-color:#00ffb4 !important;
    box-shadow:0 0 0 2px rgba(0,255,180,0.12), 0 0 10px rgba(0,255,180,0.08) !important;
}
.stNumberInput input:hover { border-color:rgba(0,255,180,0.4) !important; }

label { color:rgba(122,255,218,0.8) !important; font-family:'Share Tech Mono',monospace !important;
        font-size:0.78rem !important; letter-spacing:0.06em !important; }

.stButton > button {
    background:transparent !important; border:1px solid rgba(0,255,180,0.6) !important;
    color:#00ffb4 !important; font-family:'Share Tech Mono',monospace !important;
    font-size:0.85rem !important; letter-spacing:0.16em !important; text-transform:uppercase !important;
    border-radius:4px !important; padding:0.55rem 1.5rem !important; width:100% !important;
    transition:all 0.22s ease !important;
}
.stButton > button:hover {
    background:rgba(0,255,180,0.08) !important; border-color:#00ffb4 !important;
    box-shadow:0 0 16px rgba(0,255,180,0.22), inset 0 0 16px rgba(0,255,180,0.04) !important;
    transform:translateY(-2px) !important; color:#fff !important;
}
.stButton > button:active { transform:translateY(0) !important; }

.stDownloadButton > button {
    background:transparent !important; border:1px solid rgba(0,200,255,0.45) !important;
    color:#00ccff !important; font-family:'Share Tech Mono',monospace !important;
    font-size:0.82rem !important; letter-spacing:0.12em !important;
    border-radius:4px !important; width:100% !important; transition:all 0.22s ease !important;
}
.stDownloadButton > button:hover {
    background:rgba(0,200,255,0.07) !important; box-shadow:0 0 14px rgba(0,200,255,0.2) !important;
    transform:translateY(-2px) !important;
}

[data-testid="stExpander"] {
    background:rgba(0,12,22,0.85) !important; border:1px solid rgba(0,255,180,0.12) !important;
    border-radius:6px !important; margin-bottom:0.4rem !important;
    transition:border-color 0.22s ease, box-shadow 0.22s ease !important;
}
[data-testid="stExpander"]:hover { border-color:rgba(0,255,180,0.3) !important; box-shadow:0 0 10px rgba(0,255,180,0.05) !important; }
.streamlit-expanderHeader p { font-family:'Share Tech Mono',monospace !important; color:#7affda !important; letter-spacing:0.08em !important; transition:color 0.2s ease !important; }
.streamlit-expanderHeader:hover p { color:#00ffb4 !important; }
[data-testid="stExpanderDetails"] { animation:expandFade 0.28s ease both; }
@keyframes expandFade { from{opacity:0;transform:translateY(-4px);} to{opacity:1;transform:translateY(0);} }

[data-testid="stTabs"] button {
    font-family:'Share Tech Mono',monospace !important; color:rgba(122,255,218,0.55) !important;
    letter-spacing:0.12em !important; font-size:0.85rem !important;
    transition:color 0.2s ease !important; border-bottom:2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#00ffb4 !important; border-bottom:2px solid #00ffb4 !important;
    text-shadow:0 0 8px rgba(0,255,180,0.3) !important;
}
[data-testid="stTabs"] button:hover { color:#00ffb4 !important; }

.stDataFrame { border:1px solid rgba(0,255,180,0.12) !important; border-radius:6px !important; animation:fadeInUp 0.35s ease both; }
[data-testid="stFileUploader"] {
    background:rgba(0,12,22,0.7) !important; border:1px dashed rgba(0,200,255,0.25) !important;
    border-radius:6px !important; transition:border-color 0.22s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color:rgba(0,200,255,0.5) !important; }

.stSpinner > div { border-color:#00ffb4 transparent transparent transparent !important; }
.stCaption, small { font-family:'Share Tech Mono',monospace !important; color:rgba(122,255,218,0.38) !important; font-size:0.75rem !important; }
code {
    background:rgba(0,255,180,0.07) !important; border:1px solid rgba(0,255,180,0.15) !important;
    color:#00ffb4 !important; border-radius:3px !important;
    font-family:'Share Tech Mono',monospace !important; font-size:0.85em !important; padding:0.1em 0.4em !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA
# ============================================================

ALL_FEATURES = [
    'dur','Protocol','Length','Source Host','Destination Host',
    'Sender IP address','Target IP address','Opcode','checksom(ICMP)',
    'Sequence Number (LE)','Sequence Number (BE)','File Data',
    'Content length','Request URI Query','Request Method',
    'Full Request URI','Request Version','Response','Ack No',
    'Ack No (RAW)','Checksum(TCP)','Connection Finish ',
    'Connection Reset','Connection Establish Request',
    'Connection Establish Ack','Source Port','Destination Port',
    'TCPFlags','Acknowledgment','TCP Segment Length','TCP Options',
    'TCP Payload','TCP Seq No','Src or Drc port','Stream index',
    'Time since previous frame','Query Name','DNS retransmission',
    'DNS query retransmission','DNS query retransmission in',
    'LG bit','IG bit','LG bit.1','Duplicate IP address configured',
    'Time to Live','Conversation completeness','Push','Content Type',
    'This is an ACK to the segment in frame','ECN-Echo','Mode',
    'Type','Type.1','Window','Echo data','Accept','Status Code',
    'Transaction ID','Handshake Type','Flags','Packet Type',
    'MSS Value','Message type','Timestamp value','TSecr',
    'TCP Option - SACK permitted','Response time','No response seen',
    'Kind','Duplicate ACK #',
    'This frame is a (suspected) retransmission',
    'Previous segment(s) not captured (common at capture start)',
    'FTP Data','Bytes in flight','Request command','Request command.1',
    'BER Error: length is not valid','Length.1','Flags.1',
    'Packet Length (encrypted)','Direction',
    'TCP Option - Maximum segment size','Data','Checksum',
    'CDATA','Label','Attack_Category'
]

TOP_FEATURES = [
    'dur','Length','Source Port','Destination Port',
    'Time to Live','TCP Segment Length','TCP Payload',
    'Bytes in flight','Acknowledgment','TCP Seq No'
]

FEATURE_RANGES = {
    'dur'               :(0.0,    7.97,        0.05),
    'Length'            :(42.0,   5858.0,      112.0),
    'Source Port'       :(0.0,    65535.0,     49152.0),
    'Destination Port'  :(0.0,    65534.0,     80.0),
    'Time to Live'      :(0.0,    90.0,        64.0),
    'TCP Segment Length':(0.0,    5792.0,      40.0),
    'TCP Payload'       :(0.0,    16866.0,     1024.0),
    'Bytes in flight'   :(0.0,    505440.0,    512.0),
    'Acknowledgment'    :(0.0,    3.0,         1.0),
    'TCP Seq No'        :(0.0,    4294884607.0,1048576.0),
}

LABEL_MAP = {
    0:"ARPPoisoning", 1:"Backdoor",     2:"ICMPflood",
    3:"ICMPredirect", 4:"Normal",       5:"Password_crack",
    6:"Port_Scanning",7:"SQLInjection", 8:"SYN_FLOOD",
    9:"Smurf",       10:"UDP_flood",   11:"VUlnerability_Scan"
}

ATTACK_DESC = {
    "ARPPoisoning"    :"Pemalsuan tabel ARP untuk intersepsi trafik jaringan.",
    "Backdoor"        :"Akses tersembunyi ke sistem tanpa autentikasi.",
    "ICMPflood"       :"Banjir paket ICMP untuk melumpuhkan target (DoS).",
    "ICMPredirect"    :"Manipulasi routing via paket ICMP Redirect.",
    "Normal"          :"Trafik jaringan normal, tidak terdeteksi ancaman.",
    "Password_crack"  :"Percobaan brute-force untuk menebak password.",
    "Port_Scanning"   :"Pemindaian port untuk menemukan layanan yang terbuka.",
    "SQLInjection"    :"Injeksi perintah SQL untuk manipulasi database.",
    "SYN_FLOOD"       :"Banjir paket SYN untuk menghabiskan sumber daya server.",
    "Smurf"           :"Serangan amplifikasi ICMP menggunakan broadcast.",
    "UDP_flood"       :"Banjir paket UDP untuk melumpuhkan target.",
    "VUlnerability_Scan":"Pemindaian sistematis untuk mencari celah keamanan.",
}

def random_sample():
    return {f: round(random.uniform(FEATURE_RANGES[f][0], FEATURE_RANGES[f][1]), 4) for f in TOP_FEATURES}

def normal_sample():
    return {'dur':0.05,'Length':74.0,'Source Port':49152.0,'Destination Port':80.0,
            'Time to Live':64.0,'TCP Segment Length':20.0,'TCP Payload':512.0,
            'Bytes in flight':512.0,'Acknowledgment':1.0,'TCP Seq No':1048576.0}

def attack_sample():
    return {'dur':0.0,'Length':42.0,'Source Port':12345.0,'Destination Port':80.0,
            'Time to Live':45.0,'TCP Segment Length':0.0,'TCP Payload':0.0,
            'Bytes in flight':0.0,'Acknowledgment':0.0,'TCP Seq No':99999999.0}

# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("pipeline_terbaik.pkl"), None
    except FileNotFoundError:
        return None, "File 'pipeline_terbaik.pkl' tidak ditemukan. Jalankan mlflow.py terlebih dahulu."
    except Exception as e:
        return None, str(e)

model, load_error = load_pipeline()

# ============================================================
# SESSION STATE
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
    ca, cb = st.columns(2)
    with ca:
        st.metric("ACCURACY",  "90.03%")
        st.metric("PRECISION", "87.45%")
    with cb:
        st.metric("F1 SCORE",  "86.71%")
        st.metric("RECALL",    "86.71%")
    st.markdown("---")
    st.markdown("`// PIPELINE`")
    st.caption("① StandardScaler")
    st.caption("② SelectFromModel (max=15)")
    st.caption("③ RandomForest (n=100, d=10)")
    st.markdown("---")
    st.caption("Upload limit: 1 GB (config.toml)")
    st.caption("Mid Semester — Machine Learning II")

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="border-left:3px solid #00ffb4;padding:1rem 1.5rem;margin-bottom:1rem;
background:rgba(0,255,180,0.025);border-radius:0 6px 6px 0;animation:fadeInUp 0.5s ease both;">
<div style="font-family:'Share Tech Mono',monospace;font-size:1.7rem;color:#00ffb4;
text-shadow:0 0 18px rgba(0,255,180,0.38);letter-spacing:0.1em;">
🔐 IoT VULNERABILITY DETECTION</div>
<div style="font-family:'Share Tech Mono',monospace;font-size:0.74rem;
color:rgba(122,255,218,0.42);letter-spacing:0.14em;margin-top:0.3rem;">
PIPELINE :: StandardScaler → SelectFromModel → RandomForestClassifier · 87 FEATURES · 12 CLASSES · UPLOAD MAX 1 GB
</div></div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠ SYSTEM ERROR :: {load_error}")
    st.stop()
else:
    st.success("✓ PIPELINE LOADED :: pipeline_terbaik.pkl — Scaler → FeatureSelection → Model")

st.divider()

# ============================================================
# 3 INFO CARDS
# ============================================================

st.markdown("### `// SYSTEM INFO`")
st.caption("Klik kartu untuk melihat detail.")

c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("📊 BALANCED DATASET", expanded=False):
        st.markdown("**IoT Vulnerability Dataset**")
        st.markdown("- Sumber: `Mendeley Data`")
        st.markdown("- Jumlah fitur: `87 kolom`")
        st.markdown("- Target: `Attack_sub_category`")
        st.markdown("- Kelas: `12 kelas`")
        st.markdown("- Split: `80% train / 20% test`")
        st.markdown("---")
        for v in LABEL_MAP.values():
            icon = "🟢" if v == "Normal" else "🔴"
            st.markdown(f"{icon} `{v}` — {ATTACK_DESC[v]}")

with c2:
    with st.expander("🌲 RANDOM FOREST", expanded=False):
        st.markdown("**Random Forest Classifier**")
        st.markdown("- Ensemble berbasis Decision Tree")
        st.markdown("- Voting mayoritas → tahan overfitting")
        st.markdown("---")
        st.markdown("**Best Params (GridSearchCV):**")
        st.code("n_estimators      = 100\nmax_depth         = 10\nmin_samples_split = 2\nmin_samples_leaf  = 1\nrandom_state      = 42", language="python")
        st.markdown("- CV: `Stratified K-Fold (5 Fold)`")
        st.markdown("- Best CV Accuracy: `99.09%`")

with c3:
    with st.expander("⚙️ EMBEDDED FEATURE SELECTION", expanded=False):
        st.markdown("**SelectFromModel (Embedded)**")
        st.markdown("- `87 fitur → 15 fitur terpilih`")
        st.markdown("- Feature importance dari Random Forest")
        st.markdown("- Di dalam Pipeline → `No Data Leakage`")
        st.markdown("---")
        st.code("Filter  → SelectKBest\nWrapper → RFE\nEmbedded→ SelectFromModel ✓", language="text")
        st.markdown("---")
        st.code("Scaler → SelectFromModel → RF", language="text")

st.divider()

ic1, ic2 = st.columns(2)
with ic1:
    with st.expander("🎯 KELAS YANG DIDETEKSI (12 KELAS)", expanded=False):
        for v in LABEL_MAP.values():
            icon = "🟢" if v == "Normal" else "🔴"
            st.markdown(f"{icon} **`{v}`** — {ATTACK_DESC[v]}")

with ic2:
    with st.expander("📖 CARA PENGGUNAAN", expanded=False):
        st.markdown("""
**Tab Input Manual:**
- Klik `⚡ RANDOM`, `🟢 NORMAL`, atau `🔴 ATTACK` untuk nilai contoh
- Edit nilai jika perlu, lalu tekan `▶ ANALYZE PACKET`
- Hasil prediksi + probabilitas + deskripsi ancaman muncul otomatis

**Tab Upload CSV:**
- Upload CSV 87 kolom (tanpa `Attack_sub_category`)
- Maksimal: **1 GB** (diatur via `.streamlit/config.toml`)
- Kolom kurang otomatis diisi 0
- Tekan `▶ BATCH ANALYZE` → lihat ringkasan lengkap → export CSV

> Pipeline bekerja utuh — tidak ada preprocessing manual.
        """)

st.divider()

# ============================================================
# TAB PREDIKSI
# ============================================================

tab1, tab2 = st.tabs(["  ▶  INPUT MANUAL  ", "  ▶  UPLOAD CSV  "])

# ----------------------------------------------------------
# TAB 1 — INPUT MANUAL
# ----------------------------------------------------------
with tab1:
    st.markdown("### `// PACKET ANALYZER`")
    st.caption("Gunakan tombol sample atau isi manual. Semua 87 kolom dikirim ke Pipeline.")

    b1, b2, b3, _ = st.columns([1,1,1,3])
    with b1:
        if st.button("⚡ RANDOM", key="btn_random"):
            st.session_state.input_vals = random_sample(); st.rerun()
    with b2:
        if st.button("🟢 NORMAL", key="btn_normal"):
            st.session_state.input_vals = normal_sample(); st.rerun()
    with b3:
        if st.button("🔴 ATTACK", key="btn_attack"):
            st.session_state.input_vals = attack_sample(); st.rerun()

    st.markdown("")
    input_values = {col: 0.0 for col in ALL_FEATURES}
    col_a, col_b = st.columns(2)
    for i, fname in enumerate(TOP_FEATURES):
        default_val = float(st.session_state.input_vals.get(fname, 0.0))
        col = col_a if i % 2 == 0 else col_b
        with col:
            input_values[fname] = st.number_input(
                label=fname, value=default_val, format="%.4f", key=f"fi_{fname}"
            )

    st.markdown("")
    if st.button("▶  ANALYZE PACKET", key="btn_manual", use_container_width=True):
        try:
            data_input  = pd.DataFrame([input_values])[ALL_FEATURES]
            hasil_raw   = model.predict(data_input)[0]
            hasil_label = LABEL_MAP.get(int(hasil_raw), str(hasil_raw))
            deskripsi   = ATTACK_DESC.get(hasil_label, "-")

            if hasil_label == "Normal":
                st.success(f"🟢 RESULT :: {hasil_label} — {deskripsi}")
            else:
                st.error(f"🔴 THREAT DETECTED :: {hasil_label} — {deskripsi}")

            if hasattr(model, "predict_proba"):
                proba    = model.predict_proba(data_input)[0]
                classes  = [LABEL_MAP.get(int(c), str(c)) for c in model.classes_]
                proba_df = pd.DataFrame({
                    "Kelas"        : classes,
                    "Probabilitas" : proba.round(4),
                    "Persentase"   : [f"{p*100:.2f}%" for p in proba],
                    "Deskripsi"    : [ATTACK_DESC.get(c,"-") for c in classes],
                }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

                with st.expander("📊 PROBABILITY DISTRIBUTION", expanded=True):
                    st.dataframe(proba_df, use_container_width=True)

        except Exception as e:
            st.error(f"⚠ ERROR :: {e}")
            st.caption("Pastikan nama kolom sesuai fitur training.")

# ----------------------------------------------------------
# TAB 2 — UPLOAD CSV
# ----------------------------------------------------------
with tab2:
    st.markdown("### `// BATCH ANALYZER`")
    st.caption("Upload CSV 87 kolom (tanpa `Attack_sub_category`). Max 1 GB.")

    uploaded_file = st.file_uploader("SELECT FILE", type=["csv"], key="upload_csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown(f"**FILE LOADED** :: `{len(data):,}` rows · `{len(data.columns)}` columns")
            st.dataframe(data.head(), use_container_width=True)

            missing_cols = [c for c in ALL_FEATURES if c not in data.columns]
            extra_cols   = [c for c in data.columns  if c not in ALL_FEATURES]
            if missing_cols:
                st.warning(f"⚠ {len(missing_cols)} kolom hilang → diisi 0 :: {missing_cols[:3]}{'...' if len(missing_cols)>3 else ''}")
            if extra_cols:
                st.info(f"ℹ {len(extra_cols)} kolom tidak dikenal → diabaikan")

            if st.button("▶  BATCH ANALYZE", key="btn_csv", use_container_width=True):
                with st.spinner("Processing :: Scaler → SelectFromModel → Model..."):

                    # Siapkan data — isi kolom kurang, urutkan
                    for col in ALL_FEATURES:
                        if col not in data.columns:
                            data[col] = 0.0
                    data_ordered = data[ALL_FEATURES].copy()

                    # Prediksi
                    prediksi_raw = model.predict(data_ordered)

                    # Ubah angka → nama kelas
                    prediksi_label = [LABEL_MAP.get(int(p), str(p)) for p in prediksi_raw]

                    # Probabilitas
                    proba_batch = None
                    if hasattr(model, "predict_proba"):
                        proba_batch = model.predict_proba(data_ordered)

                    # Bangun hasil
                    data_hasil = data_ordered.copy()
                    data_hasil["Prediction"]       = prediksi_label
                    data_hasil["Threat_Type"]      = data_hasil["Prediction"].apply(
                        lambda x: "Normal" if x == "Normal" else "Attack"
                    )
                    data_hasil["Attack_Description"] = data_hasil["Prediction"].map(ATTACK_DESC)

                    if proba_batch is not None:
                        for i, c in enumerate(model.classes_):
                            col_name = LABEL_MAP.get(int(c), str(c))
                            data_hasil[f"Prob_{col_name}"] = proba_batch[:, i].round(4)
                        # Confidence = probabilitas kelas terprediksi
                        data_hasil["Confidence"] = [
                            round(float(proba_batch[i, int(prediksi_raw[i])]) * 100, 2)
                            for i in range(len(prediksi_raw))
                        ]
                        data_hasil["Confidence"] = data_hasil["Confidence"].astype(str) + "%"

                # ── Ringkasan statistik ──
                total       = len(data_hasil)
                n_normal    = (data_hasil["Threat_Type"] == "Normal").sum()
                n_attack    = (data_hasil["Threat_Type"] == "Attack").sum()
                pct_normal  = round(n_normal / total * 100, 2)
                pct_attack  = round(n_attack / total * 100, 2)

                st.success(f"✓ DONE :: {total:,} packets analyzed.")

                # ── Metric cards ringkasan ──
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Packets",  f"{total:,}")
                m2.metric("Normal Traffic", f"{n_normal:,}", f"{pct_normal}%")
                m3.metric("Threats Detected", f"{n_attack:,}", f"{pct_attack}%")
                if proba_batch is not None:
                    avg_conf = round(float(np.mean(proba_batch.max(axis=1))) * 100, 2)
                    m4.metric("Avg Confidence", f"{avg_conf}%")

                st.markdown("---")

                # ── Preview hasil ──
                st.markdown("**RESULT PREVIEW (top 10):**")
                cols_show = ["Prediction","Threat_Type","Attack_Description","Confidence"] if proba_batch is not None else ["Prediction","Threat_Type","Attack_Description"]
                st.dataframe(data_hasil[cols_show].head(10), use_container_width=True)

                # ── Distribusi kelas ──
                with st.expander("📊 PREDICTION DISTRIBUTION — LENGKAP", expanded=True):
                    dist = data_hasil["Prediction"].value_counts().reset_index()
                    dist.columns = ["Kelas","Jumlah"]
                    dist["Persentase"]  = (dist["Jumlah"] / total * 100).round(2).astype(str) + "%"
                    dist["Threat_Type"] = dist["Kelas"].apply(lambda x: "🟢 Normal" if x == "Normal" else "🔴 Attack")
                    dist["Deskripsi"]   = dist["Kelas"].map(ATTACK_DESC)
                    dist = dist.sort_values("Jumlah", ascending=False).reset_index(drop=True)
                    st.dataframe(dist, use_container_width=True)

                # ── Ringkasan per threat type ──
                with st.expander("📋 RINGKASAN THREAT TYPE", expanded=True):
                    threat_sum = data_hasil["Threat_Type"].value_counts().reset_index()
                    threat_sum.columns = ["Threat_Type","Jumlah"]
                    threat_sum["Persentase"] = (threat_sum["Jumlah"] / total * 100).round(2).astype(str) + "%"
                    st.dataframe(threat_sum, use_container_width=True)

                # ── Download ──
                st.markdown("---")
                csv_out = data_hasil.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇  EXPORT FULL RESULTS (.csv)",
                    data=csv_out, file_name="hasil_prediksi_lengkap.csv",
                    mime="text/csv", key="btn_download"
                )

        except Exception as e:
            st.error(f"⚠ ERROR :: {e}")
            st.caption("Pastikan file CSV valid dan kolomnya sesuai fitur training.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
color:rgba(122,255,218,0.22);text-align:center;letter-spacing:0.14em;padding:0.5rem 0;">
MID SEMESTER · MACHINE LEARNING II · IoT VULNERABILITY DETECTION ·
StandardScaler → SelectFromModel → RandomForestClassifier · NO DATA LEAKAGE · UPLOAD MAX 1 GB
</div>
""", unsafe_allow_html=True)