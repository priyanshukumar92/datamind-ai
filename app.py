"""
AI Data Analyst Agent - Elegant UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from groq import Groq
from core.agent import DataAnalystAgent, ask_about_image
from core.ingestion import detect_and_load

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #080C14;
    --bg-secondary: #0D1421;
    --bg-card: #111827;
    --bg-glass: rgba(17, 24, 39, 0.7);
    --accent-gold: #D4A847;
    --accent-gold-dim: rgba(212, 168, 71, 0.15);
    --accent-blue: #3B82F6;
    --accent-teal: #14B8A6;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-dim: #475569;
    --border: rgba(212, 168, 71, 0.12);
    --border-bright: rgba(212, 168, 71, 0.35);
    --shadow: 0 25px 60px rgba(0,0,0,0.5);
    --radius: 16px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Animated background grid */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(212,168,71,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(212,168,71,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] {
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-gold), var(--accent-teal), var(--accent-blue));
}

[data-testid="stSidebarContent"] {
    padding: 1.5rem 1.2rem !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Main Title Area ── */
.hero-section {
    text-align: center;
    padding: 3rem 2rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: var(--accent-gold-dim);
    border: 1px solid var(--border-bright);
    color: var(--accent-gold);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 1rem;
    background: linear-gradient(135deg, #F1F5F9 0%, var(--accent-gold) 50%, #F1F5F9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 300;
    font-style: italic;
    margin: 0 auto;
    max-width: 500px;
    line-height: 1.6;
}

.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
    margin: 2rem auto;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.sidebar-brand-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent-gold), #F59E0B);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.sidebar-brand-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.sidebar-brand-sub {
    font-size: 0.7rem;
    color: var(--text-dim);
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-gold) !important;
    background: var(--accent-gold-dim) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--text-secondary) !important;
}

/* ── Stat Cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 1rem 0;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
    transition: border-color 0.3s;
}

.stat-card:hover { border-color: var(--border-bright); }

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-gold);
}

.stat-label {
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* ── Success / Info / Error Alerts ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-left: 3px solid var(--accent-gold) !important;
    border-radius: 0 12px 12px 0 !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.08)) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
    border-radius: 10px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, var(--accent-gold), #F59E0B) !important;
    border-radius: 10px !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.3s !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 3px var(--accent-gold-dim) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-dim) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: var(--accent-gold-dim) !important;
    color: var(--accent-gold) !important;
    border-color: var(--border-bright) !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}

.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Code Blocks ── */
.stCodeBlock {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent-gold) !important;
    font-size: 1.4rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Success box ── */
.stSuccess {
    background: rgba(20, 184, 166, 0.1) !important;
    border-left: 3px solid var(--accent-teal) !important;
    border-radius: 0 12px 12px 0 !important;
    color: var(--text-primary) !important;
}

/* ── Welcome Cards ── */
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin: 2rem 0;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-gold), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.feature-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-3px);
    box-shadow: var(--shadow);
}

.feature-card:hover::before { opacity: 1; }

.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.8rem;
    display: block;
}

.feature-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}

.feature-desc {
    font-size: 0.83rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Example Pills ── */
.pill-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 1.5rem 0;
}

.pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.4rem 1rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'DM Sans', sans-serif;
}

.pill:hover {
    background: var(--accent-gold-dim);
    border-color: var(--border-bright);
    color: var(--accent-gold);
}

/* ── Section Header ── */
.chat-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}

.chat-section-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.chat-section-icon.data { background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(59,130,246,0.1)); border: 1px solid rgba(59,130,246,0.3); }
.chat-section-icon.image { background: linear-gradient(135deg, rgba(20,184,166,0.2), rgba(20,184,166,0.1)); border: 1px solid rgba(20,184,166,0.3); }
.chat-section-icon.audio { background: linear-gradient(135deg, rgba(212,168,71,0.2), rgba(212,168,71,0.1)); border: 1px solid rgba(212,168,71,0.3); }

.chat-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
}

.chat-section-sub {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 2px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-gold); }

/* ── Plotly Charts ── */
.js-plotly-plot {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent-gold) !important;
}

/* ── Text area ── */
textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "image_file" not in st.session_state:
    st.session_state.image_file = None
if "audio_text" not in st.session_state:
    st.session_state.audio_text = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">✦</div>
        <div>
            <div class="sidebar-brand-text">DataMind AI</div>
            <div class="sidebar-brand-sub">Analyst Agent v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="section-label">Upload Data</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["csv", "xlsx", "xls", "png", "jpg", "jpeg", "pdf", "mp3", "wav"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner("Processing..."):
            data, file_type = detect_and_load(uploaded_file, uploaded_file.name)
            st.session_state.file_type = file_type

            if file_type in ["csv", "excel"]:
                st.session_state.df = data
                st.session_state.agent = DataAnalystAgent(data)
                st.session_state.image_file = None
                st.session_state.audio_text = None
                st.session_state.chat_history = []
                st.success(f"✦ {file_type.upper()} loaded — {data.shape[0]:,} rows")

            elif file_type == "image":
                st.session_state.df = None
                st.session_state.agent = None
                st.session_state.image_file = uploaded_file
                st.session_state.audio_text = None
                st.session_state.chat_history = []
                st.success("✦ Image ready for analysis")
                st.image(uploaded_file, use_container_width=True)

            elif file_type == "audio":
                st.session_state.df = None
                st.session_state.agent = None
                st.session_state.image_file = None
                st.session_state.audio_text = data
                st.session_state.chat_history = []
                st.success("✦ Audio transcribed")
                with st.expander("View transcript"):
                    st.write(data)

            elif file_type == "pdf":
                st.session_state.df = None
                st.session_state.agent = None
                st.session_state.image_file = None
                st.session_state.audio_text = None
                st.session_state.chat_history = []
                st.info("✦ PDF extracted")
                with st.expander("View content"):
                    st.write(data)
            else:
                st.error("Unsupported file type")

    # Dataset stats
    if st.session_state.df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)

        df = st.session_state.df
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{df.shape[0]:,}</div>
                <div class="stat-label">Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df.shape[1]}</div>
                <div class="stat-label">Cols</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df.isnull().sum().sum()}</div>
                <div class="stat-label">Nulls</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preview data"):
            st.dataframe(df.head(8), use_container_width=True, height=200)

    # Supported formats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Supported Formats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:6px;">
        <div style="display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#64748B;">
            <span style="color:#3B82F6;">▸</span> CSV, Excel — Data Analysis
        </div>
        <div style="display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#64748B;">
            <span style="color:#14B8A6;">▸</span> PNG, JPG — Vision Analysis
        </div>
        <div style="display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#64748B;">
            <span style="color:#D4A847;">▸</span> MP3, WAV — Audio Transcription
        </div>
        <div style="display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#64748B;">
            <span style="color:#A855F7;">▸</span> PDF — Document Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Reset
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Reset Session"):
        st.session_state.chat_history = []
        st.session_state.image_file = None
        st.session_state.audio_text = None
        st.session_state.df = None
        st.session_state.agent = None
        if st.session_state.get("agent"):
            st.session_state.agent.reset()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── CASE 1: Welcome Screen ────────────────────────────────────────────────────
if (st.session_state.agent is None and
        st.session_state.image_file is None and
        st.session_state.audio_text is None):

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">✦ Powered by LLaMA 3.3 · Groq Inference</div>
        <h1 class="hero-title">Your AI Data Analyst</h1>
        <p class="hero-subtitle">Ask questions in plain English. Get instant analysis, charts, and insights — no code required.</p>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="welcome-grid">
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Structured Data</div>
            <div class="feature-desc">Upload CSV or Excel files and ask complex analytical questions in plain English.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🖼️</span>
            <div class="feature-title">Image Analysis</div>
            <div class="feature-desc">Upload charts, graphs, or data screenshots — the AI will interpret and explain them.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🎵</span>
            <div class="feature-title">Audio Intelligence</div>
            <div class="feature-desc">Upload audio recordings — Whisper transcribes them, then ask questions about the content.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📄</span>
            <div class="feature-title">Document Analysis</div>
            <div class="feature-desc">Extract text from PDFs and analyze reports, contracts, or research papers instantly.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example pills
    st.markdown('<div class="section-label" style="margin-top:2rem;">Try asking</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pill-grid">
        <div class="pill">Which industry has highest revenue?</div>
        <div class="pill">Show top 10 by sales as bar chart</div>
        <div class="pill">Are there any missing values?</div>
        <div class="pill">What's the trend over time?</div>
        <div class="pill">Summarize this dataset</div>
        <div class="pill">Show correlation heatmap</div>
        <div class="pill">Filter rows where value > 1000</div>
        <div class="pill">What does this image show?</div>
    </div>
    """, unsafe_allow_html=True)


# ── CASE 2: Image Chat ────────────────────────────────────────────────────────
elif st.session_state.image_file is not None:

    st.markdown("""
    <div class="chat-section-header">
        <div class="chat-section-icon image">🖼️</div>
        <div>
            <div class="chat-section-title">Image Analysis</div>
            <div class="chat-section-sub">Vision AI is ready — ask anything about your image</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("What would you like to know about this image?")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                st.session_state.image_file.seek(0)
                response_text = ask_about_image(st.session_state.image_file, user_input)
            st.write(response_text)

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})


# ── CASE 3: Audio Chat ────────────────────────────────────────────────────────
elif st.session_state.audio_text is not None:

    st.markdown("""
    <div class="chat-section-header">
        <div class="chat-section-icon audio">🎵</div>
        <div>
            <div class="chat-section-title">Audio Analysis</div>
            <div class="chat-section-sub">Transcription complete — ask anything about the content</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📝 View transcribed text"):
        st.write(st.session_state.audio_text)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask about the audio content...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "your-key-here"))
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": "You are an expert analyst. Answer questions based on the provided audio transcript clearly and concisely."},
                        {"role": "user", "content": f"Audio transcript:\n{st.session_state.audio_text}\n\nQuestion: {user_input}"}
                    ]
                )
                answer = response.choices[0].message.content
            st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ── CASE 4: Data Analysis Chat ────────────────────────────────────────────────
else:

    st.markdown("""
    <div class="chat-section-header">
        <div class="chat-section-icon data">📊</div>
        <div>
            <div class="chat-section-title">Data Analysis</div>
            <div class="chat-section-sub">Dataset loaded — ask anything in plain English</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart theme
    chart_theme = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(17,24,39,0)",
        plot_bgcolor="rgba(17,24,39,0)",
        font=dict(family="DM Sans", color="#94A3B8"),
        colorway=["#D4A847", "#3B82F6", "#14B8A6", "#A855F7", "#F43F5E", "#10B981"]
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                resp = message["content"]
                if resp.get("explanation"):
                    st.write(resp["explanation"])
                if resp.get("result") is not None:
                    result = resp["result"]
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                    elif isinstance(result, pd.Series):
                        st.dataframe(result.reset_index(), use_container_width=True)
                    else:
                        st.success(f"**Answer:** {result}")
                if resp.get("output"):
                    st.text(resp["output"])
                if resp.get("chart_data"):
                    chart = resp["chart_data"]
                    ctype = chart.get("type", "bar")
                    title = chart.get("title", "")
                    x, y = chart.get("x", []), chart.get("y", [])
                    fig = None
                    if ctype == "bar":
                        fig = px.bar(x=x, y=y, title=title)
                    elif ctype == "line":
                        fig = px.line(x=x, y=y, title=title)
                    elif ctype == "pie":
                        fig = px.pie(names=x, values=y, title=title)
                    elif ctype == "scatter":
                        fig = px.scatter(x=x, y=y, title=title)
                    if fig:
                        fig.update_layout(**chart_theme)
                        st.plotly_chart(fig, use_container_width=True)
                if resp.get("code"):
                    with st.expander("⟨/⟩ View generated code"):
                        st.code(resp["code"], language="python")
                if resp.get("error"):
                    st.error(f"Error: {resp['error']}")

    user_input = st.chat_input("Ask anything about your data...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.agent.ask(user_input)

            if response.get("explanation"):
                st.write(response["explanation"])
            if response.get("result") is not None:
                result = response["result"]
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                elif isinstance(result, pd.Series):
                    st.dataframe(result.reset_index(), use_container_width=True)
                else:
                    st.success(f"**Answer:** {result}")
            if response.get("output"):
                st.text(response["output"])
            if response.get("chart_data"):
                chart = response["chart_data"]
                ctype = chart.get("type", "bar")
                title = chart.get("title", "")
                x, y = chart.get("x", []), chart.get("y", [])
                fig = None
                if ctype == "bar":
                    fig = px.bar(x=x, y=y, title=title)
                elif ctype == "line":
                    fig = px.line(x=x, y=y, title=title)
                elif ctype == "pie":
                    fig = px.pie(names=x, values=y, title=title)
                elif ctype == "scatter":
                    fig = px.scatter(x=x, y=y, title=title)
                if fig:
                    fig.update_layout(**chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
            if response.get("code"):
                with st.expander("⟨/⟩ View generated code"):
                    st.code(response["code"], language="python")
            if response.get("error"):
                st.error(f"Error: {response['error']}")

        st.session_state.chat_history.append({"role": "assistant", "content": response})