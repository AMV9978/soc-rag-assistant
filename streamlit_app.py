import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

KNOWLEDGE_DIR = Path("data/knowledge")
INDEX_DIR = Path("data/faiss_indexes")
PACKS_DIR = KNOWLEDGE_DIR / "packs"
PACK_KEY_FILE = INDEX_DIR / "pack_key.txt"

# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Base ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Syne', sans-serif !important;
        background-color: #06090e !important;
        color: #e2e8f0 !important;
    }

    .stApp {
        background: #06090e !important;
        background-image:
            radial-gradient(ellipse at 15% 0%, rgba(0,255,136,0.05) 0%, transparent 55%),
            radial-gradient(ellipse at 85% 100%, rgba(56,139,255,0.05) 0%, transparent 55%) !important;
    }

    /* ── Main content padding ── */
    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0a0e14 !important;
        border-right: 1px solid #161e2d !important;
        min-width: 260px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1.2rem !important;
    }

    /* ── Sidebar heading ── */
    .cp-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        border-bottom: 1px solid #161e2d;
        margin-bottom: 1.2rem;
    }
    .cp-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #f0f6fc;
    }
    .cp-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: #00ff88;
        background: rgba(0,255,136,0.08);
        border: 1px solid rgba(0,255,136,0.25);
        padding: 0.15rem 0.45rem;
        border-radius: 3px;
    }

    /* ── Sidebar section labels ── */
    .sb-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #3d5166;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #161e2d;
    }

    /* ── Expander reset ── */
    .streamlit-expanderHeader {
        background: transparent !important;
        border: 1px solid #161e2d !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        color: #8b9ab0 !important;
        padding: 0.5rem 0.8rem !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: #00ff8840 !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #161e2d !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        background: #080c12 !important;
        padding: 0.8rem !important;
    }

    /* ── Checkbox — kill the neon green label highlight ── */
    .stCheckbox {
        margin: 0.15rem 0 !important;
    }
    .stCheckbox label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #c9d8e8 !important;
        background: transparent !important;
        padding: 0.2rem 0 !important;
    }
    .stCheckbox label p {
        color: #c9d8e8 !important;
        font-weight: 500 !important;
    }
    .stCheckbox label:hover p {
        color: #ffffff !important;
    }
    /* Checkbox box itself */
    .stCheckbox [data-baseweb="checkbox"] > div {
        border-color: #3d5166 !important;
        background: #0a0e14 !important;
    }
    .stCheckbox [aria-checked="true"] > div {
        background: #00ff88 !important;
        border-color: #00ff88 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #3d5166 !important;
    }
    .stSelectbox > div > div {
        background: #0c1118 !important;
        border: 1px solid #1e2d40 !important;
        border-radius: 5px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #00ff8860 !important;
        box-shadow: 0 0 0 1px #00ff8830 !important;
    }

    /* ── Text area ── */
    .stTextArea label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #3d5166 !important;
    }
    .stTextArea textarea {
        background: #0c1118 !important;
        border: 1px solid #1e2d40 !important;
        border-radius: 5px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
        caret-color: #00ff88 !important;
    }
    .stTextArea textarea:focus {
        border-color: #00ff8860 !important;
        box-shadow: 0 0 0 1px #00ff8830 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: #00ff88 !important;
        color: #06090e !important;
        border: none !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border-radius: 5px !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #00e07a !important;
        box-shadow: 0 0 24px rgba(0,255,136,0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Secondary button ── */
    .stButton > button:not([kind="primary"]) {
        background: #0c1118 !important;
        border: 1px solid #1e2d40 !important;
        color: #8b9ab0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        border-radius: 5px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #00ff8850 !important;
        color: #e2e8f0 !important;
        background: #0f1620 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 0 !important;
        border-bottom: 1px solid #161e2d !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: #3d5166 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        padding: 0.55rem 1.2rem !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #8b9ab0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00ff88 !important;
        border-bottom: 2px solid #00ff88 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1rem 0 0 0 !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid #1e2d40 !important;
        color: #3d5166 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
    }
    .stDownloadButton > button:hover {
        border-color: #388bff60 !important;
        color: #58a6ff !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #080c12 !important;
        border: 1px dashed #1e2d40 !important;
        border-radius: 5px !important;
    }
    [data-testid="stFileUploader"] label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        color: #3d5166 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #161e2d !important;
        margin: 1.4rem 0 !important;
    }

    /* ── Alerts ── */
    .stAlert {
        background: #0a0e14 !important;
        border-radius: 5px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #00ff88 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #06090e; }
    ::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #00ff8860; }

    /* ── App header ── */
    .soc-header {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.3rem;
    }
    .soc-accent-bar {
        width: 4px;
        height: 52px;
        background: linear-gradient(180deg, #00ff88 0%, #388bff 100%);
        border-radius: 2px;
        flex-shrink: 0;
        margin-top: 4px;
    }
    .soc-title-block h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        color: #f0f6fc;
        margin: 0 0 0.2rem 0;
        line-height: 1;
    }
    .soc-title-block h1 em {
        font-style: normal;
        color: #00ff88;
    }
    .soc-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 400;
        color: #3d5166;
        letter-spacing: 0.08em;
        line-height: 1.4;
    }

    /* ── Metric row ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1px;
        background: #161e2d;
        border: 1px solid #161e2d;
        border-radius: 8px;
        overflow: hidden;
        margin: 1.4rem 0 1.6rem 0;
    }
    .metric-card {
        background: #0a0e14;
        padding: 1.1rem 1.4rem;
        position: relative;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }
    .metric-card:nth-child(1)::before { background: linear-gradient(90deg, #00ff88, transparent); }
    .metric-card:nth-child(2)::before { background: linear-gradient(90deg, #388bff, transparent); }
    .metric-card:nth-child(3)::before { background: linear-gradient(90deg, #f78166, transparent); }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #3d5166;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #00ff88;
        line-height: 1;
    }
    .metric-value.blue    { color: #58a6ff; }
    .metric-value.orange  { color: #f78166; font-size: 1.1rem; font-weight: 700; }
    .metric-value.green-sm { color: #00ff88; font-size: 1.1rem; font-weight: 700; }

    /* ── Panel labels ── */
    .panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #3d5166;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .panel-label .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .panel-label .dot.green { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
    .panel-label .dot.blue  { background: #58a6ff; box-shadow: 0 0 6px #58a6ff; }
    .panel-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #161e2d;
    }

    /* ── Answer content ── */
    .answer-body {
        font-family: 'Syne', sans-serif;
        font-size: 0.9rem;
        font-weight: 400;
        line-height: 1.75;
        color: #c9d8e8;
    }
    .answer-body strong, .answer-body b {
        color: #f0f6fc;
        font-weight: 700;
    }

    /* ── Evidence card ── */
    .ev-card {
        background: #080c12;
        border: 1px solid #161e2d;
        border-left: 3px solid #388bff;
        border-radius: 5px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .ev-source {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #388bff;
        margin-bottom: 0.5rem;
    }
    .ev-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 400;
        color: #6b7f96;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ── History ── */
    .hist-user {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #58a6ff;
        padding: 0.35rem 0;
        border-bottom: 1px solid #0f151d;
        line-height: 1.5;
    }
    .hist-assistant {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 400;
        color: #6b7f96;
        padding: 0.35rem 0;
        border-bottom: 1px solid #0f151d;
        line-height: 1.5;
    }

    /* ── Empty state ── */
    .empty-state {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        color: #1e2d40;
        padding: 2rem 0;
        text-align: center;
        letter-spacing: 0.08em;
    }

    /* ── Active packs caption ── */
    .packs-active {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 600;
        color: #00cc70;
        margin-top: 0.6rem;
        letter-spacing: 0.05em;
        line-height: 1.5;
        word-break: break-word;
    }
    .packs-none {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 600;
        color: #f7816690;
        margin-top: 0.6rem;
        letter-spacing: 0.05em;
    }

    /* ── Sidebar footer ── */
    .sb-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem;
        font-weight: 400;
        color: #1e2d40;
        border-top: 1px solid #161e2d;
        padding-top: 0.8rem;
        margin-top: 2rem;
        line-height: 1.8;
    }
    .sb-footer strong {
        color: #3d5166;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Domain logic (unchanged) ───────────────────────────────────────────────────
def list_packs():
    if not PACKS_DIR.exists():
        return []
    return sorted([p.name for p in PACKS_DIR.iterdir() if p.is_dir()])

def count_knowledge_files(enabled_packs):
    if not PACKS_DIR.exists():
        return 0
    packs = enabled_packs if enabled_packs is not None else list_packs()
    total = 0
    for pack in packs:
        pack_path = PACKS_DIR / pack
        if pack_path.exists():
            total += sum(1 for fp in pack_path.rglob("*") if fp.suffix.lower() in [".md", ".txt"])
    return total

def get_index_status():
    if INDEX_DIR.exists():
        if (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists():
            return "ready"
        return "partial"
    return "none"

def make_pack_key(enabled_packs):
    if enabled_packs is None: return "ALL"
    if not enabled_packs: return "NONE"
    return ",".join(sorted(enabled_packs))

def save_pack_key(pack_key):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PACK_KEY_FILE.write_text(pack_key, encoding="utf-8")

def load_saved_pack_key():
    if PACK_KEY_FILE.exists():
        return PACK_KEY_FILE.read_text(encoding="utf-8").strip()
    return None

def load_knowledge_files(selected_packs=None):
    documents = []
    if not PACKS_DIR.exists():
        files = sorted(list(KNOWLEDGE_DIR.glob("*.md")) + list(KNOWLEDGE_DIR.glob("*.txt")))
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": fp.name}))
        return documents
    if selected_packs is None:
        selected_packs = list_packs()
    for pack in selected_packs:
        pack_path = PACKS_DIR / pack
        if not pack_path.exists():
            continue
        for fp in sorted(pack_path.rglob("*")):
            if fp.suffix.lower() in [".md", ".txt"]:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": str(fp.relative_to(PACKS_DIR))}))
    return documents

def build_vectorstore(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

def save_vectorstore(vectorstore):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

def load_vectorstore(embeddings):
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

PROMPT_LIBRARY = {
    "SOC Triage": [
        "Walk me through a SOC triage workflow for a phishing alert.",
        "Give me a ransomware incident runbook step-by-step.",
        "How do you investigate a possible account compromise?"
    ],
    "Splunk SPL": [
        "Write SPL to detect password spraying.",
        "Write SPL to detect DNS tunneling.",
        "Write SPL for suspicious PowerShell encoded commands.",
        "What does beaconing look like and how do you hunt it?"
    ],
    "AWS Security": [
        "What are the most important AWS logs and why?",
        "What CloudTrail events are high-risk for IAM compromise?",
        "How do you respond to an S3 bucket exposure?"
    ],
    "IAM / Identity": [
        "Explain IAM fundamentals in simple terms.",
        "What is OAuth consent abuse and how do you respond?",
        "What should an access review include?"
    ],
    "MITRE": [
        "Map phishing to MITRE and explain why.",
        "Map lateral movement and suggest detection signals.",
        "Map DNS tunneling to MITRE at a high level."
    ],
    "Fundamentals": [
        "What is cybersecurity? Explain like I am new.",
        "Explain the CIA triad with examples.",
        "Why is logging important in security?"
    ]
}

def answer_question(llm, vectorstore, question, history_text):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.55}
    )
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return [], "", "I don't have enough information from the documents."
    context_text = "\n\n".join([
        f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
        for d in retrieved_docs
    ])
    prompt = ChatPromptTemplate.from_template("""
You are a SOC analyst assistant. Use ONLY the context below to answer.
If the answer is not in the context, say: "I don't have enough information from the documents."

Chat History:
{history}

Context:
{context}

Question:
{question}

Return in this format:
1) Summary (1 sentence)
2) Key Indicators (3 bullets)
3) Triage Steps (3 bullets)
4) Recommended Actions (2 bullets)
""")
    response = llm.invoke(prompt.format(
        context=context_text,
        question=question,
        history=history_text
    ))
    return retrieved_docs, context_text, response.content

def evidence_cards(docs):
    if not docs:
        st.markdown(
            '<div class="empty-state">NO EVIDENCE RETRIEVED YET</div>',
            unsafe_allow_html=True
        )
        return
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        excerpt = d.page_content.strip()[:600]
        st.markdown(f"""
        <div class="ev-card">
            <div class="ev-source">▸ Source {i} &nbsp;/&nbsp; {src}</div>
            <div class="ev-text">{excerpt}</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            f"↓ Export Source {i}",
            data=d.page_content,
            file_name=f"evidence_{i}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_{i}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SOC RAG Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    for key, default in [
        ("chat_history", []),
        ("vectorstore", None),
        ("last_answer", ""),
        ("last_context", ""),
        ("last_docs", [])
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠ OPENAI_API_KEY not found — add it under Settings → Secrets.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="cp-header">
            <span class="cp-title">Control Panel</span>
            <span class="cp-badge">v1.0</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Knowledge Packs</div>', unsafe_allow_html=True)
        with st.expander("Manage Packs", expanded=True):
            pack_names = list_packs()
            if pack_names:
                c1, c2 = st.columns(2)
                if c1.button("Select All", use_container_width=True):
                    for name in pack_names:
                        st.session_state[f"pack_{name}"] = True
                if c2.button("Deselect All", use_container_width=True):
                    for name in pack_names:
                        st.session_state[f"pack_{name}"] = False
                enabled_packs = []
                for name in pack_names:
                    if f"pack_{name}" not in st.session_state:
                        st.session_state[f"pack_{name}"] = True
                    if st.checkbox(name, key=f"pack_{name}"):
                        enabled_packs.append(name)
                if enabled_packs:
                    st.markdown(
                        f'<div class="packs-active">● ACTIVE: {", ".join(enabled_packs)}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="packs-none">● NO PACKS SELECTED</div>',
                        unsafe_allow_html=True
                    )
            else:
                enabled_packs = None
                st.markdown(
                    '<div class="empty-state" style="padding:0.5rem 0;text-align:left;">No packs found.</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div class="sb-label">Upload Knowledge</div>', unsafe_allow_html=True)
        with st.expander("Add Files (.txt / .md)", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["txt", "md"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

        st.markdown('<div class="sb-label">Index Controls</div>', unsafe_allow_html=True)
        with st.expander("Build / Load Index", expanded=True):
            c1, c2 = st.columns(2)
            rebuild_clicked = c1.button("⟳ Rebuild", use_container_width=True)
            load_clicked    = c2.button("⚡ Load",    use_container_width=True)

        st.markdown("""
        <div class="sb-footer">
            <strong>Angelo Vasquez</strong><br>
            NYU M.S. Cybersecurity<br>
            SOC RAG Assistant · v1.0
        </div>
        """, unsafe_allow_html=True)

    # ── Load docs ───────────────────────────────────────────────────────────────
    docs = []
    if KNOWLEDGE_DIR.exists():
        docs.extend(load_knowledge_files(enabled_packs))
    if uploaded_files:
        for file in uploaded_files:
            text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": f"UPLOAD:{file.name}"}))

    enabled_count = len(enabled_packs) if enabled_packs is not None else len(list_packs())
    current_key   = make_pack_key(enabled_packs)
    saved_key     = load_saved_pack_key()
    index_present = (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()
    packs_changed = (saved_key != current_key)

    # ── Vectorstore — build/load BEFORE computing index_status ──────────────────
    if docs:
        if index_present and packs_changed:
            with st.spinner("Pack selection changed — rebuilding index…"):
                vs = build_vectorstore(docs, embeddings)
                save_vectorstore(vs)
                save_pack_key(current_key)
                st.session_state.vectorstore = vs
            st.success("Index rebuilt.")

        if rebuild_clicked:
            with st.spinner("Building FAISS index…"):
                vs = build_vectorstore(docs, embeddings)
                save_vectorstore(vs)
                save_pack_key(current_key)
                st.session_state.vectorstore = vs
            st.success("Index built and saved.")

        if load_clicked:
            if index_present and not packs_changed:
                st.session_state.vectorstore = load_vectorstore(embeddings)
                st.success("Saved index loaded.")
            else:
                st.error("Pack mismatch or no saved index — rebuild first.")

        if st.session_state.vectorstore is None:
            with st.spinner("Initializing knowledge base…"):
                if index_present and not packs_changed:
                    st.session_state.vectorstore = load_vectorstore(embeddings)
                else:
                    st.session_state.vectorstore = build_vectorstore(docs, embeddings)

    # ── Compute status AFTER build so it reflects reality ───────────────────────
    index_status  = get_index_status()
    status_label  = "ONLINE"   if index_status == "ready" else "BUILDING"
    status_class  = "green-sm" if index_status == "ready" else "orange"

    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="soc-header">
        <div class="soc-accent-bar"></div>
        <div class="soc-title-block">
            <h1>SOC RAG <em>ASSISTANT</em></h1>
            <div class="soc-caption">
                ENTERPRISE CYBERSECURITY INTELLIGENCE &nbsp;·&nbsp;
                RETRIEVAL-AUGMENTED GENERATION &nbsp;·&nbsp;
                NYU M.S. CYBERSECURITY
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Packs Active</div>
            <div class="metric-value">{enabled_count:02d}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Knowledge Files</div>
            <div class="metric-value blue">{count_knowledge_files(enabled_packs):02d}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Index Status</div>
            <div class="metric-value {status_class}">{status_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not docs:
        st.warning("No knowledge files found — check your packs directory.")
        st.stop()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Main columns ─────────────────────────────────────────────────────────────
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("""
        <div class="panel-label">
            <span class="dot green"></span> Query Interface
        </div>
        """, unsafe_allow_html=True)
        category = st.selectbox("Domain", list(PROMPT_LIBRARY.keys()))
        example  = st.selectbox("Example Queries", PROMPT_LIBRARY[category])
        question = st.text_area("Your Question", value=example, height=120)
        ask      = st.button("▶  EXECUTE QUERY", use_container_width=True, type="primary")

    with right:
        st.markdown("""
        <div class="panel-label">
            <span class="dot blue"></span> Intelligence Output
        </div>
        """, unsafe_allow_html=True)
        tab_ans, tab_ev, tab_hist = st.tabs(["ANSWER", "EVIDENCE", "HISTORY"])

        with tab_ans:
            if st.session_state.last_answer:
                st.markdown(
                    f'<div class="answer-body">{st.session_state.last_answer}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="empty-state">AWAITING QUERY INPUT…</div>',
                    unsafe_allow_html=True
                )

        with tab_ev:
            evidence_cards(st.session_state.last_docs)

        with tab_hist:
            if st.session_state.chat_history:
                for line in st.session_state.chat_history[-10:]:
                    cls = "hist-user" if line.startswith("User:") else "hist-assistant"
                    st.markdown(
                        f'<div class="{cls}">{line}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✕ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.markdown(
                    '<div class="empty-state">NO CONVERSATION LOG YET.</div>',
                    unsafe_allow_html=True
                )

    # ── Execute query ─────────────────────────────────────────────────────────
    if ask:
        if not question.strip():
            st.warning("Please enter a question before executing.")
            st.stop()
        with st.spinner("Scanning knowledge base…"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            history_text = "\n".join(st.session_state.chat_history[-6:])
            docs_ret, ctx, answer = answer_question(
                llm, st.session_state.vectorstore, question, history_text
            )
        st.session_state.chat_history.append(f"User: {question}")
        st.session_state.chat_history.append(f"Assistant: {answer}")
        st.session_state.last_answer  = answer
        st.session_state.last_context = ctx
        st.session_state.last_docs    = docs_ret
        st.rerun()


if __name__ == "__main__":
    main()
