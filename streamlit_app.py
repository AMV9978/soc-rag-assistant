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

# ── Inject custom CSS ──────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #080c10;
        color: #c9d1d9;
    }

    .stApp {
        background: #080c10;
        background-image:
            radial-gradient(ellipse at 20% 0%, rgba(0,255,136,0.04) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 100%, rgba(0,180,255,0.04) 0%, transparent 60%);
    }

    /* ── Scanline overlay ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.08) 2px,
            rgba(0,0,0,0.08) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1a2332 !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }

    /* ── Header banner ── */
    .soc-header {
        border-left: 3px solid #00ff88;
        padding: 0.6rem 0 0.6rem 1.2rem;
        margin-bottom: 0.2rem;
    }

    .soc-header h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: -0.02em;
        color: #f0f6fc;
        margin: 0;
        line-height: 1;
    }

    .soc-header h1 span {
        color: #00ff88;
    }

    .soc-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #4a5568;
        letter-spacing: 0.05em;
        margin-top: 0.4rem;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1px;
        background: #1a2332;
        border: 1px solid #1a2332;
        border-radius: 6px;
        overflow: hidden;
        margin: 1.2rem 0 1.4rem 0;
    }

    .metric-card {
        background: #0d1117;
        padding: 1rem 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00ff88, transparent);
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4a6741;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff88;
        line-height: 1;
    }

    .metric-value.status {
        font-size: 1rem;
        color: #58a6ff;
    }

    /* ── Section labels ── */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a5568;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #1a2332;
    }

    /* ── Ask panel ── */
    .ask-panel {
        background: #0d1117;
        border: 1px solid #1a2332;
        border-radius: 8px;
        padding: 1.4rem;
        height: 100%;
    }

    .ask-panel-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #00ff88;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .ask-panel-title::before {
        content: '>';
        color: #00ff88;
        font-size: 0.8rem;
    }

    /* ── Answer panel ── */
    .answer-panel {
        background: #0d1117;
        border: 1px solid #1a2332;
        border-radius: 8px;
        padding: 1.4rem;
        min-height: 300px;
    }

    .answer-panel-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #58a6ff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .answer-panel-title::before {
        content: '//';
        color: #1a2332;
        font-size: 0.8rem;
    }

    /* ── Answer content ── */
    .answer-content {
        font-family: 'Syne', sans-serif;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #c9d1d9;
    }

    .answer-content strong {
        color: #f0f6fc;
    }

    /* ── Evidence card ── */
    .evidence-card {
        background: #0a0f16;
        border: 1px solid #1a2332;
        border-left: 3px solid #58a6ff;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8b949e;
    }

    .evidence-source {
        color: #58a6ff;
        font-size: 0.7rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }

    /* ── Sidebar pack checkbox style ── */
    .pack-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid #0d1117;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
    }

    .pack-active {
        color: #00ff88;
    }

    /* ── Sidebar section header ── */
    .sidebar-section {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #2a3544;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #1a2332;
    }

    /* ── Control panel title ── */
    .cp-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        color: #f0f6fc;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .cp-badge {
        background: #00ff8820;
        border: 1px solid #00ff8840;
        color: #00ff88;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.55rem;
        letter-spacing: 0.1em;
        padding: 0.1rem 0.4rem;
        border-radius: 3px;
    }

    /* ── Streamlit widget overrides ── */
    .stSelectbox > div > div {
        background: #0a0f16 !important;
        border: 1px solid #1a2332 !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 4px !important;
    }

    .stTextArea > div > div > textarea {
        background: #0a0f16 !important;
        border: 1px solid #1a2332 !important;
        color: #c9d1d9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 4px !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 0 1px #00ff8840 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: #00ff88 !important;
        color: #080c10 !important;
        border: none !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #00cc6e !important;
        box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
    }

    /* ── Secondary button ── */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        border: 1px solid #1a2332 !important;
        color: #8b949e !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        border-color: #00ff88 !important;
        color: #00ff88 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1a2332 !important;
        gap: 0 !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: #4a5568 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.5rem 1rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: #00ff88 !important;
        border-bottom: 2px solid #00ff88 !important;
    }

    /* ── Checkbox ── */
    .stCheckbox label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        color: #8b949e !important;
    }

    .stCheckbox input:checked + div {
        background: #00ff88 !important;
        border-color: #00ff88 !important;
    }

    /* ── Metrics override ── */
    [data-testid="stMetric"] {
        background: transparent !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: transparent !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
        border: none !important;
        border-bottom: 1px solid #1a2332 !important;
    }

    .streamlit-expanderContent {
        background: transparent !important;
        border: none !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #1a2332 !important;
        margin: 1rem 0 !important;
    }

    /* ── Info/warning/success ── */
    .stAlert {
        background: #0d1117 !important;
        border: 1px solid #1a2332 !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #00ff88 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #080c10; }
    ::-webkit-scrollbar-thumb { background: #1a2332; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #00ff88; }

    /* ── History entry ── */
    .hist-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #4a5568;
        padding: 0.4rem 0;
        border-bottom: 1px solid #0d1117;
        line-height: 1.5;
    }

    .hist-entry.user { color: #58a6ff; }
    .hist-entry.assistant { color: #8b949e; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #0a0f16 !important;
        border: 1px dashed #1a2332 !important;
        border-radius: 4px !important;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #00ff8812;
        border: 1px solid #00ff8830;
        color: #00ff88;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.1em;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
    }

    .status-badge.blue {
        background: #58a6ff12;
        border-color: #58a6ff30;
        color: #58a6ff;
    }

    .status-badge::before {
        content: '●';
        font-size: 0.5rem;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid #1a2332 !important;
        color: #4a5568 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.05em !important;
        border-radius: 3px !important;
        padding: 0.25rem 0.6rem !important;
    }

    .stDownloadButton > button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }

    /* ── Label text ── */
    .stSelectbox label, .stTextArea label, .stCheckbox label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
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
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.55})
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return [], "", "I don't have enough information from the documents."
    context_text = "\n\n".join([f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in retrieved_docs])
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
    response = llm.invoke(prompt.format(context=context_text, question=question, history=history_text))
    return retrieved_docs, context_text, response.content

def evidence_cards(docs):
    if not docs:
        st.markdown('<div class="evidence-card" style="color:#4a5568;">No evidence retrieved yet.</div>', unsafe_allow_html=True)
        return
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        excerpt = d.page_content.strip()[:700]
        st.markdown(f"""
        <div class="evidence-card">
            <div class="evidence-source">[ SOURCE {i} ] &nbsp;{src}</div>
            {excerpt}
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
        st.error("NO API KEY — Add OPENAI_API_KEY under Settings → Secrets.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div class="cp-title">
            CONTROL PANEL <span class="cp-badge">v1.0</span>
        </div>
        """, unsafe_allow_html=True)

        # Knowledge Packs
        st.markdown('<div class="sidebar-section">Knowledge Packs</div>', unsafe_allow_html=True)
        with st.expander("", expanded=True):
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
                    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;color:#4a6741;margin-top:0.5rem;">ACTIVE: {", ".join(enabled_packs)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;color:#6b2737;margin-top:0.5rem;">NO PACKS ACTIVE</div>', unsafe_allow_html=True)
            else:
                enabled_packs = None
                st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#4a5568;">No packs found.</div>', unsafe_allow_html=True)

        # Upload
        st.markdown('<div class="sidebar-section">Upload Knowledge</div>', unsafe_allow_html=True)
        with st.expander("", expanded=False):
            uploaded_files = st.file_uploader(
                "Drop .txt or .md files",
                type=["txt", "md"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

        # Index Controls
        st.markdown('<div class="sidebar-section">Index Controls</div>', unsafe_allow_html=True)
        with st.expander("", expanded=True):
            left, right = st.columns(2)
            rebuild_clicked = left.button("⟳ Rebuild", use_container_width=True)
            load_clicked = right.button("⚡ Load", use_container_width=True)

        # Footer
        st.markdown("""
        <div style="position:absolute;bottom:1.5rem;left:1rem;right:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#2a3544;border-top:1px solid #1a2332;padding-top:0.8rem;line-height:1.6;">
                Angelo Vasquez<br>
                NYU M.S. Cybersecurity<br>
                <span style="color:#00ff8830;">████████████</span> SOC RAG v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Load docs ──
    docs = []
    if KNOWLEDGE_DIR.exists():
        docs.extend(load_knowledge_files(enabled_packs))
    if uploaded_files:
        for file in uploaded_files:
            text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": f"UPLOAD:{file.name}"}))

    enabled_count = len(enabled_packs) if enabled_packs is not None else len(list_packs())
    index_status = get_index_status()
    current_key = make_pack_key(enabled_packs)
    saved_key = load_saved_pack_key()
    index_present = (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()
    packs_changed = (saved_key != current_key)

    # ── Header ──
    st.markdown("""
    <div class="soc-header">
        <h1>SOC RAG <span>ASSISTANT</span></h1>
    </div>
    <div class="soc-caption">
        ENTERPRISE CYBERSECURITY INTELLIGENCE &nbsp;·&nbsp; RETRIEVAL-AUGMENTED GENERATION &nbsp;·&nbsp; NYU M.S. CYBERSECURITY
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ──
    status_label = "ONLINE" if index_status == "ready" else "BUILDING"
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Packs Active</div>
            <div class="metric-value">{enabled_count:02d}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Knowledge Files</div>
            <div class="metric-value">{count_knowledge_files(enabled_packs):02d}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Index Status</div>
            <div class="metric-value status">{status_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not docs:
        st.warning("NO KNOWLEDGE FILES FOUND — check your packs directory.")
        st.stop()

    # ── Auto rebuild / load ──
    if index_present and packs_changed:
        with st.spinner("PACK DELTA DETECTED — REBUILDING INDEX…"):
            vs = build_vectorstore(docs, embeddings)
            save_vectorstore(vs)
            save_pack_key(current_key)
            st.session_state.vectorstore = vs
        st.success("INDEX REBUILT")

    if rebuild_clicked:
        with st.spinner("BUILDING FAISS INDEX…"):
            vs = build_vectorstore(docs, embeddings)
            save_vectorstore(vs)
            save_pack_key(current_key)
            st.session_state.vectorstore = vs
        st.success("INDEX BUILT AND SAVED")

    if load_clicked:
        if index_present and not packs_changed:
            st.session_state.vectorstore = load_vectorstore(embeddings)
            st.success("SAVED INDEX LOADED")
        else:
            st.error("PACK MISMATCH OR NO SAVED INDEX — REBUILD FIRST")

    if st.session_state.vectorstore is None:
        with st.spinner("INITIALIZING KNOWLEDGE BASE…"):
            if index_present and not packs_changed:
                st.session_state.vectorstore = load_vectorstore(embeddings)
            else:
                st.session_state.vectorstore = build_vectorstore(docs, embeddings)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Main columns ──
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="ask-panel">', unsafe_allow_html=True)
        st.markdown('<div class="ask-panel-title">Query Interface</div>', unsafe_allow_html=True)
        category = st.selectbox("Domain", list(PROMPT_LIBRARY.keys()), label_visibility="visible")
        example = st.selectbox("Example Queries", PROMPT_LIBRARY[category], label_visibility="visible")
        question = st.text_area("Input", value=example, height=110, label_visibility="visible")
        ask = st.button("▶ EXECUTE QUERY", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
        st.markdown('<div class="answer-panel-title">Intelligence Output</div>', unsafe_allow_html=True)
        tab_ans, tab_ev, tab_hist = st.tabs(["ANSWER", "EVIDENCE", "HISTORY"])

        with tab_ans:
            if st.session_state.last_answer:
                st.markdown(
                    f'<div class="answer-content">{st.session_state.last_answer}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#2a3544;margin-top:1rem;">AWAITING QUERY INPUT...</div>',
                    unsafe_allow_html=True
                )

        with tab_ev:
            evidence_cards(st.session_state.last_docs)

        with tab_hist:
            if st.session_state.chat_history:
                for line in st.session_state.chat_history[-10:]:
                    role = "user" if line.startswith("User:") else "assistant"
                    st.markdown(f'<div class="hist-entry {role}">{line}</div>', unsafe_allow_html=True)
                if st.button("✕ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#2a3544;margin-top:1rem;">NO CONVERSATION LOG YET.</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Execute query ──
    if ask:
        if not question.strip():
            st.warning("EMPTY QUERY — INPUT REQUIRED")
            st.stop()
        with st.spinner("SCANNING KNOWLEDGE BASE…"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            history_text = "\n".join(st.session_state.chat_history[-6:])
            docs_ret, ctx, answer = answer_question(llm, st.session_state.vectorstore, question, history_text)
        st.session_state.chat_history.append(f"User: {question}")
        st.session_state.chat_history.append(f"Assistant: {answer}")
        st.session_state.last_answer = answer
        st.session_state.last_context = ctx
        st.session_state.last_docs = docs_ret
        st.rerun()


if __name__ == "__main__":
    main()