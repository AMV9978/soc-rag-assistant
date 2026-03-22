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
            return "Saved index found ✅"
        return "Index folder exists (partial) ⚠️"
    return "No saved index ❌"

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
        st.info("No evidence retrieved yet.")
        return
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        excerpt = d.page_content.strip()[:700]
        with st.container(border=True):
            st.markdown(f"**Source {i}:** `{src}`")
            st.write(excerpt)
            st.download_button(f"Download Source {i}", data=d.page_content, file_name=f"evidence_{i}.txt", mime="text/plain", use_container_width=True)

def main():
    st.set_page_config(page_title="SOC RAG Assistant", page_icon="🛡️", layout="wide")
    for key, default in [("chat_history", []), ("vectorstore", None), ("last_answer", ""), ("last_context", ""), ("last_docs", [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.title("🛡️ SOC RAG Assistant")
    st.caption("Enterprise-style cybersecurity assistant grounded in curated knowledge packs. Built by Angelo Vasquez · NYU M.S. Cybersecurity")

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ No OpenAI API key found. Add it under Settings → Secrets on Streamlit Cloud.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with st.sidebar:
        st.markdown("### 🧭 Control Panel")
        with st.expander("🧩 Knowledge Packs", expanded=True):
            pack_names = list_packs()
            if pack_names:
                c1, c2 = st.columns(2)
                if c1.button("Select All"):
                    for name in pack_names:
                        st.session_state[f"pack_{name}"] = True
                if c2.button("Deselect All"):
                    for name in pack_names:
                        st.session_state[f"pack_{name}"] = False
                enabled_packs = []
                for name in pack_names:
                    if f"pack_{name}" not in st.session_state:
                        st.session_state[f"pack_{name}"] = True
                    if st.checkbox(name, key=f"pack_{name}"):
                        enabled_packs.append(name)
                st.caption(f"Active: {', '.join(enabled_packs) if enabled_packs else 'None'}")
            else:
                enabled_packs = None
                st.info("No packs found.")
        with st.expander("📄 Upload Knowledge", expanded=False):
            uploaded_files = st.file_uploader("Upload .txt or .md files", type=["txt", "md"], accept_multiple_files=True)
        with st.expander("🧠 Index Controls", expanded=True):
            left, right = st.columns(2)
            rebuild_clicked = left.button("🔨 Rebuild")
            load_clicked = right.button("⚡ Load Saved")

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

    k1, k2, k3 = st.columns(3)
    k1.metric("Packs Active", enabled_count)
    k2.metric("Knowledge Files", count_knowledge_files(enabled_packs))
    k3.metric("Index Status", "Ready ✅" if "✅" in index_status else "Building…")

    if not docs:
        st.warning("No knowledge files found.")
        st.stop()

    if index_present and packs_changed:
        with st.spinner("Packs changed — rebuilding index…"):
            vs = build_vectorstore(docs, embeddings)
            save_vectorstore(vs)
            save_pack_key(current_key)
            st.session_state.vectorstore = vs
        st.success("Auto-rebuild complete ✅")

    if rebuild_clicked:
        with st.spinner("Building FAISS index…"):
            vs = build_vectorstore(docs, embeddings)
            save_vectorstore(vs)
            save_pack_key(current_key)
            st.session_state.vectorstore = vs
        st.success("Index built and saved ✅")

    if load_clicked:
        if index_present and not packs_changed:
            st.session_state.vectorstore = load_vectorstore(embeddings)
            st.success("Saved index loaded ✅")
        else:
            st.error("Pack selection changed or no saved index — rebuild first.")

    if st.session_state.vectorstore is None:
        with st.spinner("Initializing knowledge base…"):
            if index_present and not packs_changed:
                st.session_state.vectorstore = load_vectorstore(embeddings)
            else:
                st.session_state.vectorstore = build_vectorstore(docs, embeddings)

    st.divider()
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("🧠 Ask a Question")
        category = st.selectbox("Category", list(PROMPT_LIBRARY.keys()))
        example = st.selectbox("Example prompts", PROMPT_LIBRARY[category])
        question = st.text_area("Your question", value=example, height=100)
        ask = st.button("🚀 Ask", use_container_width=True, type="primary")

    with right:
        st.subheader("📌 Answer")
        tab_ans, tab_ev, tab_hist = st.tabs(["✅ Answer", "🔎 Evidence", "🗂️ History"])
        with tab_ans:
            if st.session_state.last_answer:
                st.markdown(st.session_state.last_answer)
            else:
                st.info("Ask a question to see the answer here.")
        with tab_ev:
            evidence_cards(st.session_state.last_docs)
        with tab_hist:
            if st.session_state.chat_history:
                for line in st.session_state.chat_history[-10:]:
                    st.write(line)
                if st.button("Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.write("No conversation yet.")

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        with st.spinner("Searching knowledge base…"):
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