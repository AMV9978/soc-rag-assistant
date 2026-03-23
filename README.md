# 🛡️ SOC RAG Assistant

An enterprise-style cybersecurity assistant powered by Retrieval-Augmented Generation (RAG). Ask questions in plain English — it searches curated SOC knowledge packs and responds with structured analyst guidance.

**Built by Angelo Vasquez · NYU M.S. Cybersecurity · angelovasquez.net**

## 🔴 Live Demo
[Launch SOC RAG Assistant →](https://soc-rag-assistant-nsjlagjypalkutvrszkt3j.streamlit.app)

## What It Does
Instead of relying on an LLM's general training, this assistant only answers from your curated documents — making responses accurate, auditable, and domain-specific.

## Knowledge Packs
| Pack | Contents |
|---|---|
| fundamentals | CIA triad, core concepts, logging basics |
| soc_playbooks | Phishing, ransomware, account compromise, malware runbooks |
| splunk | SPL hunting queries for DNS tunneling, beaconing, PowerShell |
| aws_security | CloudTrail triage, S3 exposure, IAM key compromise |
| iam | IAM fundamentals, MFA events, OAuth abuse, access reviews |
| mitre_attack | ATT&CK mapping examples and detection signals |

## Tech Stack
- LangChain — RAG orchestration
- FAISS — Vector similarity search
- HuggingFace all-MiniLM-L6-v2 — Embeddings
- OpenAI GPT-4o-mini — Answer generation
- Streamlit — Web UI

## Local Setup
```bash
git clone https://github.com/AMV9978/soc-rag-assistant
cd soc-rag-assistant
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
python generate_packs.py
streamlit run streamlit_app.py
```

## Portfolio
[angelovasquez.net](https://angelovasquez.net)
