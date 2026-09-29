"""
app.py — Streamlit front-end for the RAG Document Q&A Chatbot.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push this folder to a public GitHub repo.
    2. Go to share.streamlit.io -> New app -> pick the repo -> main file app.py.
    3. (Optional) Add OPENAI_API_KEY or ANTHROPIC_API_KEY under App Settings -> Secrets
       if you want generative answers instead of the offline extractive mode.
"""

import os
import tempfile

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Q&A Chatbot")
st.caption(
    "Upload documents, then ask questions about them. Retrieval works fully "
    "offline (TF-IDF + cosine similarity). Add an API key in the sidebar for "
    "natural-language generated answers."
)

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Answer generation", ["Offline (extractive)", "OpenAI", "Anthropic"])
    api_key = None
    llm_provider = None
    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API key", type="password",
                                 value=os.environ.get("OPENAI_API_KEY", ""))
        llm_provider = "openai"
    elif provider == "Anthropic":
        api_key = st.text_input("Anthropic API key", type="password",
                                 value=os.environ.get("ANTHROPIC_API_KEY", ""))
        llm_provider = "anthropic"

    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=6, value=3)

    st.divider()
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .md files", type=["txt", "md"], accept_multiple_files=True
    )
    use_sample = st.checkbox("Use bundled sample documents instead", value=not uploaded_files)

# ---------------- Build / cache the index ----------------
@st.cache_resource(show_spinner="Indexing documents...")
def build_engine(folder_path: str, cache_key: str):
    engine = RAGEngine()
    engine.build_index(folder_path)
    return engine

doc_folder = None
if use_sample:
    doc_folder = os.path.join(os.path.dirname(__file__), "sample_docs")
    cache_key = "sample"
elif uploaded_files:
    tmp_dir = tempfile.mkdtemp()
    for f in uploaded_files:
        with open(os.path.join(tmp_dir, f.name), "wb") as out:
            out.write(f.getbuffer())
    doc_folder = tmp_dir
    cache_key = ",".join(f.name for f in uploaded_files)
else:
    st.info("Upload documents in the sidebar, or check 'Use bundled sample documents'.")
    st.stop()

engine = build_engine(doc_folder, cache_key)
st.success(f"Indexed {len(engine.chunks)} chunks from your documents.")

# ---------------- Chat ----------------
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

query = st.chat_input("Ask a question about your documents...")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            result = engine.answer(query, top_k=top_k, llm_provider=llm_provider, api_key=api_key)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📎 Sources used"):
                for s in result["sources"]:
                    st.write(f"- **{s['source']}** (chunk {s['chunk_id']}, score {s['score']})")
    st.session_state.history.append(("assistant", result["answer"]))
