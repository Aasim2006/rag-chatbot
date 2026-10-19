"""
app.py - Streamlit front-end for the RAG Document Q&A Chatbot.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Answers use Google Gemini when it is configured and working, and switch to the
local (offline) RAG engine automatically when it is not. There is nothing for
the user to select.

Configuration (never typed into the page, never sent to the browser):
    GEMINI_API_KEY   your Google AI Studio key
    GEMINI_MODEL     optional, defaults to a stable Gemini Flash model
Set them as environment variables, in a local .env file, or in Streamlit secrets.
"""

import os

import streamlit as st

try:  # optional: load a local .env file during development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from chat_state import ChatHistory
from rag_engine import (SUPPORTED_EXTENSIONS, GeminiConfig, GeminiService, RAGEngine,
                        read_folder)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Q&A Chatbot")
st.caption(
    "Upload documents (TXT, MD, PDF, PPTX) and ask questions about them. Answers come from "
    "Google Gemini when it is available and switch to local document search automatically "
    "when it is not."
)


# ---------------------------------------------------------------------------
# Configuration: environment variable first, then Streamlit secrets
# ---------------------------------------------------------------------------
def _setting(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file, or the key isn't there
        return None


@st.cache_resource(show_spinner=False)
def get_gemini_service(fingerprint: str, _config: GeminiConfig) -> GeminiService:
    """One shared service per configuration, so its health status survives reruns.
    `fingerprint` is a one-way hash, so changing the key or model creates a fresh service."""
    return GeminiService(_config)


@st.cache_data(show_spinner=False)
def sample_files():
    return read_folder(SAMPLE_DIR)


config = GeminiConfig.from_getter(_setting)
gemini = get_gemini_service(config.fingerprint, config)

# ---------------------------------------------------------------------------
# Per-session state (each visitor gets their own engine and history)
# ---------------------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
if "chat" not in st.session_state:
    st.session_state.chat = ChatHistory()
engine: RAGEngine = st.session_state.engine
chat: ChatHistory = st.session_state.chat


def esc(text: str) -> str:
    """Escape $ so Streamlit doesn't render currency amounts as LaTeX math."""
    return text.replace("$", "\\$")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=6, value=3)

    st.divider()
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload TXT, MD, PDF or PPTX files",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    use_sample = st.checkbox(
        "Use bundled sample documents",
        value=not uploaded_files,
        key=f"use_sample_{'uploads' if uploaded_files else 'none'}",
    )

# Build the list of documents we should have, then let the engine do the minimum
# work to match it (unchanged files are not parsed again; removed files are dropped).
wanted_files = []
if use_sample:
    wanted_files += sample_files()
for f in uploaded_files or []:
    wanted_files.append((f.name, f.getvalue()))
engine.sync_documents(wanted_files)

with st.sidebar:
    summary = engine.document_summary()
    if summary:
        st.caption("Indexed documents:")
        for name, n in summary:
            st.caption(esc(f"• {name} ({n} passage{'s' if n != 1 else ''})"))

    st.divider()
    st.header("🤖 Answer engine")
    status_slot = st.empty()  # redrawn after each answer so it never shows stale status


def render_status() -> None:
    """Show Gemini's current health. Reads in-memory state only: no network request."""
    status = gemini.status()
    with status_slot.container():
        if status.state == "not_configured":
            st.caption("⚪ Local mode. Add a Gemini API key to the server settings to enable Gemini answers.")
        elif status.state == "disabled":
            st.caption(f"🔴 Gemini unavailable. {status.message} Using local mode.")
        elif status.state == "paused":
            st.caption(f"🟠 Gemini paused. {status.message} Using local mode and retrying automatically.")
        elif status.state == "online":
            st.caption("🟢 Gemini online.")
        else:
            st.caption("🟡 Gemini configured. It is used automatically and local search takes over if it fails.")


render_status()

for name, reason in engine.skipped.items():
    st.warning(esc(f"Couldn't read {name}: {reason}."))

if summary:
    st.success(f"Indexed {len(engine.chunks)} passages from {len(summary)} document(s).")
else:
    st.info("Upload documents in the sidebar, or tick 'Use bundled sample documents'.")

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
MODE_ICONS = {
    "gemini": "🟢", "offline": "⚪", "offline_fallback": "🟠", "smalltalk": "💬",
    "no_match": "⚪", "no_documents": "⚪", "empty": "⚪", "error": "🔴",
}


def render_assistant(msg) -> None:
    meta = msg.meta or {}
    if meta.get("notice"):
        st.warning(esc(meta["notice"]), icon="⚠️")
    st.markdown(esc(msg.content))
    if meta.get("label"):
        st.caption(f"{MODE_ICONS.get(meta.get('mode'), '⚪')} {esc(meta['label'])}")
    sources = meta.get("sources") or []
    if sources:
        with st.expander(f"📎 Sources ({len(sources)})"):
            for s in sources:
                where = f", {s['location']}" if s.get("location") else ""
                st.markdown(esc(f"**[{s['n']}] {s['source']}{where}** · relevance {s['score']}"))
                st.caption(esc(s["snippet"]))


# Take the new question (if any) BEFORE drawing history, so it appears in order.
query = st.chat_input("Ask a question about your documents...")
if query:
    chat.add_user(query)

for msg in chat.messages:
    with st.chat_message(msg.role):
        if msg.role == "user":
            st.markdown(esc(msg.content))
        else:
            render_assistant(msg)

# Answer every question that has no answer yet, exactly once each.
while True:
    pending = chat.pending()
    if pending is None:
        break
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer = engine.answer(pending.content, top_k=top_k, gemini=gemini)
                text, meta = answer.text, answer.to_meta()
            except Exception:  # last line of defence: never show a stack trace to the user
                text = "Sorry, something went wrong while answering that. Please try again."
                meta = {"mode": "error", "label": "Error"}
        added = chat.add_assistant(pending.id, text, meta)
        if added is not None:
            render_assistant(added)
    render_status()  # the answer may have changed Gemini's health (failure, cooldown, recovery)
