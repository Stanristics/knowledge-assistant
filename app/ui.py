"""
Streamlit UI — Startup Knowledge Assistant
Run: streamlit run app/ui.py
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from rag_engine import (
    load_document,
    build_vectorstore,
    load_vectorstore,
    build_chain,
    ask,
    SUPPORTED_EXTENSIONS,
)
from langchain.memory import ConversationBufferWindowMemory

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f13; color: #e8e8f0; }
    .main-header { font-size: 2rem; font-weight: 700; color: #a78bfa; letter-spacing: -0.02em; }
    .sub-header { font-size: 0.9rem; color: #6b7280; margin-top: -0.5rem; }
    .source-card { background: #1a1a24; border: 1px solid #2d2d3d; border-radius: 8px;
                   padding: 0.75rem; margin: 0.4rem 0; font-size: 0.8rem; color: #9ca3af; }
    .source-file { color: #a78bfa; font-weight: 600; }
    .metric-card { background: #1a1a24; border: 1px solid #2d2d3d; border-radius: 8px;
                   padding: 1rem; text-align: center; }
    .stChatMessage { background: #1a1a24 !important; border-radius: 12px !important; }
    div[data-testid="stSidebar"] { background-color: #0d0d11 !important; }
</style>
""", unsafe_allow_html=True)

PERSIST_DIR = ".chroma_db"

# ── Session state ─────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=6, memory_key="chat_history", output_key="answer", return_messages=True
    )
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    api_key = st.text_input("Groq API key", type="password", placeholder="gsk_...")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    model = st.selectbox("Model", [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.markdown("### Upload documents")
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index documents", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your Groq API key first.")
        else:
            with st.spinner("Loading and embedding documents..."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    docs = load_document(tmp_path)
                    # Tag each doc with original filename
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)
                    os.unlink(tmp_path)

                vectorstore = build_vectorstore(all_docs, PERSIST_DIR)
                st.session_state.chain = build_chain(
                    vectorstore, model, temperature, st.session_state.memory
                )
                st.session_state.doc_count = len(uploaded_files)
                st.session_state.chunk_count = vectorstore._collection.count()
                st.session_state.messages = []
                st.success(f"Indexed {len(uploaded_files)} document(s).")

    st.divider()
    if st.session_state.chain:
        st.markdown("### Index stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.doc_count)
        with col2:
            st.metric("Chunks", st.session_state.chunk_count)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferWindowMemory(
            k=6, memory_key="chat_history", output_key="answer", return_messages=True
        )
        if st.session_state.chain:
            st.session_state.chain.memory = st.session_state.memory
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Knowledge Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload your company documents and ask questions — answers include source citations.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.chain:
    st.info("Upload documents in the sidebar and click **Index documents** to get started.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-file">{src["file"]}</span>'
                        f' · Page {src["page"]}<br>'
                        f'<em>{src["snippet"]}…</em></div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(st.session_state.chain, prompt)
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for src in result["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-file">{src["file"]}</span>'
                        f' · Page {src["page"]}<br>'
                        f'<em>{src["snippet"]}…</em></div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
