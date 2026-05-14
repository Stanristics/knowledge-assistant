"""
RAG Engine — core retrieval and generation logic.
Supports PDF, TXT, and DOCX ingestion, ChromaDB vector store,
conversation memory, and source-cited responses.
Uses Groq (free) as the LLM provider.
"""

import os
import hashlib
from typing import Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.schema import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model, no torch required
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 4
MEMORY_WINDOW = 6

# Available free Groq models
GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
DEFAULT_MODEL = "llama-3.1-70b-versatile"


def load_document(file_path: str) -> list[Document]:
    """Load a document from disk into LangChain Document objects."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
    return loader.load()


def build_vectorstore(documents: list[Document], persist_dir: str) -> Chroma:
    """Chunk documents, embed them, and persist to ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Deduplicate by content hash to avoid re-embedding identical chunks
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_chunks.append(chunk)

    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        unique_chunks,
        embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore(persist_dir: str) -> Chroma:
    """Load an existing ChromaDB vectorstore from disk."""
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def build_chain(
    vectorstore: Chroma,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    memory: Optional[ConversationBufferWindowMemory] = None,
) -> ConversationalRetrievalChain:
    """Build the conversational retrieval chain with memory using Groq."""
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={"k": RETRIEVER_K, "fetch_k": 12},
    )

    if memory is None:
        memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": _build_prompt(),
        },
    )
    return chain


def _build_prompt():
    """System prompt that enforces citation and honest uncertainty."""
    from langchain.prompts import PromptTemplate

    template = """You are a precise knowledge assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I could not find that in the uploaded documents."
Always cite which document/page your answer comes from at the end of your response.

Context:
{context}

Chat history:
{chat_history}

Question: {question}

Answer (with source citation):"""

    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template,
    )


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a question and return answer + sources.

    Returns:
        {
            "answer": str,
            "sources": [{"file": str, "page": int, "snippet": str}]
        }
    """
    result = chain({"question": question})
    sources = []
    for doc in result.get("source_documents", []):
        meta = doc.metadata
        sources.append({
            "file": meta.get("source", "unknown"),
            "page": meta.get("page", "—"),
            "snippet": doc.page_content[:200].replace("\n", " "),
        })
    return {"answer": result["answer"], "sources": sources}


def evaluate_retrieval(chain: ConversationalRetrievalChain, test_pairs: list[dict]) -> dict:
    """
    Simple faithfulness evaluation against ground-truth Q&A pairs.
    test_pairs: [{"question": str, "expected_keywords": [str]}]
    Returns hit rate (fraction where all keywords appear in the answer).
    """
    hits = 0
    results = []
    for pair in test_pairs:
        result = ask(chain, pair["question"])
        answer_lower = result["answer"].lower()
        all_found = all(kw.lower() in answer_lower for kw in pair["expected_keywords"])
        hits += int(all_found)
        results.append({
            "question": pair["question"],
            "answer": result["answer"],
            "passed": all_found,
        })
    return {
        "hit_rate": round(hits / len(test_pairs), 3),
        "total": len(test_pairs),
        "hits": hits,
        "details": results,
    }
