from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import streamlit as st

from src.answerer import GroundedAnswerer
from src.chunker import TextChunker
from src.config import CONFIG
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingService
from src.memory import ChatMemory
from src.summarizer import ExtractiveSummarizer
from src.utils import normalize_filename
from src.vector_store import FaissVectorStore


st.set_page_config(page_title="DocuMind AI", page_icon="📘", layout="wide")


def initialize_state() -> None:
    defaults = {
        "documents": [],
        "chunks": [],
        "vector_store": None,
        "summary": "",
        "memory": ChatMemory(),
        "last_results": [],
        "knowledge_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_knowledge_base(uploaded_files: List) -> None:
    if not uploaded_files:
        st.warning("Please upload at least one PDF or TXT file.")
        return

    embedding_service = EmbeddingService(CONFIG.embedding_model)
    chunker = TextChunker(CONFIG.chunk_size, CONFIG.chunk_overlap)
    summarizer = ExtractiveSummarizer(CONFIG.max_summary_sentences)

    documents = []
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for uploaded_file in uploaded_files:
            file_name = normalize_filename(uploaded_file.name)
            saved_path = temp_path / file_name
            saved_path.write_bytes(uploaded_file.getbuffer())
            document = DocumentLoader.load_file(saved_path)
            documents.append(document)

    chunks = chunker.chunk_documents(documents)
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_service.encode(texts)

    vector_store = FaissVectorStore()
    vector_store.build(vectors, chunks)

    joined_text = "\n\n".join(document["text"][:2000] for document in documents)
    summary = summarizer.summarize(joined_text)

    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.vector_store = vector_store
    st.session_state.summary = summary
    st.session_state.knowledge_ready = True
    st.session_state.memory = ChatMemory()
    st.session_state.last_results = []


def answer_question(question: str) -> None:
    if not st.session_state.knowledge_ready or st.session_state.vector_store is None:
        st.warning("Build the knowledge base first.")
        return

    embedding_service = EmbeddingService(CONFIG.embedding_model)
    answerer = GroundedAnswerer(embedding_service, CONFIG.max_answer_sentences)

    query_vector = embedding_service.encode([question])[0]
    retrieved = st.session_state.vector_store.search(query_vector, CONFIG.top_k)
    answer, citations = answerer.answer(question, retrieved)

    st.session_state.memory.add("user", question)
    st.session_state.memory.add("assistant", answer)
    st.session_state.last_results = retrieved
    st.session_state["last_citations"] = citations


def render_chat() -> None:
    for message in st.session_state.memory.get():
        with st.chat_message(message["role"]):
            st.write(message["content"])


def main() -> None:
    initialize_state()

    st.title("📘 DocuMind AI")
    st.caption("Python-only RAG document assistant for PDF/TXT question answering")

    with st.sidebar:
        st.header("Build Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        if st.button("Build knowledge base", use_container_width=True):
            with st.spinner("Processing documents and creating vector index..."):
                build_knowledge_base(uploaded_files)
            if st.session_state.knowledge_ready:
                st.success("Knowledge base created successfully.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.memory.clear()
            st.session_state.last_results = []
            st.session_state["last_citations"] = []

        if st.session_state.knowledge_ready:
            st.subheader("Project Stats")
            st.write(f"Documents: {len(st.session_state.documents)}")
            st.write(f"Chunks: {len(st.session_state.chunks)}")
            st.write(f"Top-K retrieval: {CONFIG.top_k}")

            with st.expander("Retrieved chunks"):
                for result in st.session_state.last_results:
                    st.markdown(
                        f"**{result['file_name']}** | score={result['score']:.3f} | {result['chunk_id']}"
                    )
                    st.write(result["text"][:400] + ("..." if len(result["text"]) > 400 else ""))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat")
        render_chat()
        user_question = st.chat_input("Ask something about the uploaded documents")
        if user_question:
            with st.spinner("Searching relevant chunks and generating grounded answer..."):
                answer_question(user_question)
            st.rerun()

    with col2:
        st.subheader("Document Summary")
        if st.session_state.summary:
            st.write(st.session_state.summary)
        else:
            st.info("Upload files and build the knowledge base to generate a summary.")

        st.subheader("Citations")
        citations = st.session_state.get("last_citations", [])
        if citations:
            for item in citations:
                st.write(f"- {item}")
        else:
            st.caption("Citations will appear after you ask a question.")

        st.subheader("About")
        st.write(
            "This app uses sentence embeddings + FAISS retrieval to answer questions from your uploaded documents."
        )


if __name__ == "__main__":
    main()
