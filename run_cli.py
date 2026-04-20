from pathlib import Path

from src.answerer import GroundedAnswerer
from src.chunker import TextChunker
from src.config import CONFIG
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingService
from src.summarizer import ExtractiveSummarizer
from src.vector_store import FaissVectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents_from_data_folder(data_dir: Path):
    documents = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            documents.append(DocumentLoader.load_file(path))
    return documents


if __name__ == "__main__":
    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("Create a data/ folder and add PDF or TXT files before running the CLI.")

    documents = load_documents_from_data_folder(data_dir)
    if not documents:
        raise ValueError("No PDF or TXT files found inside the data/ folder.")

    chunker = TextChunker(CONFIG.chunk_size, CONFIG.chunk_overlap)
    embedding_service = EmbeddingService(CONFIG.embedding_model)
    answerer = GroundedAnswerer(embedding_service, CONFIG.max_answer_sentences)
    summarizer = ExtractiveSummarizer(CONFIG.max_summary_sentences)

    chunks = chunker.chunk_documents(documents)
    vectors = embedding_service.encode([chunk["text"] for chunk in chunks])

    vector_store = FaissVectorStore()
    vector_store.build(vectors, chunks)

    summary_input = "\n\n".join(document["text"][:2000] for document in documents)
    print("\n=== DOCUMENT SUMMARY ===\n")
    print(summarizer.summarize(summary_input))

    print("\n=== READY ===")
    print("Ask questions about the documents. Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        query_vector = embedding_service.encode([question])[0]
        retrieved = vector_store.search(query_vector, CONFIG.top_k)
        answer, citations = answerer.answer(question, retrieved)

        print("\nAssistant:", answer)
        print("Citations:")
        for citation in citations:
            print("-", citation)
        print()
