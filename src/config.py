from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    max_answer_sentences: int = 4
    max_summary_sentences: int = 6


CONFIG = AppConfig()
