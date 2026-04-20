from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model = get_embedding_model(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
