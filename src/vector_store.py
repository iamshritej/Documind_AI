from typing import Dict, List

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self) -> None:
        self.index = None
        self.metadata: List[Dict] = []
        self.dimension = 0

    def build(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        if vectors.size == 0:
            raise ValueError("No vectors found to build the index.")

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        scores, indices = self.index.search(query_vector, top_k)
        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
