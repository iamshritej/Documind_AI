from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import split_sentences


class GroundedAnswerer:
    def __init__(self, embedding_service, max_answer_sentences: int = 4) -> None:
        self.embedding_service = embedding_service
        self.max_answer_sentences = max_answer_sentences

    def answer(self, question: str, retrieved_chunks: List[Dict]) -> Tuple[str, List[str]]:
        if not retrieved_chunks:
            return (
                "I could not find enough supporting content in the uploaded documents to answer that question.",
                [],
            )

        candidate_sentences: List[str] = []
        citations: List[str] = []

        for chunk in retrieved_chunks:
            sentences = split_sentences(chunk["text"])
            for sentence in sentences:
                if len(sentence) >= 30:
                    candidate_sentences.append(sentence)
                    citations.append(f"{chunk['file_name']} ({chunk['chunk_id']})")

        if not candidate_sentences:
            fallback = retrieved_chunks[0]["text"][:350].strip()
            return fallback, [f"{retrieved_chunks[0]['file_name']} ({retrieved_chunks[0]['chunk_id']})"]

        question_vector = self.embedding_service.encode([question])
        sentence_vectors = self.embedding_service.encode(candidate_sentences)
        similarities = cosine_similarity(question_vector, sentence_vectors)[0]

        ranked_indices = np.argsort(similarities)[::-1]

        selected_sentences: List[str] = []
        selected_citations: List[str] = []
        used_sentence_roots = set()

        for idx in ranked_indices:
            sentence = candidate_sentences[idx].strip()
            root = sentence[:60].lower()
            if root in used_sentence_roots:
                continue
            selected_sentences.append(sentence)
            selected_citations.append(citations[idx])
            used_sentence_roots.add(root)
            if len(selected_sentences) >= self.max_answer_sentences:
                break

        answer = " ".join(selected_sentences)
        return answer, selected_citations
