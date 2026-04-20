from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import split_sentences


class ExtractiveSummarizer:
    def __init__(self, max_sentences: int = 6) -> None:
        self.max_sentences = max_sentences

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= self.max_sentences:
            return " ".join(sentences)

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        centroid = matrix.mean(axis=0)
        scores = np.asarray(matrix @ centroid.T).ravel()
        top_indices = np.argsort(scores)[::-1][: self.max_sentences]
        ordered = sorted(top_indices)
        return " ".join(sentences[index] for index in ordered)
