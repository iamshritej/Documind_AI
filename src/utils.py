import re
from typing import Iterable, List


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return safe.strip("_") or "document"


def flatten(items: Iterable[Iterable[str]]) -> List[str]:
    flattened: List[str] = []
    for item in items:
        flattened.extend(item)
    return flattened
