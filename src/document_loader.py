from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from src.utils import clean_text


class DocumentLoader:
    """Loads PDF and text documents into a normalized format."""

    @staticmethod
    def load_file(file_path: Path) -> Dict:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        if suffix == ".txt":
            return DocumentLoader._load_txt(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _load_pdf(file_path: Path) -> Dict:
        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = clean_text(extracted)
            if cleaned:
                pages.append(cleaned)
            else:
                pages.append(f"[No extractable text found on page {page_number}]")

        return {
            "doc_id": file_path.stem,
            "file_name": file_path.name,
            "file_type": "pdf",
            "text": "\n\n".join(pages),
            "pages": pages,
        }

    @staticmethod
    def _load_txt(file_path: Path) -> Dict:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(text)
        return {
            "doc_id": file_path.stem,
            "file_name": file_path.name,
            "file_type": "txt",
            "text": cleaned,
            "pages": [cleaned],
        }
