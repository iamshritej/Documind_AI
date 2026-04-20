from typing import Dict, List


class TextChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Dict) -> List[Dict]:
        text = document["text"]
        chunks: List[Dict] = []

        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{document['doc_id']}_chunk_{chunk_id}",
                        "doc_id": document["doc_id"],
                        "file_name": document["file_name"],
                        "text": chunk_text,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunk_id += 1

            if end == len(text):
                break
            start = max(0, end - self.overlap)

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks: List[Dict] = []
        for document in documents:
            all_chunks.extend(self.chunk_document(document))
        return all_chunks
