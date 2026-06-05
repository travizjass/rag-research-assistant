"""CLI document ingestion: parse files, chunk text, embed, and upsert to Qdrant.

Usage:
    python -m ingest.document_loader --path ./your-documents/
"""

import argparse
import os
from typing import Dict, List

from app.vectorstore.qdrant_client import QdrantStore

# File extensions this loader knows how to parse.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_pdf(path: str) -> List[Dict]:
    """Extract text per page from a PDF, tagging each section with its page no."""
    import pdfplumber

    sections = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append({"text": text, "page": page_number})
    return sections


def load_txt(path: str) -> List[Dict]:
    """Read a plain-text file into a single, page-less text section."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [{"text": f.read(), "page": None}]


def load_docx(path: str) -> List[Dict]:
    """Read a Word document, joining all non-empty paragraphs into one section."""
    import docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [{"text": text, "page": None}]


def load_file(path: str) -> List[Dict]:
    """Dispatch to the right parser based on the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".txt":
        return load_txt(path)
    if ext == ".docx":
        return load_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping character windows for embedding.

    The overlap preserves context that would otherwise be cut across a
    chunk boundary, which improves retrieval quality.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)  # guard against a non-advancing window
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def build_chunks(path: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Parse a single file and flatten it into embeddable chunk records."""
    source = os.path.basename(path)
    records: List[Dict] = []
    for section in load_file(path):
        for piece in chunk_text(section["text"], chunk_size, overlap):
            records.append(
                {"text": piece, "source": source, "page": section["page"]}
            )
    return records


def collect_files(path: str) -> List[str]:
    """Return every supported file under ``path`` (a file or a directory)."""
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                found.append(os.path.join(root, name))
    return found


def ingest(path: str, chunk_size: int = 1000, overlap: int = 150) -> int:
    """Ingest a file or directory into Qdrant; returns total chunks written."""
    store = QdrantStore()
    store.ensure_collection()

    total = 0
    for file_path in collect_files(path):
        chunks = build_chunks(file_path, chunk_size, overlap)
        written = store.upsert(chunks)
        total += written
        print(f"  + {file_path}: {written} chunks")
    return total


def main() -> None:
    """Parse CLI arguments and run the ingestion process."""
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--path", required=True, help="File or directory to ingest.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    print(f"Ingesting from: {args.path}")
    total = ingest(args.path, args.chunk_size, args.overlap)
    print(f"Done. {total} chunks indexed.")


if __name__ == "__main__":
    main()
