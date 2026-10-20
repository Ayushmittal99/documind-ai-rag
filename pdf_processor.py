"""PDF validation, text extraction (PyMuPDF) and chunking."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pymupdf  # PyMuPDF >= 1.24 exposes `pymupdf` (the old `fitz` name is deprecated)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFError(Exception):
    """Base class. Message is always safe to show to end users."""


class InvalidPDFError(PDFError):
    pass


class EmptyPDFError(PDFError):
    pass


class ScannedPDFError(PDFError):
    pass


class PasswordProtectedError(PDFError):
    pass


class PDFTooLargeError(PDFError):
    pass


@dataclass
class ExtractionResult:
    filename: str
    file_hash: str
    total_pages: int
    pages_with_text: int
    pages: list[Document]  # one Document per page that has text


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_pages(
    data: bytes, filename: str, max_mb: int = 20, max_pages: int = 200
) -> ExtractionResult:
    """Validate a PDF and return per-page Documents with page metadata (1-based)."""
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidPDFError("Unsupported file type. Please upload a .pdf file.")
    if not data:
        raise EmptyPDFError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise PDFTooLargeError(f"File is larger than {max_mb} MB. Please upload a smaller PDF.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise InvalidPDFError("This file is not a valid PDF (missing PDF header).")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise InvalidPDFError("The PDF is corrupt or could not be opened.") from None

    try:
        if doc.needs_pass:
            raise PasswordProtectedError("The PDF is password-protected. Remove the password and retry.")
        total = doc.page_count
        if total == 0:
            raise EmptyPDFError("The PDF has no pages.")
        if total > max_pages:
            raise PDFTooLargeError(f"PDF has {total} pages; the limit is {max_pages}.")

        pages: list[Document] = []
        for i in range(total):
            try:
                text = doc[i].get_text("text").strip()
            except Exception:
                text = ""  # a single bad page should not kill the whole document
            if text:
                pages.append(
                    Document(page_content=text, metadata={"source": filename, "page": i + 1})
                )
    finally:
        doc.close()

    if not pages:
        raise ScannedPDFError(
            "No extractable text found. This looks like a scanned/image-only PDF. "
            "OCR is not supported in this app."
        )
    return ExtractionResult(filename, file_sha256(data), total, len(pages), pages)


def chunk_pages(
    pages: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Document]:
    """Split page Documents into chunks. Metadata (source, page) is copied to every chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph > line > sentence > word
    )
    chunks = splitter.split_documents(pages)  # keeps page metadata per chunk
    for n, c in enumerate(chunks):
        c.metadata["chunk_id"] = n
    return chunks
