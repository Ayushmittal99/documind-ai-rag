import pytest
from pdf_processor import (EmptyPDFError, InvalidPDFError, PasswordProtectedError, PDFTooLargeError,
                           ScannedPDFError, chunk_pages, extract_pages)
from helpers import make_encrypted_pdf, make_image_only_pdf, make_pdf


def test_extracts_text_and_page_metadata():
    r = extract_pages(make_pdf(["Alpha on page one.", "Beta on page two."]), "a.pdf")
    assert r.total_pages == 2 and r.pages_with_text == 2
    assert [p.metadata["page"] for p in r.pages] == [1, 2]
    assert r.pages[0].metadata["source"] == "a.pdf"
    assert "Alpha" in r.pages[0].page_content


def test_blank_page_is_skipped_but_counted():
    r = extract_pages(make_pdf(["Some text", ""]), "a.pdf")
    assert r.total_pages == 2 and r.pages_with_text == 1


def test_wrong_extension_rejected():
    with pytest.raises(InvalidPDFError):
        extract_pages(b"hello", "notes.txt")


def test_empty_file_rejected():
    with pytest.raises(EmptyPDFError):
        extract_pages(b"", "a.pdf")


def test_garbage_bytes_rejected():
    with pytest.raises(InvalidPDFError):
        extract_pages(b"this is not a pdf at all", "a.pdf")


def test_corrupt_pdf_with_header_rejected():
    with pytest.raises(InvalidPDFError):
        extract_pages(b"%PDF-1.7\n garbage garbage", "a.pdf")


def test_scanned_pdf_gives_clear_ocr_message():
    with pytest.raises(ScannedPDFError, match="OCR"):
        extract_pages(make_image_only_pdf(), "scan.pdf")


def test_password_protected_rejected():
    with pytest.raises(PasswordProtectedError):
        extract_pages(make_encrypted_pdf(), "locked.pdf")


def test_size_and_page_limits():
    with pytest.raises(PDFTooLargeError):
        extract_pages(make_pdf(["a", "b", "c"]), "a.pdf", max_pages=2)


def test_chunking_respects_size_and_keeps_metadata():
    pages = extract_pages(make_pdf(["word " * 400, "other " * 400]), "doc.pdf").pages
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 2
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.metadata["source"] == "doc.pdf" and c.metadata["page"] in (1, 2) for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_overlap_creates_shared_text_between_neighbours():
    text = " ".join(f"w{i}" for i in range(300))
    from langchain_core.documents import Document
    chunks = chunk_pages([Document(page_content=text, metadata={"source": "x", "page": 1})], 200, 60)
    first_tail = chunks[0].page_content.split()[-1]
    assert first_tail in chunks[1].page_content


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], size, overlap)
