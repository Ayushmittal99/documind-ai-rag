"""Offline fakes + PDF builders used by tests (no Gemini calls, no API credits)."""
import hashlib, math, re
import pymupdf
from langchain_core.embeddings import Embeddings

DIM = 512


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings. Shared words => higher cosine."""
    def __init__(self):
        self.calls = 0
    def _vec(self, text):
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]
    def embed_query(self, text):
        return self._vec(text)


class FakeResp:
    def __init__(self, content): self.content = content


class FakeLLM:
    def __init__(self, reply="Paris is mentioned. [Source: t.pdf, page 1]"):
        self.reply, self.calls, self.last_messages = reply, 0, None
    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        return FakeResp(self.reply)


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_image_only_pdf() -> bytes:
    """A page containing only a drawn shape (no text layer) - simulates a scan."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(100, 100, 300, 300), fill=(0.5, 0.5, 0.5))
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return data
