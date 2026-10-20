import pytest
from dataclasses import replace
from config import Settings
from pdf_processor import chunk_pages, extract_pages
from rag_pipeline import (NO_ANSWER, AuthError, RAGError, RAGPipeline, RateLimitError, RetryingEmbeddings,
                          ServiceError, VectorStoreError, classify_error, with_retries, build_llm)
from helpers import FakeEmbeddings, FakeLLM, make_pdf

S = Settings(google_api_key="test", min_similarity=0.2, retry_base_delay=0.0, max_retries=3)

PAGES = [
    "The capital of France is Paris. Paris sits on the river Seine and has the Eiffel Tower.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen in plants.",
]


def build(settings=S, llm=None, name="doc.pdf", pages=PAGES):
    res = extract_pages(make_pdf(pages), name)
    chunks = chunk_pages(res.pages, 300, 30)
    emb = FakeEmbeddings()
    p = RAGPipeline(settings, embeddings=emb, llm=llm or FakeLLM(), sleep=lambda s: None)
    p.index(chunks, res.file_hash)
    return p, emb


def test_retrieval_returns_relevant_chunk_with_metadata():
    p, _ = build()
    src = p.retrieve("What is the capital of France?", k=2)
    assert src[0].page == 1 and src[0].filename == "doc.pdf"
    assert "Paris" in src[0].text
    assert src[0].similarity > src[1].similarity  # ranked best-first


def test_answer_uses_retrieved_context_only():
    llm = FakeLLM()
    p, _ = build(llm=llm)
    ans = p.answer("What is the capital of France?")
    assert ans.answered and llm.calls == 1
    human = llm.last_messages[1][1]
    assert "Paris" in human and "[Source: doc.pdf, page 1]" in human
    assert "ONLY from the CONTEXT" in llm.last_messages[0][1]


def test_no_context_fallback_skips_llm():
    llm = FakeLLM()
    p, _ = build(replace(S, min_similarity=0.9), llm=llm)  # nothing can reach 0.9
    ans = p.answer("Who won the football world cup in 2010?")
    assert not ans.answered and ans.text == NO_ANSWER and ans.reason == "below_threshold"
    assert llm.calls == 0


def test_llm_declining_is_flagged():
    p, _ = build(llm=FakeLLM(NO_ANSWER))
    ans = p.answer("capital of France")
    assert not ans.answered and ans.reason == "llm_declined"


def test_empty_question_rejected():
    p, _ = build()
    with pytest.raises(RAGError):
        p.answer("   ")


def test_retrieve_before_indexing_fails_cleanly():
    p = RAGPipeline(S, embeddings=FakeEmbeddings(), llm=FakeLLM())
    with pytest.raises(VectorStoreError):
        p.retrieve("anything")


def test_new_document_never_returns_old_chunks():
    p, _ = build()
    res = extract_pages(make_pdf(["Quantum computers use qubits and superposition."]), "new.pdf")
    p.index(chunk_pages(res.pages, 300, 30), res.file_hash)
    srcs = p.retrieve("capital of France Paris", k=10)
    assert {s.filename for s in srcs} == {"new.pdf"}


def test_index_rejects_too_many_chunks_and_empty():
    p = RAGPipeline(replace(S, max_chunks=1), embeddings=FakeEmbeddings(), llm=FakeLLM())
    res = extract_pages(make_pdf(["word " * 500]), "a.pdf")
    with pytest.raises(VectorStoreError):
        p.index(chunk_pages(res.pages, 200, 20), res.file_hash)
    with pytest.raises(VectorStoreError):
        p.index([], "x")


# ---- retry / error handling ----
def test_retries_then_succeeds_with_exponential_backoff():
    delays, n = [], {"c": 0}
    def flaky():
        n["c"] += 1
        if n["c"] < 3:
            raise Exception("503 UNAVAILABLE")
        return "ok"
    assert with_retries(flaky, replace(S, retry_base_delay=1.0), delays.append) == "ok"
    assert delays == [1.0, 2.0]


def test_retries_are_bounded():
    n = {"c": 0}
    def always():
        n["c"] += 1
        raise Exception("429 RESOURCE_EXHAUSTED")
    with pytest.raises(RateLimitError):
        with_retries(always, S, lambda s: None)
    assert n["c"] == S.max_retries + 1


def test_auth_error_not_retried_and_key_not_leaked():
    n = {"c": 0}
    def bad():
        n["c"] += 1
        raise Exception("API key not valid: AIzaSySECRETKEY123")
    with pytest.raises(AuthError) as ei:
        with_retries(bad, S, lambda s: None)
    assert n["c"] == 1 and "AIza" not in str(ei.value)


def test_unknown_error_is_generic():
    retry, err = classify_error(Exception("weird internal /home/user/secret.py traceback"))
    assert not retry and isinstance(err, ServiceError) and "secret" not in str(err)


def test_llm_failure_never_fabricates_answer():
    class Boom:
        def invoke(self, m): raise Exception("503 unavailable")
    p, _ = build(llm=Boom())
    with pytest.raises(ServiceError):
        p.answer("capital of France")


def test_empty_llm_response_raises():
    p, _ = build(llm=FakeLLM(""))
    with pytest.raises(ServiceError):
        p.answer("capital of France")


def test_list_content_blocks_are_handled():
    p, _ = build(llm=FakeLLM([{"type": "text", "text": "Paris. [Source: doc.pdf, page 1]"}]))
    assert p.answer("capital of France").text.startswith("Paris")


def test_embedding_wrapper_batches_and_retries():
    class Flaky(FakeEmbeddings):
        failed = False
        def embed_documents(self, texts):
            if not Flaky.failed:
                Flaky.failed = True
                raise Exception("503 unavailable")
            return super().embed_documents(texts)
    emb = RetryingEmbeddings(Flaky(), replace(S, embed_batch_size=2), sleep=lambda s: None)
    assert len(emb.embed_documents(["a", "b", "c", "d", "e"])) == 5


def test_missing_api_key_message():
    from rag_pipeline import MissingAPIKeyError
    with pytest.raises(MissingAPIKeyError):
        build_llm(replace(S, google_api_key=""))
