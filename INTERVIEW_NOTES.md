# Interview Notes — DocuMind AI

## 60-second architecture explanation
"DocuMind is a RAG app. When a PDF is uploaded I validate it, extract text per page with PyMuPDF, and split it with RecursiveCharacterTextSplitter (1000 chars, 150 overlap) while keeping filename and page number on every chunk. Each chunk is embedded with Gemini's embedding model and stored in a per-document ChromaDB collection using cosine distance. When the user asks a question, I embed it, fetch the top-k chunks, and check the best similarity against a threshold. If it is too low, I return a fixed fallback without calling the LLM. Otherwise only those chunks go to Gemini with a strict prompt: answer only from context, cite pages, say when unknown. API calls use bounded exponential backoff, errors are mapped to safe messages, and the UI shows the retrieved chunks so answers can be verified."

```
PDF -> extract(page) -> chunk(+metadata) -> embed -> Chroma(cosine)
Question -> embed -> top-k -> threshold gate -> [no] fallback / [yes] Gemini(context only) -> answer + citations
```

## Test scenarios and expected results
| Scenario | Expected |
|---|---|
| Valid text PDF | Indexed, chat enabled |
| Scanned / encrypted / corrupt / .txt / empty | Specific error, chat disabled |
| In-document question | Answer + `[Source: file, page N]` |
| Out-of-document question | Fallback sentence |
| Empty question | Rejected |
| Upload second PDF | Old chunks gone, chat cleared |
| Gemini 429/503 | Up to 4 retries (1s, 2s, 4s, 8s), then friendly error |
| Invalid key | No retry, "key invalid" message, key never shown |

## 10 likely questions and answers
1. **What is RAG and why use it here?** Retrieve relevant text first, then generate from it. It grounds answers in the PDF, avoids context-window limits, lowers cost, and lets me show evidence.
2. **What is an embedding?** A fixed-length vector from a model where semantically similar text sits close together, so search becomes nearest-neighbour math instead of keyword matching.
3. **Cosine vs Euclidean?** Cosine compares direction (angle) and ignores length; Euclidean is straight-line distance and is affected by length. On unit-normalised vectors their rankings match; otherwise they can differ (see `docs/similarity_demo.py`).
4. **How does Chroma rank results?** HNSW approximate nearest-neighbour index; with `hnsw:space=cosine` it returns distance = 1 − similarity and sorts ascending. Approximate, so tiny recall loss is possible at scale.
5. **How did you choose chunk size/overlap?** 1000/150 is a common starting point: big enough for context, small enough for focused embeddings. Overlap protects facts at boundaries. Both are configurable and should be tuned per document type.
6. **Retrieval error vs generation error?** Retrieval: right chunk not fetched. Generation: right chunk fetched but answer wrong. I debug by inspecting the retrieved chunks first; if the answer is there, it is a prompt/LLM problem.
7. **How do you prevent hallucinations?** Layers: retrieval threshold gate (LLM not called on weak matches), strict prompt, temperature 0, citations, untrusted-context instruction, and no fabricated answer on API failure. None is a guarantee.
8. **What are the limits of the threshold?** Scores depend on embedding model, metric and corpus, so the default (0.45) is uncalibrated. To calibrate: build a labelled set of answerable and unanswerable questions, plot best-similarity for each, choose a cut that balances false refusals and false answers.
9. **How do you handle API failures?** Classify the error; retry only transient ones (429/5xx/timeouts) with exponential backoff and a hard cap; auth errors fail immediately; user sees a generic message, logs hold details.
10. **What would you improve next?** OCR, hybrid BM25+vector search, reranking, query rewriting using chat history, persistent multi-doc store, and an evaluation set measuring faithfulness and retrieval recall.
