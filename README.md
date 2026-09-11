# Research Paper RAG Assistant

A command-line tool that lets you ask questions across one or more PDF
documents (research papers, technical docs, etc.) and get answers grounded
in the actual text, with chunk-level citations — powered by a local LLM.

## Why this needs RAG (not just a bigger prompt)

A single document can fit in an LLM's context window. A *library* of many
documents cannot — you'd blow past the context limit, and even if you had
a huge context window, re-sending every document on every question is slow
and expensive. This project indexes documents once, then retrieves only
the handful of chunks relevant to each specific question at query time.

## Architecture

```
data/*.pdf
    │
    ▼
ingest.py          -- extract text per page (PyMuPDF), split into
                       overlapping chunks (LangChain's
                       RecursiveCharacterTextSplitter)
    │
    ▼
embed_store.py      -- embed each chunk (Sentence Transformers,
                       all-MiniLM-L6-v2), store in a persistent
                       ChromaDB vector index with metadata
                       (paper title, chunk index)
    │
    ▼
rag_pipeline.py      -- embed the question, retrieve top-k nearest
                        chunks from ChromaDB, build a grounded prompt,
                        send it to Llama 3.2 via a local Ollama server
    │
    ▼
app.py              -- CLI entry point: index a folder of PDFs,
                       then ask questions in a loop
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull the model once:
ollama pull llama3.2
```

## Usage

Put one or more PDFs in `data/`, then:

```bash
python app.py index    # extract, chunk, embed, and store every PDF in data/
python app.py ask      # interactive Q&A loop over everything indexed so far
```

Example:

```
Q: How should the system handle two people editing the same document at the same time?

A: The system should reject stale writes with a conflict response, and show
   a clear message to the user, ensuring the final state stays correct.

Sources:
  - ElevateBox Challenge [chunk 15] (distance=0.4859)
  - ElevateBox Challenge [chunk 9]  (distance=0.5142)
  - ElevateBox Challenge [chunk 10] (distance=0.5277)
```

## Files

| File | Responsibility |
|---|---|
| `ingest.py` | PDF → page text → overlapping chunks. No embedding logic here. |
| `embed_store.py` | Chunks → embeddings → ChromaDB (`store_chunks`), question → top-k retrieval (`retrieve`). |
| `rag_pipeline.py` | Retrieved chunks → grounded prompt → Ollama call → answer + sources (`ask`). |
| `app.py` | CLI orchestration only — `index` and `ask` commands. No pipeline logic lives here. |

Each file maps to one stage of the pipeline, not one function — this keeps
the project readable without over-fragmenting it into many tiny files.

## Design decisions

**Chunk size (800 chars) + overlap (150 chars).**
Small chunks embed a focused idea; a huge chunk embeds as a blurry average
of everything in it, which retrieves badly. Overlap exists so an idea
sitting near a chunk boundary doesn't get sliced in half and lost from
both halves' embeddings — the shared text gives at least one chunk a
complete version of the idea.

**LangChain's `RecursiveCharacterTextSplitter` over raw character splitting.**
It tries paragraph breaks first, then line breaks, then sentence breaks,
and only falls back to splitting mid-word as a last resort — so chunk
boundaries land on natural text boundaries wherever possible.

**Cosine similarity for the vector index.**
Standard choice for sentence embeddings, since what matters is the
*direction* of the vector (meaning), not its magnitude.

**Page-level and chunk-level metadata on every stored chunk.**
This is what makes citations possible — an answer can be traced back to
a specific chunk index (and, with page-level ingestion, a specific page),
not just "somewhere in this document."

**The generation prompt explicitly instructs the model to say "not enough
information" rather than guess**, when the retrieved context doesn't
actually answer the question — this reduces hallucination compared to
just asking the LLM the question directly with no grounding.

**Local LLM (Ollama + Llama 3.2), not a hosted API.**
No document content leaves the machine — no API keys, no external calls.
This matters if the documents are sensitive, unpublished, or proprietary.

## What was tested, and how

- PDF extraction and chunking were verified against real PDFs, confirming
  page numbers are preserved and chunk boundaries don't silently drop text.
- ChromaDB storage and nearest-neighbor retrieval were verified with a
  small set of test sentences, confirming semantically related sentences
  ("How do I reset my password?" / "I forgot my login credentials") score
  as closer matches than an unrelated sentence, despite sharing almost no
  words in common.
- The full pipeline was run end-to-end against a real PDF, retrieving
  the correct section (concurrency handling) for a question that shares
  almost no vocabulary with that section's text — confirming retrieval is
  matching on meaning, not keyword overlap.

## Known limitations / what I'd improve with more time

- **No re-ranking step.** Retrieval currently returns the top-k nearest
  chunks by embedding distance alone. A cross-encoder re-ranking pass over
  a larger candidate set (e.g. top-20 → re-rank → top-4) would likely
  improve precision, at the cost of extra latency.
- **No cross-document synthesis prompt.** With multiple documents indexed,
  the model isn't explicitly instructed to compare/contrast across them —
  it just receives whichever chunks are closest, which may all come from
  one document even when a question spans several.
- **Chunking is character-count-based, not section-aware.** Splitting by
  detected headings (Abstract, Method, Results) instead would let retrieval
  be filtered or weighted by section type.
- **No incremental indexing.** Re-running `index` re-embeds everything in
  `data/` rather than only newly added files.
- **No UI.** This is deliberately a CLI-first project — the focus was the
  retrieval/generation pipeline itself, not a frontend.

## Tech stack

- **PyMuPDF (`fitz`)** — PDF text extraction
- **LangChain (`langchain-text-splitters`)** — chunking
- **Sentence Transformers (`all-MiniLM-L6-v2`)** — embeddings
- **ChromaDB** — vector storage and similarity search
- **Ollama + Llama 3.2** — local LLM generation