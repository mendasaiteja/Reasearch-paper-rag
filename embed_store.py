import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(chunks, paper_id, paper_title):
    """
    chunks: a list of plain text strings (from ingest.py's chunk_text())
    paper_id: a short unique identifier for this document, e.g. "paper1"
    paper_title: a human-readable title, e.g. "LoRA Paper"
    """
    texts = chunks
    embeddings = model.encode(texts).tolist()

    ids = [f"{paper_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [{"paper_title": paper_title, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks from '{paper_title}'")


def retrieve(question, top_k=3):
    question_vector = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_vector,
        n_results=top_k
    )

    matches = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        matches.append({
            "text": text,
            "paper_title": meta["paper_title"],
            "chunk_index": meta["chunk_index"],
            "distance": distance
        })
    return matches


if __name__ == "__main__":
    from ingest import extract_pages, chunk_text

    pages = extract_pages("ElevateBox_SWE_Challenge.pdf")
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page["text"]))

    store_chunks(all_chunks, paper_id="elevatebox", paper_title="ElevateBox Challenge")

    results = retrieve("What should the app refuse to do?", top_k=3)
    for r in results:
        print(f"\n[{r['paper_title']}] distance={r['distance']:.4f}")
        print(r["text"][:200])