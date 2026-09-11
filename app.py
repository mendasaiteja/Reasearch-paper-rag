import requests
from embed_store import retrieve

def build_prompt(question, chunks):
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask(question, top_k=3):
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )

    answer = response.json()["response"]
    return answer, chunks


if __name__ == "__main__":
    question = "How should the system handle two people editing the same document at the same time?"
    answer, sources = ask(question)

    print("\nSOURCES USED:")
    for s in sources:
        print(f"  - {s['paper_title']} [chunk {s['chunk_index']}] (distance={s['distance']:.4f})")
        print(f"    \"{s['text'][:100]}...\"")