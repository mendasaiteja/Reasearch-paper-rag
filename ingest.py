import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page_number": i + 1, "text": text.strip()})
    doc.close()
    return pages


def chunk_text(text, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    pages = extract_pages("ElevateBox_SWE_Challenge.pdf")
    print(f"Extracted {len(pages)} pages")

    first_page_text = pages[0]["text"]
    chunks = chunk_text(first_page_text)
    print(f"\nPage 1 split into {len(chunks)} chunks")
    print("\n--- Chunk 1 ---")
    print(chunks[0])
    if len(chunks) > 1:
        print("\n--- Chunk 2 ---")
        print(chunks[1])