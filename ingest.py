import fitz

def extract_pages(pdf_path):
    doc=fitz.open(pdf_path)
    pages=[]
    for i,page in enumerate(doc):
        text=page.get_text("text")
        if text.strip():
            pages.append({"page number":i+1,"text":text.strip()})
    doc.close()
    return pages

if __name__ == "__main__":
    pages = extract_pages("ElevateBox_SWE_Challenge.pdf")
    print(f"Extracted {len(pages)} pages")
    print(pages[1])  