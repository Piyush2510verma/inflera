import fitz  # PyMuPDF

def extract_text_from_pdf_path(pdf_path: str) -> str:
    # Open the PDF from the given path
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        # Consider removing or commenting out these prints for cleaner logs in production
        # print(f"\n--- Page {page_num} ---\n")
        # print(page_text)
        text += page_text
    doc.close()
    return text
