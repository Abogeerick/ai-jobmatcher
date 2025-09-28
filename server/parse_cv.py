import os
import pdfplumber
from docx import Document

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def extract_text_from_txt(path):
    """
    Read a plain text file.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def extract_text(path):
    """
    Convenience wrapper that detects file extension and calls the right extractor
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx": 
        return extract_text_from_docx(path) 
    elif ext == ".txt": 
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
