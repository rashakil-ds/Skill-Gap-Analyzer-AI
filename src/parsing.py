from io import BytesIO
from docx import Document
import pdfplumber

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def extract_text_from_upload(uploaded_file) -> str:
    b = uploaded_file.read()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)
    if name.endswith(".docx"):
        return extract_text_from_docx(b)
    raise ValueError("Unsupported file type. Please upload .pdf or .docx.")
