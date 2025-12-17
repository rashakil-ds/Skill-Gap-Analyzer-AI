import os
from pathlib import Path
from typing import Optional, Dict, List

#from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / ".chroma"

def _load_markdown_docs(folder: Path, doc_type: str) -> List[Document]:
    docs = []
    for p in sorted(folder.glob("*.md")):
        loader = TextLoader(str(p), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["type"] = doc_type
            d.metadata["source"] = p.name
        docs.extend(loaded)
    return docs

def _attach_skill_metadata(doc: Document) -> Document:
    # Expect a "Skills:" line in markdown for metadata extraction
    text = doc.page_content
    skills = ""
    for line in text.splitlines():
        if line.strip().lower().startswith("skills:"):
            skills = line.split(":", 1)[1].strip()
            break
    # normalize to pipe-separated
    if skills:
        skills = "|".join([s.strip() for s in skills.split(",") if s.strip()])
        doc.metadata["skills"] = skills
    return doc

def get_or_build_vectordb():
    embed_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    if CHROMA_DIR.exists():
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    role_docs = _load_markdown_docs(DATA_DIR / "roles", "role")
    playbook_docs = _load_markdown_docs(DATA_DIR / "playbooks", "playbook")

    all_docs = []
    for d in role_docs + playbook_docs:
        all_docs.append(_attach_skill_metadata(d))

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectordb.persist()
    return vectordb

def rag_retrieve(vectordb, query: str, k: int = 4, filters: Optional[Dict] = None):
    if filters:
        return vectordb.similarity_search(query, k=k, filter=filters)
    return vectordb.similarity_search(query, k=k)
