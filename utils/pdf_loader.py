"""
Utilitário para carregar e processar PDFs da pasta de artigos.
"""
import os
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai o texto completo de um arquivo PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_all_pdfs(pdf_folder: str) -> List[Dict[str, str]]:
    """
    Carrega todos os PDFs da pasta e retorna uma lista de dicts
    com 'title' (nome do arquivo) e 'content' (texto extraído).
    """
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            try:
                text = extract_text_from_pdf(pdf_path)
                if text.strip():
                    documents.append({
                        "title": filename.replace(".pdf", ""),
                        "content": text
                    })
            except Exception as e:
                print(f"⚠️ Erro ao processar {filename}: {e}")
    return documents


def split_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List:
    """
    Divide os documentos em chunks para indexação.
    Retorna uma lista de Document do LangChain.
    """
    from langchain_core.documents import Document as LCDocument

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    lc_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            lc_docs.append(LCDocument(
                page_content=chunk,
                metadata={
                    "source": doc["title"],
                    "chunk": i
                }
            ))
    return lc_docs
