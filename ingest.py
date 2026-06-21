"""
Script para processar todos os PDFs da pasta 'artigos' e criar
o índice vetorial (FAISS) que será usado pelo chatbot.

Execute uma vez (ou quando adicionar novos artigos):
    python ingest.py
"""
import os
import sys
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_loader import load_all_pdfs, split_documents
from utils.rag import get_embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("❌ OPENAI_API_KEY não encontrada. Configure no arquivo .env")
    sys.exit(1)

ARTIGOS_DIR = os.path.join(os.path.dirname(__file__), "artigos")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

print("📚 Carregando PDFs da pasta 'artigos'...")
docs = load_all_pdfs(ARTIGOS_DIR)
print(f"✅ {len(docs)} artigos carregados.")

print("✂️ Dividindo em chunks...")
chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
print(f"✅ {len(chunks)} chunks criados.")

print("🧠 Gerando embeddings e criando índice FAISS...")
embeddings = get_embeddings(API_KEY)
vectorstore = FAISS.from_documents(chunks, embeddings)

print("💾 Salvando vectorstore...")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
vectorstore.save_local(VECTORSTORE_DIR)

print(f"\n🎉 Pronto! Vectorstore salva em '{VECTORSTORE_DIR}'.")
print(f"   Agora execute: streamlit run app.py")
