# backend/vectorstore.py
import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ==========================
# Config
# ==========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

VECTORSTORE_PATH = Path("vectorstore/faiss_index")
LOCAL_EMBED_PATH = Path("vector_data")
LOCAL_EMBED_PATH.mkdir(exist_ok=True, parents=True)

EMBEDDING_FILE = LOCAL_EMBED_PATH / "embeddings.npy"
META_FILE = LOCAL_EMBED_PATH / "meta.json"

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

# ==========================
# Helpers
# ==========================
def chunk_text(text, chunk_size=1500, overlap=200):
    """Split text into overlapping chunks for embeddings."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str):
    text = text.strip()
    if not text:
        return None
    response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)

def _load():
    if EMBEDDING_FILE.exists() and META_FILE.exists():
        embs = np.load(EMBEDDING_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return embs, meta
    return None, []

def _save(embs, meta):
    np.save(EMBEDDING_FILE, embs)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ==========================
# Add PDF to FAISS + local backup
# ==========================
def add_pdf_to_faiss(pdf_path: str):
    """
    Load PDF, split into chunks, embed, and save to FAISS + local backup.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        filename = Path(pdf_path).name
        for d in docs:
            d.metadata["filename"] = filename
            d.metadata["source"] = filename

        # Create or append FAISS
        if VECTORSTORE_PATH.exists():
            db = FAISS.load_local(str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
        else:
            db = FAISS.from_documents(docs, embeddings)

        db.save_local(str(VECTORSTORE_PATH))

        # Also save to local fallback
        add_texts([
            {"id": filename, "title": filename, "content": d.page_content, "source": filename}
            for d in docs
        ])
        return f"✅ PDF indexed: {filename}, chunks: {len(docs)}"
    except Exception as e:
        return f"❌ Error adding PDF: {e}"

# ==========================
# Add raw text to local fallback
# ==========================
def add_texts(items: list):
    """Add plain text chunks to local numpy+json fallback store."""
    embs, meta = _load()
    new_embs = []

    for it in items:
        chunks = chunk_text(it["content"])
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            if emb is not None:
                new_embs.append(emb)
                meta.append({
                    "id": f"{it.get('id')}_part{i}",
                    "title": it.get("title"),
                    "source": it.get("source"),
                    "text": chunk
                })

    if not new_embs:
        return

    new_embs = np.vstack(new_embs).astype(np.float32)
    if embs is None:
        embs = new_embs
    else:
        embs = np.vstack([embs, new_embs])
    _save(embs, meta)

# ==========================
# Search FAISS + fallback
# ==========================
def search(query: str, top_k: int = 4):
    """Search query in FAISS first, fallback to local numpy+json if FAISS fails."""
    # Try FAISS
    try:
        if VECTORSTORE_PATH.exists():
            db = FAISS.load_local(str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True)
            results = db.similarity_search_with_score(query, k=top_k)
            return [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("filename", "unknown"),
                    "title": doc.metadata.get("title", ""),
                    "score": float(score)
                }
                for doc, score in results
            ]
    except Exception as e:
        print(f"⚠️ FAISS search failed, using fallback: {e}")

    # Fallback: local numpy+json
    embs, meta = _load()
    if embs is None or not meta:
        return []

    qemb = get_embedding(query).astype(np.float32)
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    qnorm = qemb / np.linalg.norm(qemb)
    sims = np.dot(embs_norm, qnorm)
    idxs = np.argsort(-sims)[:top_k]

    results = []
    for i in idxs:
        results.append({"score": float(sims[i]), **meta[i]})
    return results
