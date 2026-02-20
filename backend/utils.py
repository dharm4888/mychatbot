# backend/utils.py
import os
from pathlib import Path
import smtplib
from email.message import EmailMessage
import pdfplumber
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================
# Environment & directories
# ==========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", "no-reply@example.com")

VECTOR_DIR = Path("vectorstore/faiss_index")
DOCS_DIR = Path("documents")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# OpenAI client & embeddings
# ==========================
client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ==========================
# Email Verification
# ==========================
def send_verification_email(to_email: str, token: str):
    """Send verification email with token."""
    verify_link = f"{FRONTEND_URL}/?verify_token={token}"
    msg = EmailMessage()
    msg["Subject"] = "Verify your email"
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg.set_content(f"Click to verify your email:\n\n{verify_link}\n\nToken: {token}")

    if not SMTP_HOST or not SMTP_USER:
        print(f"⚠️ SMTP not configured — dev mode:\n{msg}")
        return

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ Verification email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send verification email: {e}")

# ==========================
# Audio Transcription
# ==========================
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio file using OpenAI Whisper."""
    audio_file = Path(file_path)
    if not audio_file.exists() or not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        with audio_file.open("rb") as f:
            response = client.audio.transcriptions.create(model="whisper-1", file=f)
        return response.text
    except Exception as e:
        raise RuntimeError(f"❌ Transcription failed: {e}")

# ==========================
# OpenAI fallback
# ==========================
def ask_openai(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 600, temperature: float = 0.2) -> str:
    """Ask OpenAI model for a response (fallback when RAG has no context)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error in OpenAI fallback: {e}"

# ==========================
# PDF Extraction & FAISS indexing
# ==========================
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def save_and_index_pdf(uploaded_file):
    """
    Save uploaded PDF and index it into FAISS vectorstore.
    Splits long PDFs into chunks with metadata.
    """
    try:
        # Save PDF
        save_path = DOCS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text
        text = extract_text_from_pdf(save_path)
        if not text.strip():
            return {"status": "empty", "filename": uploaded_file.name}

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        # Create LangChain Documents
        docs = [
            Document(
                page_content=chunk,
                metadata={"filename": uploaded_file.name, "source": uploaded_file.name}
            )
            for chunk in chunks
        ]

        # Load existing FAISS or create new
        try:
            if (VECTOR_DIR / "index.faiss").exists():
                vectorstore = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(docs)
            else:
                vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(str(VECTOR_DIR))
        except Exception as e:
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(str(VECTOR_DIR))

        return {"status": "indexed", "filename": uploaded_file.name, "chunks": len(docs)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==========================
# RAG query helper
# ==========================
def build_context_snippet(results):
    """Build readable context snippet from retrieved FAISS chunks."""
    snippets = []
    for r in results:
        text = getattr(r, "page_content", r.get("text", ""))
        src = getattr(getattr(r, "metadata", {}), "get", lambda x, y=None: r.get("source", "unknown"))("filename", r.get("source", "unknown"))
        snippet = text[:1000] + "..." if len(text) > 1000 else text
        snippets.append(f"[Source: {src}]\n{snippet}")
    return "\n\n---\n\n".join(snippets)

def answer_query(user_query: str, top_k: int = 4):
    """Answer query using FAISS + RAG with OpenAI fallback."""
    from backend import vectorstore  # ensure import after FAISS setup

    try:
        results = vectorstore.search(user_query, top_k=top_k)
        if results:
            context = build_context_snippet(results)
            prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.
Use ONLY the following context to answer the user's question.
If the answer is not in the context, say:
"I couldn’t find relevant information in the uploaded documents."

Context:
{context}

User Question: {user_query}
"""
            answer = ask_openai(prompt)
            return {"answer": answer, "retrieved": results}
        else:
            # Fallback to general OpenAI response
            fallback_prompt = f"The user asked: '{user_query}'. No document context found. Answer using general knowledge."
            answer = ask_openai(fallback_prompt)
            return {"answer": answer, "retrieved": []}
    except Exception as e:
        return {"answer": f"❌ Error in RAG processing: {e}", "retrieved": []}
