import os
from langchain_community.vectorstores import FAISS
from .utils import embeddings
from .utils import VECTOR_DIR
from .utils import ask_openai

def load_vectorstore():
    if not os.path.exists(VECTOR_DIR):
        return None
    try:
        return FAISS.load_local(
            VECTOR_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print("FAISS load error:", e)
        return None


def answer_query(query, top_k=4):
    vectorstore = load_vectorstore()

    if vectorstore is None:
        return {
            "answer": "I couldn’t find relevant information because no documents are indexed.",
            "retrieved": []
        }

    docs = vectorstore.similarity_search(query, k=top_k)

    if not docs:
        return {
            "answer": "I couldn’t find relevant information in the documents.",
            "retrieved": []
        }

    # Combine context
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a RAG assistant. Use the context below to answer the question.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
    """

    answer = ask_openai(prompt)

    return {
        "answer": answer,
        "retrieved": docs
    }
