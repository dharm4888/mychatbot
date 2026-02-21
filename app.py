# ==============================================
# app.py — RAG Chatbot + SQL Console (Cloud Ready)
# ==============================================

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from backend import db, models, auth, utils, rag, sql_tool
from backend.db import SessionLocal
from backend.models import Document

# ==============================================
# 1. Load Environment & Init DB
# ==============================================
load_dotenv()
models.Base.metadata.create_all(db.engine)

st.set_page_config(page_title="RAG Chatbot — SQL", layout="wide")

# ==============================================
# 2. Sidebar Navigation
# ==============================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Documents", "SQL Console", "Settings"])

# ==============================================
# 3. Auth / Token Verification
# ==============================================
auth.verify_query_param_token()
user_email = st.session_state.get("user_email", "guest")

# ==============================================
# 4. Chat Page — RAG
# ==============================================
if page == "Chat":
    st.title("💬 Chat (RAG)")
    st.markdown("Type your question — the bot will respond using your uploaded PDFs or general knowledge.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("Message", placeholder="Ask something about your PDFs...")
    with col2:
        top_k = st.number_input("Top-K", min_value=1, max_value=10, value=4)

    if user_query:
        with st.spinner("🤖 Thinking..."):
            try:
                result = rag.answer_query(user_query, top_k=top_k)
                rag_answer = result.get("answer", "")
                retrieved = result.get("retrieved", [])

                if not rag_answer or "couldn’t find relevant information" in rag_answer.lower():
                    rag_answer = utils.ask_openai(user_query)

                st.session_state.chat_history.append((user_query, rag_answer))

                st.markdown("### 🧠 Answer")
                st.write(rag_answer)

                if retrieved:
                    with st.expander("📚 Top-K Retrieved Passages"):
                        for i, doc in enumerate(retrieved, 1):
                            txt = getattr(doc, "page_content", str(doc))
                            src = getattr(doc, "metadata", {}).get("source", "unknown")

                            st.markdown(f"**{i}. Source:** {src}")
                            st.markdown(txt[:800] + "...")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**🧑‍💻 You:** {q}")
            st.markdown(f"**🤖 Bot:** {a}")
            st.divider()

# ==============================================
# 5. Documents Page
# ==============================================
elif page == "Documents":
    st.title("📄 Manage Documents")

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing and indexing..."):
            for file in uploaded_files:
                utils.save_and_index_pdf(file)
        st.success("✅ Documents indexed successfully!")

    with SessionLocal() as session:
        docs = session.query(Document).all()

        if docs:
            df_db = pd.DataFrame([
                {"ID": d.id, "Name": getattr(d, "filename", "unknown")}
                for d in docs
            ])
            st.dataframe(df_db)
        else:
            st.info("No documents in database.")

    if os.path.exists(utils.VECTOR_DIR):
        try:
            vectorstore = FAISS.load_local(
                utils.VECTOR_DIR,
                utils.embeddings,
                allow_dangerous_deserialization=True
            )
            faiss_docs = list(vectorstore.docstore._dict.values())

            if faiss_docs:
                df_faiss = pd.DataFrame([
                    {"Name": d.metadata.get("source", "unknown")}
                    for d in faiss_docs
                ])
                st.dataframe(df_faiss)
            else:
                st.info("No documents in FAISS.")
        except Exception as e:
            st.error(f"⚠️ FAISS Error: {e}")
    else:
        st.info("Upload PDFs to enable RAG search.")

# ==============================================
# 6. SQL Console Page
# ==============================================
elif page == "SQL Console":
    st.title("🧮 SQL Console")

    user_input = st.text_area("Enter SQL or question:")

    if st.button("Run"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            try:
                if user_input.strip().lower().startswith(
                        ("select", "insert", "update", "delete", "create", "drop")):
                    sql_query = user_input.strip()
                else:
                    with st.spinner("🧠 Converting to SQL..."):
                        sql_prompt = f"""
Convert this question into SQL for table 't_shirts'.
Return ONLY SQL:

{user_input}
"""
                        sql_query = utils.ask_openai(sql_prompt)

                        sql_query = (
                            sql_query
                            .replace("```sql", "")
                            .replace("```", "")
                            .replace("sql", "")
                            .strip()
                        )

                st.code(sql_query, language="sql")

                res = sql_tool.run_sql_query(sql_query)

                if res["status"] == "success":
                    df = res["data"]

                    if not df.empty:
                        st.success("✅ Success")

                        if df.shape[1] == 1:
                            values = df.iloc[:, 0].astype(str).tolist()
                            st.write(", ".join(values))
                        else:
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No data returned.")
                else:
                    st.error(res["error"])

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==============================================
# 7. Settings Page
# ==============================================
elif page == "Settings":
    st.title("⚙️ Settings")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )

    if st.button("Update API Key"):
        if openai_key.strip():
            os.environ["OPENAI_API_KEY"] = openai_key.strip()
            st.success("✅ API key updated for this session.")
        else:
            st.warning("⚠️ Please enter a valid key.")
