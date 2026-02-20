# ==============================================
# app.py — RAG Chatbot + Voice + SQL Console
# ==============================================
import os
import tempfile
import streamlit as st
import pandas as pd
import sounddevice as sd
import soundfile as sf
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

st.set_page_config(page_title="RAG Chatbot — Voice + SQL", layout="wide")

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
# 4. Voice Recorder Helper
# ==============================================
def record_audio(duration=10, samplerate=16000):
    st.info("🎙️ Recording... Speak now!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, recording, samplerate)
    st.success(f"✅ Recording complete ({duration}s)")
    st.audio(tmp.name)
    return tmp.name

# ==============================================
# 5. Chat Page — RAG + Voice
# ==============================================
if page == "Chat":
    st.title("💬 Chat (RAG + Real-Time Voice)")
    st.markdown("Type or speak your question — the bot will respond using your uploaded PDFs or general knowledge.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("Message", placeholder="Ask me something about your uploaded PDFs...")
    with col2:
        top_k = st.number_input("Top-K retrieved passages", min_value=1, max_value=10, value=4)

    # --- Voice Recording ---
    st.markdown("### 🎤 Speak to Chatbot")
    if st.button("Start Voice Recording"):
        audio_path = record_audio()
        st.session_state.audio_path = audio_path

    if st.button("Transcribe & Ask"):
        if "audio_path" not in st.session_state:
            st.warning("⚠️ No audio recorded yet. Please record first.")
        else:
            transcription = utils.transcribe_audio(st.session_state.audio_path)
            st.write(f"🗣️ You said: **{transcription}**")
            user_query = transcription

    # --- Process query ---
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
# 6. Documents Page — Upload & Index
# ==============================================
elif page == "Documents":
    st.title("📄 Manage Documents")

    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing and indexing documents..."):
            for file in uploaded_files:
                utils.save_and_index_pdf(file)
            st.success("✅ Documents uploaded and indexed successfully!")

    with SessionLocal() as session:
        docs = session.query(Document).all()
        if docs:
            df_db = pd.DataFrame([{"ID": d.id, "Name": getattr(d, "filename", getattr(d, "name", "unknown"))} for d in docs])
            st.markdown("### 🗂 Documents in Database")
            st.dataframe(df_db)
        else:
            st.info("No documents stored in database yet.")

    if os.path.exists(utils.VECTOR_DIR):
        try:
            vectorstore = FAISS.load_local(utils.VECTOR_DIR, utils.embeddings, allow_dangerous_deserialization=True)
            faiss_docs = list(vectorstore.docstore._dict.values())
            if faiss_docs:
                df_faiss = pd.DataFrame([{"Name": d.metadata.get("source", "unknown")} for d in faiss_docs])
                st.markdown("### 📚 Documents in FAISS Vectorstore")
                st.dataframe(df_faiss)
            else:
                st.info("No documents in FAISS vectorstore yet.")
        except Exception as e:
            st.error(f"⚠️ Error loading FAISS vectorstore: {e}")
    else:
        st.info("FAISS vectorstore not found. Upload PDFs to enable RAG search.")

# ==============================================
# # ==========================
# # ==========================
# # ==========================
## ==========================
# SQL Console Page
# ==========================
elif page == "SQL Console":
    st.title("🧮 SQL Console")
    st.markdown("Type an SQL query or ask in plain English — e.g. *'Show top 5 t-shirt brands'*")

    user_input = st.text_area("Enter your SQL query or question:")

    if st.button("Run SQL / Ask"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a query or question.")
        else:
            try:
                # Determine if user typed SQL directly
                if user_input.strip().lower().startswith(
                    ("select", "insert", "update", "delete", "create", "drop")
                ):
                    sql_query = user_input.strip()
                else:
                    # Convert natural language to SQL
                    with st.spinner("🧠 Translating your question to SQL..."):
                        sql_prompt = f"""
You are a SQL expert. Convert the following natural language question into a valid SQL query
for a table named 't_shirts'. Only return the SQL query:

Question: "{user_input}"
"""
                        sql_query = utils.ask_openai(sql_prompt)

                        # ===== FIX: Clean the LLM output =====
                        sql_query = sql_query.strip()              # remove whitespace
                        if sql_query.lower().startswith("sql"):
                            sql_query = sql_query[3:].strip()     # remove 'sql' prefix
                        # Remove backticks or triple quotes
                        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                # Show the generated SQL
                st.markdown("### 🧾 Generated SQL Query")
                st.code(sql_query, language="sql")

                # Execute the query
                res = sql_tool.run_sql_query(sql_query)

                # Display results
                if res["status"] == "success":
                    df = res["data"]
                    if not df.empty:
                        st.success("✅ Query executed successfully!")

                        if df.shape[1] == 1:
                            # Single-column → comma-separated list
                            col_name = df.columns[0]
                            values = df[col_name].dropna().astype(str).tolist()
                            st.markdown("🧠 Top Results:")
                            st.markdown(", ".join(values))
                        else:
                            # Multi-column → table
                            st.markdown("🧠 Top Results Table:")
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No data returned from query.")
                else:
                    st.error(f"❌ SQL Error: {res['error']}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")



# ==============================================
# 8. Settings Page
# ==============================================
elif page == "Settings":
    st.title("⚙️ Settings")
    st.markdown("Manage your chatbot and database configuration here.")

    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("✅ API key updated for this session.")
