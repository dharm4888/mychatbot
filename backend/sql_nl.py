# backend/sql_nl.py
import os
import openai
from dotenv import load_dotenv
from .sql_tool import run_query

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DB_SCHEMA = """
Table t_shirts:
- id (int)
- brand (varchar)
- color (varchar)
- size (varchar)
- price (float)
"""

def text_to_sql(nl_query: str):
    """
    Convert natural language to SQL using OpenAI
    """
    prompt = f"""
You are an expert SQL generator. 
Given a user's question and the database schema, generate a correct SELECT SQL query.
Only return the SQL query, no explanation.

Database schema:
{DB_SCHEMA}

User question: {nl_query}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert SQL generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        sql_query = response.choices[0].message.content.strip()
        return sql_query
    except Exception as e:
        return f"❌ LLM error: {e}"

def run_nl_query(nl_query: str):
    """
    Convert natural language to SQL and run it
    """
    sql_query = text_to_sql(nl_query)
    if sql_query.startswith("❌"):
        return sql_query
    try:
        results = run_query(sql_query)
        return results
    except Exception as e:
        return f"Query failed: {e}"
