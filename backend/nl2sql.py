# backend/nl2sql.py

import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_SQL_MODEL = os.getenv("OPENAI_SQL_MODEL", "gpt-4o-mini")  # or "gpt-3.5-turbo"

openai.api_key = OPENAI_API_KEY

def generate_select_query(nl_question: str) -> str:
    """
    Convert natural language question into a safe SELECT SQL query.
    Ensures only SELECT statements are returned.
    """
    prompt = f"""
    Convert the following natural language request into a SQL SELECT query for the table `t_shirts`.
    Only generate SELECT statements. Never write INSERT, UPDATE, DELETE, or DROP.

    Request: "{nl_question}"

    Return only the SQL query.
    """

    try:
        response = openai.chat.completions.create(
            model=OPENAI_SQL_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant that only writes read-only SELECT SQL queries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=150,
        )
        sql_query = response.choices[0].message.content.strip()

        # Ensure it’s a SELECT statement
        if not sql_query.lower().startswith("select"):
            return "SELECT * FROM t_shirts LIMIT 0;"  # fallback safe query
        return sql_query
    except Exception as e:
        return f"SELECT * FROM t_shirts LIMIT 0;"  # safe fallback
