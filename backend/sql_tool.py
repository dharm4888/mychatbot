import pandas as pd
from sqlalchemy import text
from backend.db import engine

def run_sql_query(query: str):
    """
    Execute an SQL query and return results as a pandas DataFrame.

    Handles single-column queries (returns comma-separated list) and multi-column queries (table).
    Fixes SQLAlchemy key handling for new versions.
    """
    try:
        if not query or not query.strip():
            return {"status": "error", "error": "Empty query provided."}

        query = query.strip()
        if query.lower().startswith("sql "):
            query = query[4:]

        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            keys = list(result.keys())  # ✅ convert RMKeyView to list

            if not rows:
                df = pd.DataFrame(columns=keys)
            elif len(keys) == 1:
                # Single-column query → DataFrame with one column
                df = pd.DataFrame([r[0] for r in rows], columns=[keys[0]])
            else:
                # Multi-column query → DataFrame with all columns
                df = pd.DataFrame(rows, columns=keys)

        return {"status": "success", "data": df}

    except Exception as e:
        return {"status": "error", "error": f"SQL query failed: {e}"}
