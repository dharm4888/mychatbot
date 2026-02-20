# backend/db.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Fetch DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment")

# Create engine (MySQL)
engine = create_engine(DATABASE_URL, future=True)

# Session class
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class for ORM models
Base = declarative_base()

# Test connection
if __name__ == "__main__":
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DATABASE();"))  # ✅ use text()
            print("✅ Connected to database:", result.scalar())
    except Exception as e:
        print("❌ Database connection failed:", e)
