# backend/auth.py

import os, secrets
from datetime import datetime, timedelta
import streamlit as st

from sqlalchemy.orm import Session
from jose import jwt, JWTError
from passlib.context import CryptContext

from .db import Base, engine, SessionLocal
from .utils import send_verification_email
from .models import User

# -----------------------------
# Config
# -----------------------------
SECRET_KEY = os.getenv("JWT_SECRET", "change_me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60 * 24 * 7))  # 7 days

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------------
# Create DB tables
# -----------------------------
Base.metadata.create_all(bind=engine)

# -----------------------------
# Utility functions
# -----------------------------
def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    """Decode JWT token safely"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# -----------------------------
# Auth functions
# -----------------------------
def register_user(email: str, password: str):
    db: Session = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            return {"error": "Email already registered"}

        hashed = hash_password(password)
        token = secrets.token_urlsafe(16)

        new_user = User(
            email=email,
            hashed_password=hashed,
            verification_token=token,
            is_verified=False
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        send_verification_email(email, token)
        return {"message": "User registered. Verification email sent."}
    finally:
        db.close()

def verify_user(token: str):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.verification_token == token).first()
        if not user:
            return {"error": "Invalid token"}
        user.is_verified = True
        user.verification_token = None
        db.commit()
        return {"message": "Email verified successfully"}
    finally:
        db.close()

def login_user(email: str, password: str):
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return {"error": "User not found"}
        if not user.is_verified:
            return {"error": "Email not verified"}
        if not verify_password(password, user.hashed_password):
            return {"error": "Incorrect password"}

        token = create_access_token({"sub": email})
        st.session_state["user"] = {"email": email, "access_token": token}
        return {"access_token": token, "token_type": "bearer"}
    finally:
        db.close()

# -----------------------------
# Streamlit helper
# -----------------------------
def get_current_user():
    """Return the current logged-in user info from Streamlit session."""
    return st.session_state.get("user", None)

# -----------------------------
# Email verification via Streamlit URL (?verify_token=...)
# -----------------------------
def verify_query_param_token():
    """Check Streamlit URL (?verify_token=...) and mark user as verified."""
    try:
        params = st.query_params  # ✅ Streamlit >= 1.34 API
        token = params.get("verify_token")
        if token:
            if isinstance(token, list):
                token = token[0]

            payload = decode_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    with SessionLocal() as session:
                        user = session.query(User).filter(User.email == email).first()
                        if user and not user.is_verified:
                            user.is_verified = True
                            session.add(user)
                            session.commit()
                            st.success("✅ Your email has been verified successfully.")
                        else:
                            st.info("✅ Already verified or invalid user.")
    except Exception as e:
        st.warning(f"⚠️ Token verification skipped: {e}")
