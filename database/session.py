"""
Database Session & Connection Management Module
===============================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Configures database engine (SQLite default / PostgreSQL cloud support) and session factory.

Author: Senior Software Architect
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# Database Connection String (SQLite default, PostgreSQL via ENV)
DB_PATH = os.path.join("database", "retinax.db")
os.makedirs("database", exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# SQLite requires check_same_thread=False for async FastAPI worker threads
engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    print(f"[OK] Database initialized successfully at: '{DATABASE_URL}'")

def get_db():
    """FastAPI Dependency yield for scoped session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
