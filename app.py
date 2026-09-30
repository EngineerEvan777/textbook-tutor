"""
Textbook Office-Hours Tutor (single-file FastAPI)

Render-friendly version:
- NO sentence-transformers / torch
- Uses SQLite FTS5/BM25 by default for fast indexing
- Optional OpenAI embeddings + FAISS (faiss-cpu)
- Visual PDF questions require PyMuPDF and Pillow in requirements.txt
"""

import os
import logging
import traceback
import json
import io
import base64
import re
import sqlite3
import uuid
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple



import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header
from fastapi.responses import HTMLResponse, Response
from pypdf import PdfReader
from openai import OpenAI
import requests

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("textbook_tutor")


# ----------------------------
# OpenAI client (lazy)
# ----------------------------
_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: OPENAI_API_KEY is not set.",
        )
    _client = OpenAI(api_key=api_key)
    return _client


# ----------------------------
# FAISS (lazy import)
# ----------------------------
_FAISS = None

def get_faiss():
    global _FAISS
    if _FAISS is None:
        import faiss
        _FAISS = faiss
    return _FAISS


# ----------------------------
# Config
# ----------------------------
def int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    return value if value > 0 else default


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
