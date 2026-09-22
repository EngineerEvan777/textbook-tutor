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
        return False
    logger.warning("Invalid %s=%r; using default %s", name, raw, default)
    return default


DATA_DIR = Path("./data")
BOOK_DIR = DATA_DIR / "books"
SESSION_DIR = DATA_DIR / "sessions"
BOOK_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval params
TOP_K = 6
MAX_CONTEXT_CHARS = 12000
CHUNK_TARGET_CHARS = int_env("CHUNK_TARGET_CHARS", 1800)
CHUNK_OVERLAP_CHARS = int_env("CHUNK_OVERLAP_CHARS", 100)
MIN_CHUNK_CHARS = int_env("MIN_CHUNK_CHARS", 80)
USE_BM25_RETRIEVAL = bool_env("USE_BM25_RETRIEVAL", True)
BM25_CANDIDATE_K = int_env("BM25_CANDIDATE_K", 30)

# Book limits
MAX_BOOKS_PER_USER = int(os.getenv("MAX_BOOKS_PER_USER", "3"))
MAX_PDF_MB = int_env("MAX_PDF_MB", 80)
MAX_VISUAL_PAGES = int_env("MAX_VISUAL_PAGES", 1)

# Embeddings (OpenAI)
EMBED_MODEL_NAME = "text-embedding-3-small"  # small + cheap + good enough
EMBED_DIM = 1536  # for text-embedding-3-small
EMBED_BATCH_SIZE = int_env("EMBED_BATCH_SIZE", 128)
BUILD_EMBEDDINGS_ON_UPLOAD = bool_env("BUILD_EMBEDDINGS_ON_UPLOAD", False)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    book_id: str
    page_pdf: int          # 1-indexed PDF page
    page_total: int
    text: str


@dataclass
class BookIndex:
    book_id: str
    owner_user_id: str
    owner_email: str
    title: str
    page_total: int
    chunks: List[Chunk]
    index: Optional[object]  # faiss index when embeddings are enabled
    embeddings: Optional[np.ndarray] = None


BOOKS: Dict[str, BookIndex] = {}
SESSIONS: Dict[str, List[Dict[str, str]]] = {}  # cache key = f"{user_id}:{session_id}"


# ----------------------------
# Auth helpers (Supabase)
# ----------------------------
def get_supabase_url() -> str:
    url = os.getenv("SUPABASE_URL", "").strip()
    if not url:
        raise HTTPException(status_code=500, detail="Server misconfigured: SUPABASE_URL is not set.")
    return url.rstrip("/")


def get_supabase_publishable_key() -> str:
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="Server misconfigured: SUPABASE_PUBLISHABLE_KEY is not set.")
    return key


def parse_bearer_token(authorization: Optional[str]) -> str:
    auth = (authorization or "").strip()
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return token


def get_current_user(authorization: Optional[str]) -> Dict[str, str]:
    token = parse_bearer_token(authorization)
    url = f"{get_supabase_url()}/auth/v1/user"
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": get_supabase_publishable_key(),
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        logger.error("Supabase auth request failed: %s", repr(e))
        raise HTTPException(status_code=502, detail="Could not verify login with Supabase.")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired login. Please sign in again.")

    data = resp.json()
    user_id = (data.get("id") or "").strip()
    email = (data.get("email") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Could not determine authenticated user.")

    return {"id": user_id, "email": email}


def get_session_cache_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def safe_user_key(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)


def session_file_path(user_id: str, session_id: str) -> Path:
    return SESSION_DIR / f"{safe_user_key(user_id)}__{session_id}.json"


def usage_file_path(user_id: str, session_id: str) -> Path:
    return SESSION_DIR / f"{safe_user_key(user_id)}__{session_id}_usage.json"


# ----------------------------
# Helpers
# ----------------------------
def normalize_query(query: str) -> str:
    q = query.strip()

    replacements = [
        (r"\b0th\b", "zeroth"),
        (r"\b1st\b", "first"),
        (r"\b2nd\b", "second"),
        (r"\b3rd\b", "third"),
        (r"\b4th\b", "fourth"),
        (r"\b5th\b", "fifth"),
        (r"\b6th\b", "sixth"),
        (r"\b7th\b", "seventh"),
        (r"\b8th\b", "eighth"),
        (r"\b9th\b", "ninth"),
        (r"\b10th\b", "tenth"),
    ]

    for pat, repl in replacements:
        q = re.sub(pat, repl, q, flags=re.IGNORECASE)

    return q

def extract_page_filter(question: str) -> Optional[int]:
    q = question.lower()
    patterns = [
        r"\bp\.?\s*(\d+)",
        r"\bpage\s*(\d+)",
        r"\bpdf\s*p\.?\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def book_paths(book_id: str, create: bool = True) -> Dict[str, Path]:
    d = BOOK_DIR / book_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "meta": d / "meta.json",
        "chunks": d / "chunks.jsonl",
        "bm25": d / "bm25.sqlite",
        "emb": d / "embeddings.npy",
        "faiss": d / "index.faiss",
        "pdf": d / "source.pdf",
    }


def build_bm25_index(book_id: str, chunks: List[Chunk]) -> bool:
    p = book_paths(book_id)
    db_path = p["bm25"]

    for suffix in ("", "-wal", "-shm"):
        old_path = Path(str(db_path) + suffix)
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception as e:
                logger.warning("Could not remove old BM25 index file %s: %s", old_path, repr(e))

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61')")
        conn.executemany(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            ((i + 1, chunk.text) for i, chunk in enumerate(chunks)),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.warning("Could not build BM25 index for book_id=%s: %s", book_id, repr(e))
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if conn:
            conn.close()


def ensure_bm25_index(book: BookIndex) -> bool:
    if not USE_BM25_RETRIEVAL:
        return False
    p = book_paths(book.book_id, create=False)
    if p["bm25"].exists():
        return True
    return build_bm25_index(book.book_id, book.chunks)


def save_book_to_disk(book: BookIndex) -> None:
    p = book_paths(book.book_id)

    meta = {
        "book_id": book.book_id,
        "owner_user_id": book.owner_user_id,
        "owner_email": book.owner_email,
        "title": book.title,
        "page_total": book.page_total,
        "num_chunks": len(book.chunks),
        "embed_model": EMBED_MODEL_NAME,
        "retrieval": "bm25+faiss" if book.index is not None else "bm25",
        "created_at": time.time(),
    }
    p["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with p["chunks"].open("w", encoding="utf-8") as f:
        for c in book.chunks:
            f.write(json.dumps({
                "book_id": c.book_id,
                "page_pdf": c.page_pdf,
                "page_total": c.page_total,
                "text": c.text,
            }, ensure_ascii=False) + "\n")

    if book.embeddings is not None:
        np.save(p["emb"], book.embeddings)

    build_bm25_index(book.book_id, book.chunks)

    if book.index is not None:
        faiss = get_faiss()
        faiss.write_index(book.index, str(p["faiss"]))
    else:
        for path in (p["emb"], p["faiss"]):
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass


def load_book_from_disk(book_id: str) -> Optional[BookIndex]:
    p = book_paths(book_id, create=False)
    if not p["meta"].exists() or not p["chunks"].exists():
        return None

    meta = json.loads(p["meta"].read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    with p["chunks"].open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(Chunk(
                book_id=obj["book_id"],
                page_pdf=int(obj["page_pdf"]),
                page_total=int(obj["page_total"]),
                text=obj["text"],
            ))
    
    idx = None
    if p["faiss"].exists():
        try:
            faiss = get_faiss()
            idx = faiss.read_index(str(p["faiss"]))
        except Exception as e:
            logger.warning("Could not load FAISS index for book_id=%s: %s", book_id, repr(e))

    book = BookIndex(
        book_id=meta["book_id"],
        owner_user_id=(meta.get("owner_user_id") or ""),
        owner_email=(meta.get("owner_email") or ""),
        title=meta["title"],
        page_total=int(meta["page_total"]),
        chunks=chunks,
        index=idx
    )

    ensure_bm25_index(book)
    return book


def save_session_to_disk(user_id: str, session_id: str, history: List[Dict[str, str]]) -> None:
    path = session_file_path(user_id, session_id)
    payload = {
        "owner_user_id": user_id,
        "session_id": session_id,
        "history": history,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session_from_disk(user_id: str, session_id: str) -> List[Dict[str, str]]:
    path = session_file_path(user_id, session_id)
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("history", [])


def split_into_chunks(text: str, target_chars: int, overlap_chars: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged: List[str] = []
    cur = ""

    for p in paras:
        if len(cur) + len(p) + 2 <= target_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                merged.append(cur)
            cur = p
    if cur:
        merged.append(cur)

    if overlap_chars > 0 and len(merged) > 1:
        out = []
        prev_tail = ""
        for chunk in merged:
            chunk2 = (prev_tail + chunk).strip()
            out.append(chunk2)
            prev_tail = chunk[-overlap_chars:]
        return out

    return merged


def embed_texts_openai(texts: List[str]) -> np.ndarray:
    """
    OpenAI embeddings, batched.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    client = get_openai_client()

    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBED_MODEL_NAME,
            input=batch,
        )
        # resp.data is in the same order as input
        for item in resp.data:
            all_vecs.append(item.embedding)

    embs = np.array(all_vecs, dtype=np.float32)

    # Normalize for cosine / inner product
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms
    return embs.astype(np.float32)


def build_faiss_ip_index(embs: np.ndarray):
    faiss = get_faiss()
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    if embs.shape[0] > 0:
        idx.add(embs)
    return idx


def llm_generate(prompt: str, images: Optional[List[Tuple[int, bytes]]] = None) -> Tuple[str, Dict[str, int]]:
    """
    Calls OpenAI and returns (answer_text, usage_dict).
    """
    try:
        client = get_openai_client()
        if images:
            content = [{"type": "input_text", "text": prompt}]
            for page_number, png in images:
                content.append({"type": "input_text", "text": f"Rendered PDF page {page_number} (cropped section):"})
                content.append({
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                    "detail": "high",
                })
            model_input = [{"role": "user", "content": content}]
        else:
            model_input = prompt
        resp = client.responses.create(model="gpt-5-mini", input=model_input, timeout=90 if images else 60)

        text = (resp.output_text or "").strip()

        usage = getattr(resp, "usage", None)
        if usage:
            usage_data = {
                "input_tokens": int(usage.input_tokens or 0),
                "output_tokens": int(usage.output_tokens or 0),
                "total_tokens": int(usage.total_tokens or 0),
            }
        else:
            usage_data = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        return text, usage_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error("OpenAI call failed: %s", repr(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail="LLM call failed. Check OPENAI_API_KEY and Render logs.",
        )


def format_citations(hits: List[Tuple[Chunk, float]], answer_text: str) -> str:
    seen = set()
    pages: List[str] = []
    for ch, _ in hits:
        plain = f"p. {ch.page_pdf} of {ch.page_total}"
        bracketed = f"[{plain}]"
        if (plain in answer_text or bracketed in answer_text) and plain not in seen:
            seen.add(plain)
            pages.append(plain)
    return ", ".join(pages[:3])


def build_prompt(question: str, hits: List[Tuple[Chunk, float]], history: List[Dict[str, str]]) -> str:
    ctx_parts = []
    used = 0
    for ch, _score in hits:
        tag = f"[p. {ch.page_pdf} of {ch.page_total}]"
        piece = f"{tag}\n{ch.text}\n"
        if used + len(piece) > MAX_CONTEXT_CHARS:
            break
        ctx_parts.append(piece)
        used += len(piece)

    context_block = "\n---\n".join(ctx_parts) if ctx_parts else "(No relevant textbook passages were retrieved.)"

    history_tail = history[-8:] if history else []
    convo = []
    for m in history_tail:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            convo.append(f"{role.upper()}: {content}")
    convo_block = "\n".join(convo) if convo else "(No prior conversation.)"

    return f"""
You are a helpful textbook office-hours tutor.

Rules:
- Use ONLY the provided textbook context to answer factual textbook questions.
- If the context is insufficient, say what is missing and ask a targeted follow-up question.
- Explain clearly and step-by-step when helpful.
- ALWAYS include citations in-line like [p. 73 of 1062] for key claims.
- Do NOT invent equations, figures, or page references.

CONVERSATION (recent):
{convo_block}

TEXTBOOK CONTEXT:
{context_block}

STUDENT QUESTION:
{question}

Answer:
""".strip()


def wants_visual_connections(question: str) -> bool:
    """Only inspect PDF pixels when the user asks about visual schedule links."""
    return bool(re.search(r"\b(arrow|arrows|connector|connectors|dependency|dependencies|predecessor|predecessors|flowchart|diagram|interdependencies)\b", question, re.I))


def visual_pages(book: BookIndex, question: str, hits: List[Tuple[Chunk, float]]) -> List[int]:
    explicit = extract_page_filter(question)
    if explicit is not None:
        return [explicit] if 1 <= explicit <= book.page_total else []

    # Prefer a page headed "Gantt Chart" over a textual task listing. The
    # retrieved pages remain candidates for other kinds of diagrams.
    candidates = set(ch.page_pdf for ch, _ in hits)
    candidates.update(ch.page_pdf for ch in book.chunks if "gantt chart" in ch.text.lower())
    ranked = sorted(candidates, key=lambda p: (
        max((ch.text.lower().count("gantt chart") for ch in book.chunks if ch.page_pdf == p), default=0),
        sum(ch.page_pdf == p for ch, _ in hits),
    ), reverse=True)
    return ranked[:MAX_VISUAL_PAGES]


def render_visual_page(pdf_path: Path, page_number: int) -> List[bytes]:
    """Trim empty margins and split tall charts into readable, overlapping images."""
    import fitz  # PyMuPDF
    from PIL import Image

    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_number - 1]
        # Bound raster memory for unusually large PDF page dimensions.
        scale = min(250 / 72, 4000 / max(page.rect.width, page.rect.height),
                    (10_000_000 / (page.rect.width * page.rect.height)) ** 0.5)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
        image = Image.frombytes("L", (pix.width, pix.height), pix.samples)

    # Ignore the running header when locating the document's main content.
    top = int(image.height * 0.10)
    ink = image.crop((0, top, image.width, image.height)).point(lambda x: 255 if x < 210 else 0)
    bbox = ink.getbbox()
    if bbox:
        left, upper, right, lower = bbox
        margin = 25
        image = image.crop((max(0, left - margin), max(0, top + upper - margin),
                            min(image.width, right + margin), min(image.height, top + lower + margin)))

    # Each piece retains its row labels and bars. An overly tall one-page
    # timeline would otherwise be shrunk until arrows become invisible.
    max_height = 1600
    step = 1450
    parts = []
    for y in range(0, image.height, step):
        piece = image.crop((0, y, image.width, min(image.height, y + max_height)))
        if piece.width > 2200:
            piece.thumbnail((2200, max_height))
        output = io.BytesIO()
        piece.save(output, format="PNG", optimize=True)
        parts.append(output.getvalue())
        if y + max_height >= image.height:
            break
    return parts[:3]

def keyword_bonus(query: str, text: str) -> float:
    q = query.lower()
    t = text.lower()
    bonus = 0.0

    # Strong phrase matches
    if "zeroth law" in q and "zeroth law" in t:
        bonus += 0.40
    if "first law" in q and "first law" in t:
        bonus += 0.40
    if "second law" in q and "second law" in t:
        bonus += 0.40
    if "third law" in q and "third law" in t:
        bonus += 0.40

    # Weaker word-level support
    if "zeroth" in q and "zeroth" in t:
        bonus += 0.20
    if "first" in q and "first" in t:
        bonus += 0.20
    if "second" in q and "second" in t:
        bonus += 0.20
    if "third" in q and "third" in t:
        bonus += 0.20

    if "law" in q and "law" in t:
        bonus += 0.10

    # Helpful thermo concept tied to zeroth law
    if "thermal equilibrium" in q and "thermal equilibrium" in t:
        bonus += 0.20
    elif "zeroth" in q and "thermal equilibrium" in t:
        bonus += 0.10

    return bonus


BM25_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "answer", "because",
    "before", "being", "between", "both", "could", "does", "doing", "each",
    "explain", "from", "have", "into", "like", "more", "most", "section",
    "should", "show", "simply", "studying", "textbook", "than", "that", "then",
    "there", "these", "they", "this", "those", "through", "what", "when", "where",
    "which", "while", "with", "would", "your",
}


def bm25_query_terms(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    terms: List[str] = []
    seen = set()

    for token in tokens:
        if len(token) < 2:
            continue
        if token in BM25_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)

    return terms[:20]


def bm25_match_query(query: str) -> str:
    terms = bm25_query_terms(query)
    if not terms:
        return ""
    return " OR ".join(f'"{term}"' for term in terms)


def retrieve_bm25(book: BookIndex, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if not ensure_bm25_index(book):
        return []

    match_query = bm25_match_query(query)
    if not match_query:
        return []

    p = book_paths(book.book_id, create=False)
    candidate_k = min(max(top_k * 4, BM25_CANDIDATE_K), len(book.chunks))
    conn = None

    try:
        conn = sqlite3.connect(str(p["bm25"]))
        rows = conn.execute(
            """
            SELECT rowid, bm25(chunks_fts) AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, candidate_k),
        ).fetchall()
    except sqlite3.Error as e:
        logger.warning("BM25 retrieval failed for book_id=%s: %s", book.book_id, repr(e))
        return []
    finally:
        if conn:
            conn.close()

    hits: List[Tuple[Chunk, float]] = []
    for rowid, rank in rows:
        idx = int(rowid) - 1
        if idx < 0 or idx >= len(book.chunks):
            continue
        chunk = book.chunks[idx]
        adjusted_score = -float(rank) + keyword_bonus(query, chunk.text)
        hits.append((chunk, adjusted_score))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:top_k]


def retrieve_keyword_fallback(book: BookIndex, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    terms = bm25_query_terms(query)
    if not terms:
        return []

    hits: List[Tuple[Chunk, float]] = []
    for chunk in book.chunks:
        text = chunk.text.lower()
        score = keyword_bonus(query, chunk.text)
        for term in terms:
            if term in text:
                score += text.count(term)
        if score > 0:
            hits.append((chunk, float(score)))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[:top_k]


def retrieve_faiss(book: BookIndex, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if book.index is None:
        return []

    q = embed_texts_openai([query])

    candidate_k = min(max(top_k * 3, 12), len(book.chunks))
    scores, ids = book.index.search(q, candidate_k)

    rescored: List[Tuple[Chunk, float]] = []
    for i, score in zip(ids[0], scores[0]):
        if i < 0:
            continue

        chunk = book.chunks[int(i)]
        adjusted_score = float(score) + keyword_bonus(query, chunk.text)
        rescored.append((chunk, adjusted_score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_k]


def retrieve(book: BookIndex, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if USE_BM25_RETRIEVAL:
        hits = retrieve_bm25(book, query, top_k)
        if hits:
            return hits

    hits = retrieve_faiss(book, query, top_k)
    if hits:
        return hits

    return retrieve_keyword_fallback(book, query, top_k)


# ----------------------------
# FastAPI app + UI
# ----------------------------
app = FastAPI(title="Textbook Office-Hours Tutor")


@app.on_event("startup")
def load_all_books():
    # Load any previously indexed books from disk
    for d in BOOK_DIR.iterdir():
        if d.is_dir():
            b = load_book_from_disk(d.name)
            if b:
                BOOKS[d.name] = b


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Textbook Tutor</title>
  <style>
    :root{
      --bg:#f6f7fb;
      --panel:#ffffff;
      --border:#d9d9e3;
      --text:#111;
      --muted:#667085;
      --accent:#2563eb;
      --accent2:#1d4ed8;
      --ok:#16a34a;
      --warn:#f59e0b;
      --bad:#dc2626;

      --shadow: 0 10px 25px rgba(17,24,39,.08);
      --radius: 16px;
    }

    *{box-sizing:border-box;}
    html,body{height:100%;}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      color:var(--text);
      background:linear-gradient(180deg,#f7f8ff 0%, var(--bg) 40%, #f4f6fb 100%);
    }

    .wrap{
      max-width: 1020px;
      margin: 0 auto;
      padding: 22px 18px 40px;
    }

    /* Header */
    .topbar{
      display:flex;
      gap:12px;
      align-items:center;
      justify-content:space-between;
      margin-bottom:14px;
    }
    .brand{
      display:flex;
      gap:12px;
      align-items:center;
    }
    .logo{
      width:38px;height:38px;
      border-radius:12px;
      background: radial-gradient(circle at 30% 30%, #60a5fa 0%, #2563eb 45%, #1e40af 100%);
      box-shadow: var(--shadow);
    }
    .brand h1{
      font-size: 22px;
      margin:0;
      letter-spacing:.2px;
    }
    .brand .sub{
      margin:2px 0 0;
      color:var(--muted);
      font-size:13px;
    }

    /* Cards / sections */
    .card{
      background:var(--panel);
      border:1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .card.pad{ padding: 16px; }

    .grid{
      display:grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }

    .row{
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      align-items:center;
    }

    label{
      color:var(--muted);
      font-size: 13px;
    }

    input, select, textarea, button{
      font: inherit;
      font-size: 14px;
    }

    input[type="text"], input[type="email"], input[type="password"], select{
      padding: 10px 12px;
      border:1px solid var(--border);
      border-radius: 12px;
      outline:none;
      background:#fff;
      min-height: 40px;
      transition: all 0.15s ease;
    }

    input[type="text"]:focus, 
    input[type="email"]:focus, 
    input[type="password"]:focus, 
    select:focus, 
    textarea:focus{
      border-color: rgba(37,99,235,.5);
      box-shadow: 0 0 0 4px rgba(37,99,235,.12);
    }

    textarea{
      width:100%;
      padding: 12px;
      border:1px solid var(--border);
      border-radius: 14px;
      outline:none;
      resize: none;
      min-height: 56px;
      max-height: 220px;
    }

    button{
      border:1px solid rgba(37,99,235,.25);
      background: linear-gradient(180deg, #2f6cf6 0%, var(--accent) 100%);
      color:#fff;
      padding: 10px 14px;
      border-radius: 12px;
      cursor:pointer;
      min-height: 40px;
      font-weight: 600;
      letter-spacing:.2px;
    }
    button:hover{ background: linear-gradient(180deg, #2b66ee 0%, var(--accent2) 100%); }
    button.secondary{
      background:#fff;
      color:var(--text);
      border:1px solid var(--border);
      font-weight:600;
    }
    button.secondary:hover{ background:#f9fafb; }
    button:disabled{
      opacity:.55;
      cursor:not-allowed;
    }

    .pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding: 8px 10px;
      border:1px solid var(--border);
      border-radius: 999px;
      background:#fff;
      color:var(--muted);
      font-size: 12px;
      white-space:nowrap;
    }
    .dot{ width:8px;height:8px;border-radius:999px;background:var(--warn); }
    .dot.ok{ background: var(--ok); }
    .dot.bad{ background: var(--bad); }

    details summary{
      list-style:none;
      cursor:pointer;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      user-select:none;
      font-weight: 800;
      font-size: 16px;
      padding: 14px 16px;
    }
    details summary::-webkit-details-marker{display:none;}
    details .content{
      border-top:1px solid var(--border);
      padding: 14px 16px 16px;
    }

    /* Upload status line */
    .statusLine{
      display:flex;
      gap:10px;
      align-items:center;
      color:var(--muted);
      font-size: 13px;
      margin-top:10px;
    }
    .spinner{
      width:14px;height:14px;
      border-radius:50%;
      border:2px solid rgba(37,99,235,.2);
      border-top-color: rgba(37,99,235,.9);
      animation: spin 0.9s linear infinite;
      display:none;
    }
    @keyframes spin{ to{ transform: rotate(360deg);} }

    /* Chat */
    .chatTitle{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      padding: 14px 16px;
      border-bottom:1px solid var(--border);
    }
    .chatTitle h2{ margin:0; font-size:16px; }
    .chat{
      padding: 14px 14px 6px;
      max-height: 58vh;
      overflow:auto;
    }

    .msg{
      max-width: 900px;
      margin: 10px 0;
      padding: 12px 12px;
      border:1px solid var(--border);
      border-radius: 14px;
      white-space: pre-wrap;
      line-height: 1.42;
      background:#fff;
    }
    .msg.user{
      margin-left:auto;
      background: #e9f0ff;
      border-color: rgba(37,99,235,.18);
    }
    .msg.assistant{
      margin-right:auto;
    }
    .meta{
      margin: 6px 2px 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .inputBar{
      display:flex;
      gap:10px;
      align-items:flex-end;
      padding: 12px 14px 14px;
      border-top:1px solid var(--border);
    }
    .kbdHelp{
      color: var(--muted);
      font-size: 12px;
      padding: 0 16px 14px;
    }

    /* Toast */
    .toast{
      position: fixed;
      right: 14px;
      bottom: 14px;
      background: #111827;
      color: #fff;
      padding: 12px 14px;
      border-radius: 14px;
      box-shadow: var(--shadow);
      max-width: 420px;
      display:none;
      z-index: 9999;
      font-size: 13px;
      line-height: 1.35;
    }
    .toast.ok{ background: #0f3d1f; }
    .toast.bad{ background: #4c1010; }
    .toast .tTitle{ font-weight:800; margin-bottom:4px; }
    .muted{ color: var(--muted); font-size: 13px; }

    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>

<body>
  <div style="margin:20px 0;padding:15px;border:1px solid #ccc;border-radius:8px;">
    <h3>Login or Create Account</h3>

    <div style="margin-bottom:10px;color:#555;font-size:14px;line-height:1.4;">
        New user? Enter your email and password, then click <b>Create Account</b>.
        Already have an account? Use <b>Login with Password</b>.
        Used magic link before? Log in, then click <b>Set / Update Password</b>.
    </div>

    <div class="row">
      <input
        type="email"
        id="loginEmail"
        placeholder="Enter your email"
        style="padding:8px;width:250px;margin-right:10px;"
      />

      <input
        type="password"
        id="loginPassword"
        placeholder="Enter your password"
        style="padding:8px;width:220px;margin-right:10px;"
      />

      <button id="loginBtn" onclick="loginWithPassword()" style="padding:8px 14px;">
        Login with Password
      </button>

      <button id="signupBtn" onclick="signUpWithPassword()" style="padding:8px 14px;margin-left:10px;">
        Create Account
      </button>

      <button id="setPasswordBtn" onclick="setPasswordForCurrentUser()" style="padding:8px 14px;margin-left:10px;">
        Set / Update Password
      </button>

      <button id="magicLinkBtn" onclick="sendMagicLink()" style="padding:8px 14px;margin-left:10px;">
        Email Me a Magic Link
      </button>

      <button id="logoutBtn" onclick="logout()" style="padding:8px 14px;margin-left:10px;">
        Logout
      </button>
    </div>

    <div id="loginStatus" style="margin-top:10px;color:#555;"></div>
  </div>

  <div id="appContent" class="wrap" style="display:none;">
    <div class="topbar">
      <div class="brand">
        <div class="logo"></div>
        <div>
          <h1>Textbook Office-Hours Tutor</h1>
          <div class="sub">Upload a PDF, pick the book, ask questions — answers include citations and page numbers.</div>
        </div>
      </div>
      <div class="pill" id="servicePill">
        <span class="dot" id="svcDot"></span>
        <span id="svcText">Ready</span>
      </div>
    </div>

    <div class="grid">
      <details class="card" id="setupPanel" open>
        <summary>
          <span>Setup (Upload / Book / Session)</span>
          <div class="row" style="margin-left:auto;">
            <button class="secondary" type="button" onclick="toggleSetup(event)">Hide setup</button>
          </div>
        </summary>

        <div class="content">
          <div class="card pad" style="box-shadow:none;">
            <div style="font-weight:800; margin-bottom:10px;">1) Upload textbook (PDF)</div>

            <div class="row">
              <input type="file" id="pdf" accept="application/pdf" />
              <input type="text" id="title" placeholder="Optional book title" style="min-width:260px; flex:1;" />
              <button id="uploadBtn" onclick="upload()">Upload</button>
            </div>

            <div class="statusLine">
              <div class="spinner" id="upSpin"></div>
              <div id="uploadStatus">No upload in progress.</div>
              <div class="muted" id="uploadTimer" style="margin-left:auto;"></div>
            </div>

            <div class="muted" style="margin-top:10px;">
              Tip: Large PDFs can take a bit to extract + index. If you’re on Render Free, keep the book under ~__MAX_PDF_MB__MB.
            </div>
          </div>

          <div style="height:12px;"></div>

          <div class="card pad" style="box-shadow:none;">
            <div style="font-weight:800; margin-bottom:10px;">2) Ask a question</div>

            <div class="row">
              <label>Book</label>
              <select id="bookSelect" style="min-width: 320px;"></select>

              <label style="margin-left:6px;">Session</label>
              <input id="sessionId" class="mono" style="width: 360px;" />

              <button class="secondary" onclick="newSession()">New session</button>
            </div>

            <div class="row" style="margin-top:10px;">
              <button class="secondary" id="refreshBtn" onclick="refreshBooks()">Refresh books</button>
              <button class="secondary" onclick="resetSession()">Reset session</button>
              <button class="secondary" id="deleteBookBtn" onclick="deleteSelectedBook()">Delete selected book</button>

              <div class="muted" id="usageLine" style="margin-left:auto;">Usage: (loading...)</div>
            </div>

            <div class="muted" style="margin-top:10px;">
              “Session” keeps a conversation thread. Share one session ID with a student for a single chat.
            </div>
          </div>
        </div>
      </details>

      <div class="card">
        <div class="chatTitle">
          <h2>Chat</h2>
          <div class="pill" id="chatPill">
            <span class="dot ok" id="chatDot"></span>
            <span id="chatText">Idle</span>
          </div>
        </div>

        <div id="chat" class="chat"></div>

        <div class="inputBar">
          <textarea id="q" placeholder="Ask anything like office hours..."></textarea>
          <button id="askBtn" onclick="ask()">Ask</button>
        </div>

        <div class="kbdHelp">Press Enter to send • Shift+Enter for a new line</div>
      </div>
    </div>
  </div>

  <div class="toast" id="toast">
    <div class="tTitle" id="toastTitle"></div>
    <div id="toastBody"></div>
  </div>

<script>

const SUPABASE_URL = __SUPABASE_URL__;
const SUPABASE_PUBLISHABLE_KEY = __SUPABASE_KEY__;

let sb = null;

try {
  console.log("SUPABASE_URL:", SUPABASE_URL);
  console.log("SUPABASE_PUBLISHABLE_KEY exists:", !!SUPABASE_PUBLISHABLE_KEY);

  sb = supabase.createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);

  sb.auth.onAuthStateChange(() => {
    showUser();
  });
} catch (err) {
  console.error("Supabase client init failed:", err);
  document.getElementById("loginStatus").textContent =
    "Error initializing login: " + (err.message || String(err));
}

async function setPasswordForCurrentUser() {
  const password = document.getElementById("loginPassword").value;
  const status = document.getElementById("loginStatus");

  if (!sb) {
    status.textContent = "Login is not initialized. Check Supabase settings.";
    return;
  }

  if (!password) {
    status.textContent = "Please enter a password first.";
    return;
  }

  if (password.length < 8) {
    status.textContent = "Password must be at least 8 characters.";
    return;
  }

  try {
    const { data } = await sb.auth.getSession();

    if (!data.session) {
      status.textContent = "Please log in with a magic link first, then click Set Password.";
      return;
    }

    status.textContent = "Setting password...";

    const { error } = await sb.auth.updateUser({
      password: password
    });

    if (error) {
      status.textContent = "Error: " + error.message;
      return;
    }

    document.getElementById("loginPassword").value = "";
    status.textContent = "Password set/updated successfully. You can now use Login with Password next time.";
  } catch (err) {
    console.error("setPasswordForCurrentUser failed:", err);
    status.textContent = "Error: " + (err.message || String(err));
  }
}

function setAuthUI(isLoggedIn) {
  const loginBtn = document.getElementById("loginBtn");
  const signupBtn = document.getElementById("signupBtn");
  const magicLinkBtn = document.getElementById("magicLinkBtn");
  const setPasswordBtn = document.getElementById("setPasswordBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const passwordInput = document.getElementById("loginPassword");

  if (isLoggedIn) {
    if (loginBtn) loginBtn.style.display = "none";
    if (signupBtn) signupBtn.style.display = "none";
    if (magicLinkBtn) magicLinkBtn.style.display = "none";
    if (logoutBtn) logoutBtn.style.display = "inline-block";
    if (setPasswordBtn) setPasswordBtn.style.display = "inline-block";
    if (passwordInput) passwordInput.placeholder = "Enter a new password";
  } else {
    if (loginBtn) loginBtn.style.display = "inline-block";
    if (signupBtn) signupBtn.style.display = "inline-block";
    if (magicLinkBtn) magicLinkBtn.style.display = "inline-block";
    if (logoutBtn) logoutBtn.style.display = "none";
    if (setPasswordBtn) setPasswordBtn.style.display = "none";
    if (passwordInput) passwordInput.placeholder = "Enter your password";
  }
}

async function showUser() {
  if (!sb) {
    document.getElementById("loginStatus").textContent =
      "Login is not initialized. Check Supabase settings.";
    document.getElementById("appContent").style.display = "none";
    setAuthUI(false);
    return;
  }

  const { data } = await sb.auth.getSession();

  if (data.session) {
    const email = data.session.user.email;

    document.getElementById("loginStatus").textContent =
      "Logged in as: " + email;

    document.getElementById("loginEmail").value = email || "";
    document.getElementById("appContent").style.display = "block";
    setAuthUI(true);
    refreshBooks();
    refreshUsage();
  } else {
    document.getElementById("loginStatus").textContent =
      "Please login to use Textbook Tutor.";

    document.getElementById("appContent").style.display = "none";
    setAuthUI(false);

    const sel = document.getElementById("bookSelect");
    if (sel) {
      sel.innerHTML = '';
      const opt = document.createElement('option');
      opt.value = "";
      opt.textContent = "Login to view your books";
      sel.appendChild(opt);
      sel.disabled = true;
    }
  }
}

async function sendMagicLink() {
  const email = document.getElementById("loginEmail").value.trim();
  const status = document.getElementById("loginStatus");

  if (!sb) {
    status.textContent = "Login is not initialized. Check Supabase settings.";
    return;
  }

  if (!email) {
    status.textContent = "Please enter your email first.";
    return;
  }

  status.textContent = "Sending login link...";

  try {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out after 15 seconds.")), 15000)
    );

    const otpPromise = sb.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: window.location.origin
      }
    });

    const result = await Promise.race([otpPromise, timeoutPromise]);
    const { error } = result;

    if (error) {
      status.textContent = "Error: " + error.message;
    } else {
      status.textContent = "Check your email for the login link!";
    }
  } catch (err) {
    console.error("sendMagicLink failed:", err);
    status.textContent = "Error: " + (err.message || String(err));
  }
}

async function loginWithPassword() {
  const email = document.getElementById("loginEmail").value.trim();
  const password = document.getElementById("loginPassword").value;
  const status = document.getElementById("loginStatus");

  if (!sb) {
    status.textContent = "Login is not initialized. Check Supabase settings.";
    return;
  }

  if (!email || !password) {
    status.textContent = "Please enter both email and password.";
    return;
  }

  status.textContent = "Logging in...";

  try {
    const { error } = await sb.auth.signInWithPassword({
      email,
      password
    });

    if (error) {
      if ((error.message || "").toLowerCase().includes("invalid login credentials")) {
        status.textContent = "Login failed. Double-check your email and password.";
      } else {
        status.textContent = "Error: " + error.message;
      }
    } else {
      status.textContent = "Logged in successfully.";
      document.getElementById("loginPassword").value = "";
      await showUser();
    }
  } catch (err) {
    console.error("loginWithPassword failed:", err);
    status.textContent = "Error: " + (err.message || String(err));
  }
}

async function signUpWithPassword() {
  const email = document.getElementById("loginEmail").value.trim();
  const password = document.getElementById("loginPassword").value;
  const status = document.getElementById("loginStatus");

  if (!sb) {
    status.textContent = "Login is not initialized. Check Supabase settings.";
    return;
  }

  if (!email || !password) {
    status.textContent = "Please enter both email and password.";
    return;
  }

  if (password.length < 8) {
    status.textContent = "Password must be at least 8 characters.";
    return;
  }

  status.textContent = "Creating account...";

  try {
    const { data, error } = await sb.auth.signUp({
      email,
      password
    });

    if (error) {
      const msg = (error.message || "").toLowerCase();

      if (msg.includes("user already registered")) {
        status.textContent = "This email already has an account. Use Login with Password, or log in with a magic link and click Set / Update Password.";
      } else {
        status.textContent = "Error: " + error.message;
      }
      return;
    }

    const { error: loginError } = await sb.auth.signInWithPassword({
      email,
      password
    });

    if (loginError) {
      status.textContent = "Account created. Now log in with your email and password.";
      return;
    }

    document.getElementById("loginPassword").value = "";
    status.textContent = "Account created and logged in successfully.";
    await showUser();
  } catch (err) {
    console.error("signUpWithPassword failed:", err);
    status.textContent = "Error: " + (err.message || String(err));
  }
}

async function logout() {
  if (!sb) {
    document.getElementById("loginStatus").textContent =
      "Login is not initialized. Check Supabase settings.";
    return;
  }

  await sb.auth.signOut();
  document.getElementById("loginStatus").textContent = "Logged out";
  document.getElementById("appContent").style.display = "none";
  document.getElementById("loginPassword").value = "";
  document.getElementById("loginEmail").value = "";
  setAuthUI(false);
}

async function requireLogin() {
  if (!sb) {
    alert("Login is not initialized. Check Supabase settings.");
    return false;
  }

  const { data } = await sb.auth.getSession();
  if (!data.session) {
    alert("Please login first.");
    return false;
  }

  return true;
}

async function authHeaders(extra = {}) {
  if (!sb) return { ...extra };

  const { data } = await sb.auth.getSession();
  const token = data?.session?.access_token;
  if (!token) return { ...extra };

  return {
    Authorization: `Bearer ${token}`,
    ...extra
  };
}

/* -------------------------
   Utilities / state
-------------------------- */
function toggleSetup(e){
  e.preventDefault();
  e.stopPropagation();
  const d = $("setupPanel");
  d.open = !d.open;
  const btn = e.target;
  btn.textContent = d.open ? "Hide setup" : "Show setup";
}

function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function $(id){ return document.getElementById(id); }

function setServiceState(state, text){
  const dot = $("svcDot");
  const t = $("svcText");
  dot.className = "dot" + (state === "ok" ? " ok" : state === "bad" ? " bad" : "");
  t.textContent = text || (state === "ok" ? "OK" : state === "bad" ? "Error" : "Working");
}

function setChatState(state, text){
  const dot = $("chatDot");
  const t = $("chatText");
  dot.className = "dot" + (state === "ok" ? " ok" : state === "bad" ? " bad" : "");
  t.textContent = text || (state === "ok" ? "Idle" : state === "bad" ? "Error" : "Working");
}

let toastTimer = null;
function toast(kind, title, body){
  const el = $("toast");
  el.className = "toast" + (kind === "ok" ? " ok" : kind === "bad" ? " bad" : "");
  $("toastTitle").textContent = title || "";
  $("toastBody").textContent = body || "";
  el.style.display = "block";
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.style.display = "none"; }, 3600);
}

function scrollChatToBottom() {
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text) {
  const chat = $("chat");
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chat.appendChild(div);
  scrollChatToBottom();
}

function addMeta(text) {
  const chat = $("chat");
  const div = document.createElement('div');
  div.className = 'meta';
  div.textContent = text;
  chat.appendChild(div);
  scrollChatToBottom();
}

function collapseSetupIfReady() {
  const d = $("setupPanel");
  const sel = $("bookSelect");
  if (d && sel && sel.options.length > 0) d.open = false;
}

function persist(){
  localStorage.setItem("tt_session_id", $("sessionId").value.trim());
  localStorage.setItem("tt_book_id", $("bookSelect").value || "");
}

function restore(){
  const sid = localStorage.getItem("tt_session_id");
  if (sid) $("sessionId").value = sid;

  // book id is applied after refreshBooks() loads options
}

/* -------------------------
   Books / usage
-------------------------- */
async function refreshBooks() {
  const btn = $("refreshBtn");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  try{
    setServiceState("work", "Refreshing books");
    const headers = await authHeaders();
    const r = await fetch('/books', { headers });
    const data = await r.json();

    if (!r.ok) {
      throw new Error(data.detail || 'Failed to load books');
    }

    const sel = $("bookSelect");
    const prev = localStorage.getItem("tt_book_id") || sel.value;

    sel.innerHTML = '';
    const books = (data.books || []);

    if (books.length === 0){
      const opt = document.createElement('option');
      opt.value = "";
      opt.textContent = "No books yet — upload a PDF above";
      sel.appendChild(opt);
      sel.disabled = true;
      setServiceState("ok", "Ready (no books)");
      return;
    }

    sel.disabled = false;
    books.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b.book_id;
      opt.textContent = `${b.title} (${b.page_total} pages)`;
      sel.appendChild(opt);
    });

    if (prev){
      const match = Array.from(sel.options).find(o => o.value === prev);
      if (match) sel.value = prev;
    }

    setServiceState("ok", "Ready");
    collapseSetupIfReady();
    persist();
  } catch(e){
    setServiceState("bad", "Error");
    toast("bad", "Couldn’t load books", String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh books";
  }
}

async function deleteSelectedBook() {
  if (!(await requireLogin())) return;

  const sel = $("bookSelect");
  const book_id = sel.value;

  if (!book_id) {
    toast("bad", "No book selected", "Choose a book first.");
    return;
  }

  const selectedText = sel.options[sel.selectedIndex]?.textContent || "this book";
  const ok = confirm(`Delete "${selectedText}"? This cannot be undone.`);
  if (!ok) return;

  const btn = $("deleteBookBtn");
  btn.disabled = true;
  btn.textContent = "Deleting…";

  try {
    const headers = await authHeaders({ 'Content-Type': 'application/json' });
    const r = await fetch('/books/delete', {
      method: 'POST',
      headers,
      body: JSON.stringify({ book_id })
    });

    const data = await r.json();

    if (!r.ok) {
      throw new Error(data.detail || "Failed to delete book");
    }

    toast("ok", "Book deleted", `"${selectedText}" was removed.`);
    await refreshBooks();
  } catch (e) {
    toast("bad", "Couldn’t delete book", String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Delete selected book";
  }
}

async function refreshUsage() {
  const sid = $("sessionId").value.trim();
  const el = $("usageLine");
  if (!sid) {
    el.textContent = 'Usage: (no session)';
    return;
  }

  try{
    const headers = await authHeaders();
    const r = await fetch('/session/usage?session_id=' + encodeURIComponent(sid), { headers });
    const data = await r.json();

    if (!r.ok) {
      throw new Error(data.detail || 'Failed to load usage');
    }

    const total = (data.total_tokens || 0).toLocaleString();
    const input = (data.input_tokens || 0).toLocaleString();
    const output = (data.output_tokens || 0).toLocaleString();

    el.textContent = `Session usage: ${total} tokens • ${input} in • ${output} out`;
  } catch(e){
    el.textContent = "Usage: (unavailable)";
  }
}

/* -------------------------
   Session controls
-------------------------- */
function newSession() {
  $("sessionId").value = uuidv4();
  $("chat").innerHTML = '';
  persist();
  refreshUsage();
  toast("ok", "New session created", "This session ID will keep your chat thread.");
}

async function resetSession() {
  const sid = $("sessionId").value.trim();
  if (!sid) return;
  try{
    const headers = await authHeaders({'Content-Type':'application/json'});
    await fetch('/session/reset', {
      method:'POST',
      headers,
      body: JSON.stringify({session_id: sid})
    });
    $("chat").innerHTML = '';
    refreshUsage();
    toast("ok", "Session reset", "Conversation memory cleared for this session.");
  } catch(e){
    toast("bad", "Couldn’t reset session", String(e));
  }
}

/* -------------------------
   Upload
-------------------------- */
let uploadT0 = null;
let uploadTick = null;

function startUploadTimer(){
  uploadT0 = Date.now();
  $("uploadTimer").textContent = "0.0s";
  uploadTick = setInterval(() => {
    const s = (Date.now() - uploadT0) / 1000;
    $("uploadTimer").textContent = s.toFixed(1) + "s";
  }, 100);
}
function stopUploadTimer(){
  if (uploadTick) clearInterval(uploadTick);
  uploadTick = null;
}

function setUploadUI(isUploading){
  $("uploadBtn").disabled = isUploading;
  $("pdf").disabled = isUploading;
  $("title").disabled = isUploading;
  $("upSpin").style.display = isUploading ? "inline-block" : "none";
  setServiceState(isUploading ? "work" : "ok", isUploading ? "Uploading/indexing" : "Ready");
}

async function upload() {
  if (!(await requireLogin())) return;
  
  const f = $("pdf").files[0];
  const title = $("title").value.trim();
  if (!f) {
    toast("bad", "No file selected", "Choose a PDF to upload.");
    return;
  }

  const fd = new FormData();
  fd.append('file', f);
  if (title) fd.append('title', title);

  $("uploadStatus").textContent = 'Uploading and indexing…';
  setUploadUI(true);
  startUploadTimer();

  try{
    const headers = await authHeaders();
    const r = await fetch('/upload', {method:'POST', body: fd, headers});
    const data = await r.json();

    if (!r.ok) {
      $("uploadStatus").textContent = 'Error: ' + (data.detail || 'upload failed');
      toast("bad", "Upload failed", data.detail || "Unknown error");
      return;
    }

    const secs = ((Date.now() - uploadT0) / 1000).toFixed(1);
    $("uploadStatus").textContent =
      `Uploaded: ${data.title} • ${data.page_total} pages • ${data.num_chunks} chunks (in ${secs}s)`;

    toast("ok", "Upload complete", `Indexed “${data.title}” in ${secs}s.`);
    await refreshBooks();
    collapseSetupIfReady();
  } catch(e){
    $("uploadStatus").textContent = "Error: " + String(e);
    toast("bad", "Upload error", String(e));
  } finally {
    stopUploadTimer();
    setUploadUI(false);
  }
}

/* -------------------------
   Chat ask
-------------------------- */
function setAskingState(isAsking) {
  $("askBtn").disabled = isAsking;
  $("q").disabled = isAsking;
  setChatState(isAsking ? "work" : "ok", isAsking ? "Asking…" : "Idle");
  $("askBtn").textContent = isAsking ? "Asking…" : "Ask";
}

async function ask() {
  if (!(await requireLogin())) return;

  const book_id = $("bookSelect").value;
  const session_id = $("sessionId").value.trim();
  const question = $("q").value.trim();

  if (!book_id) { toast("bad", "Pick a book first", "Upload/select a book before asking."); return; }
  if (!session_id) { toast("bad", "Missing session ID", "Click “New session” to generate one."); return; }
  if (!question) return;

  persist();

  addMessage('user', question);
  $("q").value = '';
  autoGrow();
  setAskingState(true);

  try{
    const headers = await authHeaders({'Content-Type':'application/json'});
    const r = await fetch('/chat', {
      method:'POST',
      headers,
      body: JSON.stringify({book_id, session_id, question})
    });
    const data = await r.json();
    setAskingState(false);

    if (!r.ok) {
      addMessage('assistant', '[ERROR] ' + (data.detail || 'chat failed'));
      toast("bad", "Chat failed", data.detail || "Unknown error");
      return;
    }

    addMessage('assistant', data.answer || '(no answer)');
    addMeta('CITATIONS: ' + (data.citations || '(none)'));

    if (data.usage) {
      const total = (data.usage.total_tokens || 0).toLocaleString();
      const input = (data.usage.input_tokens || 0).toLocaleString();
      const output = (data.usage.output_tokens || 0).toLocaleString();
      addMeta(`Session usage: ${total} tokens • ${input} in • ${output} out`);
    }

    refreshUsage();
  } catch(e){
    setAskingState(false);
    addMessage('assistant', '[ERROR] ' + String(e));
    toast("bad", "Network/app error", String(e));
  }
}

function autoGrow() {
  const ta = $("q");
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 220) + 'px';
}
$("q").addEventListener('input', autoGrow);

$("q").addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

/* Boot */
restore();
if (!$("sessionId").value.trim()) $("sessionId").value = uuidv4();
setServiceState("ok","Ready");
setChatState("ok","Idle");
showUser();
</script>
</body>
</html>
"""
    html = html.replace("__SUPABASE_URL__", json.dumps(os.getenv("SUPABASE_URL", "")))
    html = html.replace("__SUPABASE_KEY__", json.dumps(os.getenv("SUPABASE_PUBLISHABLE_KEY", "")))
    html = html.replace("__MAX_PDF_MB__", str(MAX_PDF_MB))
    return html


@app.get("/books")
def list_books(authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    out = []
    for b in BOOKS.values():
        if b.owner_user_id != user["id"]:
            continue
        out.append({
            "book_id": b.book_id,
            "title": b.title,
            "page_total": b.page_total,
            "num_chunks": len(b.chunks),
        })
    out.sort(key=lambda x: x["title"].lower())
    return {"books": out}

@app.post("/books/delete")
def delete_book(payload: Dict[str, str], authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    book_id = (payload.get("book_id") or "").strip()

    if not book_id:
        raise HTTPException(status_code=400, detail="Missing book_id.")

    if book_id not in BOOKS:
        raise HTTPException(status_code=404, detail="Book not found.")

    book = BOOKS[book_id]
    if book.owner_user_id != user["id"]:
        raise HTTPException(status_code=403, detail="You do not have access to delete this textbook.")

    # Remove from memory
    del BOOKS[book_id]

    # Remove from disk
    p = book_paths(book_id, create=False)
    if p["dir"].exists():
        for child in p["dir"].iterdir():
            try:
                child.unlink()
            except Exception:
                pass
        try:
            p["dir"].rmdir()
        except Exception:
            pass

    return {"ok": True, "deleted_book_id": book_id}

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    authorization: Optional[str] = Header(default=None),
):
    user = get_current_user(authorization)
    logger.info("Upload started for user_id=%s", user["id"])

    user_book_count = sum(
        1 for b in BOOKS.values()
        if b.owner_user_id == user["id"]
    )

    if user_book_count >= MAX_BOOKS_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"You can save up to {MAX_BOOKS_PER_USER} books. Delete one before uploading another."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    logger.info("Reading PDF")
    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"PDF too large (max {MAX_PDF_MB} MB).")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    logger.info("Extracting text")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    book_id = str(uuid.uuid4())
    book_title = (title or file.filename).strip() or "Untitled PDF"

    logger.info(
        "Splitting into chunks with target=%s overlap=%s min=%s",
        CHUNK_TARGET_CHARS,
        CHUNK_OVERLAP_CHARS,
        MIN_CHUNK_CHARS,
    )
    chunks: List[Chunk] = []
    seen_chunk_fingerprints = set()
    for p in range(num_pages):
        try:
            raw = reader.pages[p].extract_text() or ""
        except Exception:
            raw = ""
        raw = clean_text(raw)
        if not raw:
            continue

        page_chunks = split_into_chunks(raw, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
        for ch_text in page_chunks:
            ch_text = ch_text.strip()
            if len(ch_text) < MIN_CHUNK_CHARS:
                continue
            fingerprint = re.sub(r"\s+", " ", ch_text.lower())
            if fingerprint in seen_chunk_fingerprints:
                continue
            seen_chunk_fingerprints.add(fingerprint)
            chunks.append(Chunk(
                book_id=book_id,
                page_pdf=p + 1,
                page_total=num_pages,
                text=ch_text
            ))

    if not chunks:
        # Image-only diagrams can still be inspected on demand. This marker
        # makes the document selectable without pretending OCR was performed.
        chunks = [Chunk(book_id=book_id, page_pdf=p + 1, page_total=num_pages,
                        text="Visual-only PDF page; ask about a specific page to inspect it.")
                  for p in range(num_pages)]

    texts = [c.text for c in chunks]

    embs = None
    idx = None
    if BUILD_EMBEDDINGS_ON_UPLOAD:
        logger.info("Creating embeddings for %s chunks with batch size %s", len(texts), EMBED_BATCH_SIZE)
        embs = embed_texts_openai(texts)

        logger.info("Building FAISS index")
        idx = build_faiss_ip_index(embs)
    else:
        logger.info("Skipping embeddings on upload; BM25 retrieval will be used.")

    book = BookIndex(
        book_id=book_id,
        owner_user_id=user["id"],
        owner_email=user["email"],
        title=book_title,
        page_total=num_pages,
        chunks=chunks,
        index=idx,
        embeddings=None,
    )

    BOOKS[book_id] = book

    book.embeddings = embs
    save_book_to_disk(book)
    # Keep the original pages for visual questions in later sessions/restarts.
    pdf_path = book_paths(book_id)["pdf"]
    temporary_pdf = pdf_path.with_suffix(".pdf.tmp")
    temporary_pdf.write_bytes(pdf_bytes)
    temporary_pdf.replace(pdf_path)
    book.embeddings = None

    logger.info("Upload finished for user_id=%s book_id=%s", user["id"], book_id)

    return {"book_id": book_id, "title": book_title, "page_total": num_pages, "num_chunks": len(chunks)}


@app.post("/session/reset")
def reset_session(payload: Dict[str, str], authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    sid = (payload.get("session_id") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    cache_key = get_session_cache_key(user["id"], sid)
    SESSIONS[cache_key] = []

    path = session_file_path(user["id"], sid)
    if path.exists():
        path.unlink()

    usage_path = usage_file_path(user["id"], sid)
    if usage_path.exists():
        usage_path.unlink()

    return {"ok": True}


@app.get("/session/usage")
def get_session_usage(session_id: str, authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    usage_file = usage_file_path(user["id"], sid)
    if not usage_file.exists():
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        return json.loads(usage_file.read_text(encoding="utf-8"))
    except Exception:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def log_asked_question(user: Dict[str, str], question: str, authorization: Optional[str]) -> None:
    try:
        token = parse_bearer_token(authorization)

        url = f"{get_supabase_url()}/rest/v1/asked_questions"

        headers = {
            "apikey": get_supabase_publishable_key(),
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        payload = {
            "user_id": user["id"],
            "user_email": user["email"],
            "question": question,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=10)

        if resp.status_code >= 300:
            logger.warning("Question logging failed: %s %s", resp.status_code, resp.text)

    except Exception as e:
        logger.warning("Failed to log question: %s", repr(e))

@app.post("/chat")
def chat(payload: Dict[str, str], authorization: Optional[str] = Header(default=None)):
    user = get_current_user(authorization)

    book_id = (payload.get("book_id") or "").strip()
    session_id = (payload.get("session_id") or "").strip()
    question = (payload.get("question") or "").strip()

    if len(question) > 4000:
        raise HTTPException(status_code=400, detail="Question too long (max 4000 chars).")

    if len(question) < 2:
        raise HTTPException(status_code=400, detail="Please type a longer question.")

    if not book_id or book_id not in BOOKS:
        raise HTTPException(status_code=400, detail="Invalid or missing book_id.")

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    log_asked_question(user, question, authorization)

    book = BOOKS[book_id]
    if book.owner_user_id != user["id"]:
        raise HTTPException(status_code=403, detail="You do not have access to this textbook.")

    cache_key = get_session_cache_key(user["id"], session_id)
    history = SESSIONS.get(cache_key)
    if history is None:
        history = load_session_from_disk(user["id"], session_id)

    normalized_question = normalize_query(question)

    page_filter = extract_page_filter(question)
    if page_filter:
        filtered = [(c, 1.0) for c in book.chunks if c.page_pdf == page_filter]
        hits = filtered[:TOP_K]
    else:
        hits = retrieve(book, normalized_question, TOP_K)

    prompt = build_prompt(question, hits, history)
    images = []
    source_pdf = book_paths(book_id, create=False)["pdf"]
    if wants_visual_connections(question) and source_pdf.exists():
        pages = visual_pages(book, question, hits)
        for page_number in pages:
            try:
                images.extend((page_number, png) for png in render_visual_page(source_pdf, page_number))
            except Exception as e:
                logger.warning("Could not render PDF page %s for book %s: %s", page_number, book_id, repr(e))
        if images:
            prompt += "\n\nVISUAL PDF PAGES: " + ", ".join(f"p. {p} of {book.page_total}" for p in pages)
            prompt += """

The attached images are rendered crops of the PDF pages named above, in page order.
For every claimed arrow connection, name its source task and destination task,
the page, and what visible line/arrowhead establishes its direction. Separate
those directly visible links from plausible date-based inferences. A row being
earlier, adjacent, or nested does NOT by itself prove a dependency. If labels,
arrowheads, or endpoints are too small, crossing, or ambiguous, say so; do not
invent a formal predecessor. Do not claim to have seen pages that were not attached.
"""
    elif wants_visual_connections(question):
        prompt += "\n\nNo PDF page image was provided for visual inspection. You have text only. Do not claim to see arrows or offer image upload: this interface accepts PDFs. If the PDF was uploaded before visual support was added, ask for a PDF re-upload."
    answer, usage = llm_generate(prompt, images or None)
    citations = format_citations(hits, answer)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    SESSIONS[cache_key] = history
    save_session_to_disk(user["id"], session_id, history)

    usage_file = usage_file_path(user["id"], session_id)
    current = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if usage_file.exists():
        try:
            current = json.loads(usage_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    current["input_tokens"] += usage["input_tokens"]
    current["output_tokens"] += usage["output_tokens"]
    current["total_tokens"] += usage["total_tokens"]
    usage_file.write_text(json.dumps(current, indent=2), encoding="utf-8")

    return {
        "answer": answer,
        "citations": citations,
        "book_title": book.title,
        "book_id": book_id,
        "usage": usage,
    }
