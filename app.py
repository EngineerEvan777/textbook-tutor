"""
Textbook Office-Hours Tutor (single-file FastAPI)

Render-friendly version:
- NO sentence-transformers / torch
- Uses OpenAI embeddings + FAISS (faiss-cpu)
"""

import os
import logging
import traceback
import json
import io
import re
import uuid
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from pypdf import PdfReader
from openai import OpenAI


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
DATA_DIR = Path("./data")
BOOK_DIR = DATA_DIR / "books"
SESSION_DIR = DATA_DIR / "sessions"
BOOK_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval params
TOP_K = 6
MAX_CONTEXT_CHARS = 12000
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Embeddings (OpenAI)
EMBED_MODEL_NAME = "text-embedding-3-small"  # small + cheap + good enough
EMBED_DIM = 1536  # for text-embedding-3-small


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
    title: str
    page_total: int
    chunks: List[Chunk]
    index: object          # faiss index
    embeddings: np.ndarray # shape (n, d), float32


BOOKS: Dict[str, BookIndex] = {}
SESSIONS: Dict[str, List[Dict[str, str]]] = {}  # session_id -> [{"role": "...", "content": "..."}]


# ----------------------------
# Helpers
# ----------------------------
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
        "emb": d / "embeddings.npy",
        "faiss": d / "index.faiss",
    }


def save_book_to_disk(book: BookIndex) -> None:
    faiss = get_faiss()
    p = book_paths(book.book_id)

    meta = {
        "book_id": book.book_id,
        "title": book.title,
        "page_total": book.page_total,
        "num_chunks": len(book.chunks),
        "embed_model": EMBED_MODEL_NAME,
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

    np.save(p["emb"], book.embeddings)
    faiss.write_index(book.index, str(p["faiss"]))


def load_book_from_disk(book_id: str) -> Optional[BookIndex]:
    p = book_paths(book_id, create=False)
    if not p["meta"].exists() or not p["chunks"].exists() or not p["emb"].exists() or not p["faiss"].exists():
        return None

    faiss = get_faiss()

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

    embs = np.load(p["emb"]).astype(np.float32)
    idx = faiss.read_index(str(p["faiss"]))

    return BookIndex(
        book_id=meta["book_id"],
        title=meta["title"],
        page_total=int(meta["page_total"]),
        chunks=chunks,
        index=idx,
        embeddings=embs,
    )


def save_session_to_disk(session_id: str, history: List[Dict[str, str]]) -> None:
    path = SESSION_DIR / f"{session_id}.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session_from_disk(session_id: str) -> List[Dict[str, str]]:
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


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

    # Keep batches reasonable
    BATCH = 64
    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
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


def llm_generate(prompt: str) -> Tuple[str, Dict[str, int]]:
    """
    Calls OpenAI and returns (answer_text, usage_dict).
    """
    try:
        client = get_openai_client()
        resp = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            timeout=60,
        )

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


def retrieve(book: BookIndex, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    q = embed_texts_openai([query])
    if book.embeddings.shape[0] == 0:
        return []
    scores, ids = book.index.search(q, top_k)
    hits: List[Tuple[Chunk, float]] = []
    for i, score in zip(ids[0], scores[0]):
        if i < 0:
            continue
        hits.append((book.chunks[int(i)], float(score)))
    return hits


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
    return """
<!DOCTYPE html>
<html>
<head>
<title>Textbook Tutor</title>
<style>
body { font-family: Arial; max-width: 900px; margin: 40px auto; }
textarea { width: 100%; height: 100px; }
button { padding: 8px 16px; margin-top: 10px; }
#answer { margin-top: 20px; padding: 10px; background: #f5f5f5; }
</style>
</head>

<body>

<h2>📘 Textbook Tutor</h2>

<h3>Upload a textbook</h3>
<input type="file" id="file"/>
<button onclick="upload()">Upload</button>

<h3>Ask a question</h3>
<textarea id="question" placeholder="Ask about the textbook..."></textarea>
<br>
<button onclick="ask()">Ask</button>

<div id="answer"></div>

<script>

let book_id = null;
let session_id = crypto.randomUUID();

async function upload() {

    const file = document.getElementById("file").files[0];
    if (!file) {
        alert("Please choose a PDF.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const r = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await r.json();

    if (data.book_id) {
        book_id = data.book_id;
        alert("Uploaded: " + data.title);
    } else {
        alert("Upload failed.");
    }
}

async function ask() {

    if (!book_id) {
        alert("Upload a book first.");
        return;
    }

    const q = document.getElementById("question").value;

    const r = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            book_id: book_id,
            session_id: session_id,
            question: q
        })
    });

    const data = await r.json();

    document.getElementById("answer").innerHTML =
        "<b>Answer:</b><br>" + data.answer +
        "<br><br><b>Citations:</b> " + (data.citations || "None");
}

</script>

</body>
</html>
"""


@app.get("/books")
def list_books():
    out = []
    for b in BOOKS.values():
        out.append({
            "book_id": b.book_id,
            "title": b.title,
            "page_total": b.page_total,
            "num_chunks": len(b.chunks),
        })
    out.sort(key=lambda x: x["title"].lower())
    return {"books": out}


@app.post("/upload")
async def upload(file: UploadFile = File(...), title: Optional[str] = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    MAX_PDF_MB = 80
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"PDF too large (max {MAX_PDF_MB} MB).")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    book_id = str(uuid.uuid4())
    book_title = (title or file.filename).strip() or "Untitled PDF"

    chunks: List[Chunk] = []
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
            chunks.append(Chunk(
                book_id=book_id,
                page_pdf=p + 1,
                page_total=num_pages,
                text=ch_text
            ))

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found. If this PDF is scanned, you’ll need OCR.")

    texts = [c.text for c in chunks]

    # Embeddings via OpenAI (fast + low RAM server-side)
    embs = embed_texts_openai(texts)
    idx = build_faiss_ip_index(embs)

    BOOKS[book_id] = BookIndex(
        book_id=book_id,
        title=book_title,
        page_total=num_pages,
        chunks=chunks,
        index=idx,
        embeddings=embs,
    )

    save_book_to_disk(BOOKS[book_id])

    return {
        "book_id": book_id,
        "title": book_title,
        "page_total": num_pages,
        "num_chunks": len(chunks),
    }


@app.post("/session/reset")
def reset_session(payload: Dict[str, str]):
    sid = (payload.get("session_id") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id.")
    SESSIONS[sid] = []
    path = SESSION_DIR / f"{sid}.json"
    if path.exists():
        path.unlink()
    usage_path = SESSION_DIR / f"{sid}_usage.json"
    if usage_path.exists():
        usage_path.unlink()
    return {"ok": True}


@app.get("/session/usage")
def get_session_usage(session_id: str):
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    usage_file = SESSION_DIR / f"{sid}_usage.json"
    if not usage_file.exists():
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        return json.loads(usage_file.read_text(encoding="utf-8"))
    except Exception:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


@app.post("/chat")
def chat(payload: Dict[str, str]):
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

    book = BOOKS[book_id]
    history = SESSIONS.get(session_id)
    if history is None:
        history = load_session_from_disk(session_id)

    page_filter = extract_page_filter(question)
    if page_filter:
        filtered = [(c, 1.0) for c in book.chunks if c.page_pdf == page_filter]
        hits = filtered[:TOP_K]
    else:
        hits = retrieve(book, question, TOP_K)

    prompt = build_prompt(question, hits, history)
    answer, usage = llm_generate(prompt)
    citations = format_citations(hits, answer)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    SESSIONS[session_id] = history
    save_session_to_disk(session_id, history)

    usage_file = SESSION_DIR / f"{session_id}_usage.json"
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
