"""
Textbook Office-Hours Tutor (single-file FastAPI)

What students can do:
- Upload one or more PDF textbooks
- Ask questions conversationally ("like office hours")
- Get immediate answers with citations like: p. 73 of 1062
- Switch which book they are asking about

Requirements (install once):
  python -m venv venv
  # Windows:
  venv\\Scripts\\activate
  # macOS/Linux:
  source venv/bin/activate

pip install fastapi uvicorn pypdf sentence-transformers faiss-cpu numpy python-multipart openai

Set env var:
  OPENAI_API_KEY=...

Run in terminal:
    python -m uvicorn app:app --reload

Open:
  http://127.0.0.1:8000
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
import faiss

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("textbook_tutor")


# ----------------------------
# OpenAI client
# ----------------------------
_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """
    Create the OpenAI client lazily (on first use) so the app doesn't crash
    at import/startup if the key is missing or wrong.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: OPENAI_API_KEY is not set. Set it in the terminal and restart the server.",
        )

    _client = OpenAI(api_key=api_key)
    return _client


# ----------------------------
# Config
# ----------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

# Embeddings
embedder = SentenceTransformer(EMBED_MODEL_NAME)


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
    index: faiss.IndexFlatIP
    embeddings: np.ndarray  # shape (n, d), float32


BOOKS: Dict[str, BookIndex] = {}
SESSIONS: Dict[str, List[Dict[str, str]]] = {}  # session_id -> [{"role": "...", "content": "..."}]


# ----------------------------
# Helpers
# ----------------------------
def extract_page_filter(question: str) -> Optional[int]:
    """
    Detect patterns like:
    - 'p. 32'
    - 'pdf p.32'
    - 'page 99'
    Returns the PDF page number if found.
    """
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
    """
    Chunk text by paragraphs, then merge to target size with overlap.
    Keeps chunks readable for Q&A.
    """
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

    # Add overlap by character tail
    if overlap_chars > 0 and len(merged) > 1:
        out = []
        prev_tail = ""
        for chunk in merged:
            chunk2 = (prev_tail + chunk).strip()
            out.append(chunk2)
            prev_tail = chunk[-overlap_chars:]
        return out

    return merged


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embs.astype(np.float32)


def build_faiss_ip_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs)
    return idx


def llm_generate(prompt: str) -> str:
    """
    Calls OpenAI without leaking internal errors to the browser.
    Detailed error info goes only to server logs.
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
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        else:
            usage_data = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        return text, usage_data

    except HTTPException:
        raise

    except Exception as e:
        logger.error("OpenAI call failed: %s", repr(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail="LLM call failed. Check your OPENAI_API_KEY and see server logs for details.",
        )


def format_citations(hits: List[Tuple[Chunk, float]], answer_text: str) -> str:
    """
    Return up to 3 unique pages that actually appear in the final answer text.
    Matches both:
      - p. X of Y
      - [p. X of Y]
    """
    seen = set()
    pages: List[str] = []
    for ch, _ in hits:
        plain = f"p. {ch.page_pdf} of {ch.page_total}"
        bracketed = f"[{plain}]"
        if (plain in answer_text or bracketed in answer_text) and plain not in seen:
            seen.add(plain)
            pages.append(plain)
    return ", ".join(pages[:3])


def build_prompt(
    question: str,
    hits: List[Tuple[Chunk, float]],
    history: List[Dict[str, str]],
) -> str:
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
    q = embed_texts([query])
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
    for d in BOOK_DIR.iterdir():
        if not d.is_dir():
            continue
        book_id = d.name
        b = load_book_from_disk(book_id)
        if b:
            BOOKS[book_id] = b


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Textbook Tutor</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d9d9e3;
      --text: #111;
      --muted: #666;
      --user: #e9f0ff;
      --assistant: #ffffff;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .container {
      max-width: 980px;
      width: 100%;
      margin: 0 auto;
      padding: 18px 18px 0 18px;
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
    }

    h2 { margin: 6px 0 8px; }
    hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
    .small { color: var(--muted); font-size: 13px; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    input, select, button, textarea { font-size: 14px; }
    input[type="text"] { padding: 6px 8px; }
    button { padding: 6px 10px; cursor: pointer; }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }

    /* Collapsible setup panel */
    details.panel summary {
      list-style: none;
      cursor: pointer;
      font-weight: bold;
      font-size: 18px;
    }
    details.panel summary::-webkit-details-marker { display: none; }
    details.panel summary:before { content: "▾ "; }
    details.panel[open] summary:before { content: "▴ "; }

    /* Chat layout */
    .chatWrap {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
      margin-top: 14px;
    }

    #chat {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
    }

    .msg {
      max-width: 860px;
      margin: 10px 0;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      white-space: pre-wrap;
      line-height: 1.35;
    }
    .msg.user { background: var(--user); margin-left: auto; }
    .msg.assistant { background: var(--assistant); margin-right: auto; }

    .metaLine {
      font-size: 12px;
      color: var(--muted);
      margin: 6px 0 0 4px;
    }

    /* Pinned input bar */
    .inputBar {
      margin-top: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }

    #q {
      flex: 1;
      resize: none;
      min-height: 44px;
      max-height: 180px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 12px;
      outline: none;
    }

    #askBtn { min-width: 90px; height: 44px; }

    .disabled {
      opacity: 0.6;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Textbook Office-Hours Tutor</h2>
    <div class="small">Upload a PDF, pick the book, ask questions, and you’ll get answers with citations.</div>

    <details class="panel" id="setupPanel" open>
      <summary>
        Setup (Upload / Book / Session)
        <span class="small" style="font-weight:normal;"> — click to expand/collapse</span>
      </summary>

      <div style="margin-top:12px;">
        <h3 style="margin:0 0 10px;">1) Upload textbook (PDF)</h3>
        <div class="row">
          <input type="file" id="pdf" accept="application/pdf"/>
          <input type="text" id="title" placeholder="Optional book title" style="min-width:260px;"/>
          <button onclick="upload()">Upload</button>
        </div>
        <div id="uploadStatus" class="small" style="margin-top:8px;"></div>

        <hr/>

        <h3 style="margin:0 0 10px;">2) Ask a question</h3>
        <div class="row">
          <label>Book:</label>
          <select id="bookSelect"></select>

          <label style="margin-left:8px;">Session:</label>
          <input id="sessionId" style="width: 320px;" />
          <button onclick="newSession()">New session</button>
        </div>

        <div class="row" style="margin-top:10px;">
          <button onclick="refreshBooks()">Refresh books</button>
          <button onclick="resetSession()">Reset session</button>
        </div>

        <div class="small" style="margin-top:10px;">
          Tip: “Session” keeps a conversation thread. Share one session ID with a student for a single chat.

          <div class="small" id="usageLine" style="margin-top:8px;">Usage: (loading...)</div>
        </div>
      </div>
    </details>

    <div class="chatWrap">
      <h3 style="margin:14px 0 8px;">Chat</h3>
      <div id="chat"></div>

      <div class="inputBar">
        <textarea id="q" placeholder="Ask anything like office hours..."></textarea>
        <button id="askBtn" onclick="ask()">Ask</button>
      </div>

      <div class="small" style="margin:8px 0 14px;">
        Press Enter to send • Shift+Enter for a new line
      </div>
    </div>
  </div>

<script>
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function scrollChatToBottom() {
  const chat = document.getElementById('chat');
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chat.appendChild(div);
  scrollChatToBottom();
}

function addMeta(text) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'metaLine';
  div.textContent = text;
  chat.appendChild(div);
  scrollChatToBottom();
}

function collapseSetup() {
  const d = document.getElementById('setupPanel');
  if (d) d.open = false;
}

async function refreshBooks() {
  const r = await fetch('/books');
  const data = await r.json();
  const sel = document.getElementById('bookSelect');
  sel.innerHTML = '';
  (data.books || []).forEach(b => {
    const opt = document.createElement('option');
    opt.value = b.book_id;
    opt.textContent = b.title + ' (' + b.page_total + ' pages)';
    sel.appendChild(opt);
  });

  if ((data.books || []).length > 0) {
    collapseSetup(); // auto-collapse once books exist
  }
}

async function refreshUsage() {
  const sid = document.getElementById('sessionId').value.trim();
  const el = document.getElementById('usageLine');
  if (!el) return;

  if (!sid) {
    el.textContent = 'Usage: (no session)';
    return;
  }

  const r = await fetch('/session/usage?session_id=' + encodeURIComponent(sid));
  const data = await r.json();

  const total = (data.total_tokens || 0).toLocaleString();
  const input = (data.input_tokens || 0).toLocaleString();
  const output = (data.output_tokens || 0).toLocaleString();

  el.innerHTML = `
    <div>Session usage: ${total} tokens processed</div>
    <div>${input} read from textbook/context • ${output} written in response</div>
  `;
}

function newSession() {
  document.getElementById('sessionId').value = uuidv4();
  document.getElementById('chat').innerHTML = '';
  refreshUsage();
}

async function resetSession() {
  const sid = document.getElementById('sessionId').value.trim();
  if (!sid) return;
  await fetch('/session/reset', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sid})
  });
  document.getElementById('chat').innerHTML = '';
  refreshUsage();
}

async function upload() {
  const f = document.getElementById('pdf').files[0];
  const title = document.getElementById('title').value.trim();
  if (!f) return;

  const fd = new FormData();
  fd.append('file', f);
  if (title) fd.append('title', title);

  document.getElementById('uploadStatus').textContent = 'Uploading and indexing...';
  const r = await fetch('/upload', {method:'POST', body: fd});
  const data = await r.json();

  if (!r.ok) {
    document.getElementById('uploadStatus').textContent = 'Error: ' + (data.detail || 'upload failed');
    return;
  }
  document.getElementById('uploadStatus').textContent =
    'Uploaded: ' + data.title + ' (book_id=' + data.book_id + ')';

  await refreshBooks();
  collapseSetup(); // collapse after successful upload
}

function setAskingState(isAsking) {
  const btn = document.getElementById('askBtn');
  const q = document.getElementById('q');
  if (isAsking) {
    btn.classList.add('disabled');
    btn.textContent = 'Asking...';
    q.classList.add('disabled');
  } else {
    btn.classList.remove('disabled');
    btn.textContent = 'Ask';
    q.classList.remove('disabled');
  }
}

async function ask() {
  const book_id = document.getElementById('bookSelect').value;
  const session_id = document.getElementById('sessionId').value.trim();
  const question = document.getElementById('q').value.trim();
  if (!book_id || !session_id || !question) return;

  addMessage('user', question);
  document.getElementById('q').value = '';
  autoGrow();
  setAskingState(true);

  const r = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({book_id, session_id, question})
  });

  const data = await r.json();
  setAskingState(false);

  if (!r.ok) {
    addMessage('assistant', '[ERROR] ' + (data.detail || 'chat failed'));
    return;
  }

  addMessage('assistant', data.answer || '(no answer)');

  addMeta('CITATIONS: ' + (data.citations || '(none)'));

  if (data.usage) {
    const total = data.usage.total_tokens.toLocaleString();
    const input = data.usage.input_tokens.toLocaleString();
    const output = data.usage.output_tokens.toLocaleString();

    addMeta(`Session usage: ${total} tokens processed`);
    addMeta(`${input} read from textbook/context • ${output} written in response`);
  }

  refreshUsage();
}

function autoGrow() {
  const ta = document.getElementById('q');
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
}

document.getElementById('q').addEventListener('input', autoGrow);

document.getElementById('q').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

newSession();
refreshBooks();
refreshUsage();
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
    embs = embed_texts(texts)
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
        raise HTTPException(status_code=400, detail="Question too long (max 4000 chars). Please shorten it.")
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

    # Step 1: enforce page restriction when requested
    page_filter = extract_page_filter(question)
    if page_filter:
        filtered_hits = [(c, 1.0) for c in book.chunks if c.page_pdf == page_filter]
        hits = filtered_hits[:TOP_K]
    else:
        hits = retrieve(book, question, TOP_K)

    prompt = build_prompt(question, hits, history)
    answer, usage = llm_generate(prompt)

    # Step 2: citation precision
    citations = format_citations(hits, answer)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    SESSIONS[session_id] = history
    save_session_to_disk(session_id, history)

    usage_file = SESSION_DIR / f"{session_id}_usage.json"

    if usage_file.exists():
        current = json.loads(usage_file.read_text(encoding="utf-8"))
    else:
        current = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

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