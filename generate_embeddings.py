#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate embeddings.npy and meta.jsonl from data/data.txt for SmartMind chatbot.

Saves:
  data/index/embeddings.npy  (N x D, float32, row-normalized)
  data/index/meta.jsonl      (one JSON per line; fields: doc_id, title, page_number, chunk_id, text)

Usage:
  python generate_embeddings.py
"""

import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# OpenAI client import style used in the app
from openai import OpenAI

# ----------------- CONFIG -----------------
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

TXT_PATH = PROJECT_ROOT / "data.txt"           # change if your file has another name
META_PATH = INDEX_DIR / "meta.jsonl"
EMB_PATH = INDEX_DIR / "embeddings.npy"

# Embedding model & size (match your app)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = 3072

# batching & chunking params
BATCH_SIZE = 64
TARGET_CHARS = 1800
OVERLAP_CHARS = 250

# retry/backoff params for OpenAI calls
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0

# load env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower())
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or export OPENAI_API_KEY in your environment.")

# ----------------- Helpers -----------------
def normalize_ar(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u0640]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text.strip()

def read_txt_to_pages(txt_path: Path) -> List[Dict]:
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")
    title = txt_path.name
    raw = txt_path.read_text(encoding="utf-8")
    if "\f" in raw:
        raw_pages = [p for p in raw.split("\f")]
    else:
        raw_pages = re.split(r'\n{2,}', raw)
    pages = []
    page_no = 1
    for p in raw_pages:
        t = (p or "").strip()
        if not t:
            continue
        pages.append({
            "doc_id": txt_path.stem,
            "title": title,
            "page_number": page_no,
            "text": normalize_ar(t)
        })
        page_no += 1
    return pages

def chunk_pages(pages: List[Dict], target_chars=TARGET_CHARS, overlap_chars=OVERLAP_CHARS) -> List[Dict]:
    chunks = []
    for pg in pages:
        raw = pg["text"]
        segments = re.split(r"(?:\n\n+|•|- |\u2022|\u25CF|•|\.|؛|:)\s*", raw)
        segments = [s.strip() for s in segments if s and s.strip()]
        buf = ""
        for seg in segments:
            if not buf:
                buf = seg
                continue
            candidate = (buf + " " + seg).strip()
            if len(candidate) <= target_chars:
                buf = candidate
            else:
                if buf:
                    chunks.append({
                        "doc_id": pg["doc_id"],
                        "title": pg["title"],
                        "page_number": pg["page_number"],
                        "chunk_id": f"{pg['doc_id']}_p{pg['page_number']}_c{len(chunks)}",
                        "text": buf
                    })
                overlap = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
                buf = (overlap + " " + seg).strip()
        if buf:
            chunks.append({
                "doc_id": pg["doc_id"],
                "title": pg["title"],
                "page_number": pg["page_number"],
                "chunk_id": f"{pg['doc_id']}_p{pg['page_number']}_c{len(chunks)}",
                "text": buf
            })
    return chunks

def embed_batch(openai_client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    # Retry with exponential backoff
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai_client.embeddings.create(model=model, input=[(t or " ").replace("\n", " ").strip() for t in texts])
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"[embed_batch] attempt {attempt} failed: {e}. retrying in {backoff}s ...")
            time.sleep(backoff)
            backoff *= 2.0
    raise RuntimeError("unreachable")

# ----------------- Main -----------------
def main():
    print(">>> Generating embeddings from:", TXT_PATH)
    pages = read_txt_to_pages(TXT_PATH)
    print(f" - read {len(pages)} page(s) from txt")

    chunks = chunk_pages(pages)
    print(f" - created {len(chunks)} chunk(s)")

    if len(chunks) == 0:
        raise RuntimeError("No text chunks produced. Check your input file or chunking parameters.")

    # Prepare metadata list and texts
    texts = [c["text"] for c in chunks]
    meta_lines = chunks  # each chunk is already a dict with needed fields

    # Initialize embeddings array
    N = len(texts)
    D = EMBED_DIM
    E = np.zeros((N, D), dtype="float32")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Batch embed
    idx = 0
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Embedding batches"):
        batch_texts = texts[i:i+BATCH_SIZE]
        embeddings = embed_batch(client, batch_texts, EMBED_MODEL)
        for j, emb in enumerate(embeddings):
            E[idx + j, :] = np.array(emb, dtype="float32")
        idx += len(embeddings)

    # Normalize rows (unit vectors)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = E / norms

    # Save embeddings and meta.jsonl (one JSON per line, utf-8)
    np.save(EMB_PATH, E)
    print(f"Saved embeddings to {EMB_PATH} (shape: {E.shape})")

    with open(META_PATH, "w", encoding="utf-8") as f:
        for ch in meta_lines:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"Saved metadata to {META_PATH} (lines: {len(meta_lines)})")

    print("✅ Done. You can now run the app (streamlit run app.py)")

if __name__ == "__main__":
    main()
