#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Arabic Q&A — DOCX edition (single knowledge pack)
- Reads: /Users/ha/My Drive/SmartMind Consultation/Chatbots/docs/smartmind_docs_v2.docx
- Arabic-aware normalization + chunking
- OpenAI embeddings (text-embedding-3-large)
- Cosine retrieval (scikit-learn)
- DeepSeek generation with strict grounding + OpenAI fallback
- CLI + logging + retries

Run examples:
  python rag_qa.py --question "اشرح لي سياسة الانسحاب"
  python rag_qa.py --reindex --question "ما هي الرسوم البنكية عند الانسحاب؟"
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from docx import Document
from sklearn.neighbors import NearestNeighbors

# ---------- Logging ----------
LOG = logging.getLogger("rag_qa")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)


# ---------- Config ----------
@dataclass
class Config:
    # Paths
    project_root: Path
    data_dir: Path
    index_dir: Path
    outputs_dir: Path
    meta_path: Path
    emb_path: Path
    docx_path: Path  # single consolidated knowledge doc

    # Models
    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072

    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"   # we'll auto-add /v1 if missing
    deepseek_model: str = "deepseek-reasoner"             # or deepseek-chat
    openai_fallback_model: str = "gpt-4o-mini"

    # Chunking
    target_chars: int = 1800
    overlap_chars: int = 250

    # Retrieval
    top_k: int = 5
    coverage_threshold: float = 0.22
    avg_threshold: float = 0.18

    # Generation
    max_answer_tokens: int = 1200
    temperature: float = 0.2
    top_p: float = 1.0

    # Infra
    batch_size: int = 64
    retries: int = 3
    retry_backoff: float = 1.8


def load_config() -> Config:
    # Load .env from current working directory (Spyder/Terminal-safe)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

    root = Path.cwd()
    data_dir = root / "data"
    index_dir = data_dir / "index"
    outputs_dir = root / "outputs"

    # YOUR consolidated DOCX path
    docx_path = Path("./smartmind_docs_v2.docx")
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required in .env")

    cfg = Config(
        project_root=root,
        data_dir=data_dir,
        index_dir=index_dir,
        outputs_dir=outputs_dir,
        meta_path=index_dir / "meta.jsonl",
        emb_path=index_dir / "embeddings.npy",
        docx_path=docx_path,
        openai_api_key=openai_key,
        deepseek_api_key=(os.getenv("DEEPSEEK_API_KEY", "").strip() or None),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        deepseek_model=(os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()
                        .replace("–", "-").replace("—", "-")),
        openai_fallback_model=os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini").strip(),
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ---------- Utilities ----------
def normalize_arabic(text: str) -> str:
    """Conservative Arabic normalization."""
    if not text:
        return ""
    text = re.sub(r"[\u0640]", "", text)  # remove tatweel
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text.strip()


# ---------- DOCX loading ----------
def read_docx_to_pages(docx_path: Path) -> List[Dict]:
    """
    Read a .docx and produce pseudo-pages for citation:
    - Each block separated by a blank paragraph becomes a 'page'.
    - We assign increasing page numbers so citations remain [العنوان، ص.X].
    """
    title = docx_path.name
    doc = Document(str(docx_path))

    pages: List[Dict] = []
    buf: List[str] = []
    page_no = 1

    def flush():
        nonlocal buf, page_no
        text = "\n".join(buf).strip()
        if text:
            pages.append({
                "doc_id": docx_path.stem,
                "title": title,
                "page_number": page_no,
                "text": normalize_arabic(text)
            })
        buf = []
        page_no += 1

    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if not t:
            if buf:
                flush()
            continue
        buf.append(t)

    if buf:
        flush()

    LOG.info("Loaded %d pseudo-pages from %s", len(pages), title)
    return pages


# ---------- Chunking ----------
def chunk_pages(pages: List[Dict], target_chars: int, overlap_chars: int) -> List[Dict]:
    """Split 'pages' into overlapping chunks preserving metadata."""
    chunks: List[Dict] = []
    for page in pages:
        raw = page["text"]
        # Split by paragraph/bullets/punctuation (simple robust splitter)
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
                        "doc_id": page["doc_id"],
                        "title": page["title"],
                        "page_number": page["page_number"],
                        "chunk_id": f"{page['doc_id']}_p{page['page_number']}_c{len(chunks)}",
                        "text": buf
                    })
                overlap = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
                buf = (overlap + " " + seg).strip()

        if buf:
            chunks.append({
                "doc_id": page["doc_id"],
                "title": page["title"],
                "page_number": page["page_number"],
                "chunk_id": f"{page['doc_id']}_p{page['page_number']}_c{len(chunks)}",
                "text": buf
            })

    LOG.info("Built %d chunks", len(chunks))
    return chunks


# ---------- Embeddings ----------
def embed_chunks(cfg: Config, chunks: List[Dict]) -> np.ndarray:
    """Create embeddings with OpenAI; returns (N, dim). Retries on transient errors."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg.openai_api_key)

    N = len(chunks)
    E = np.zeros((N, cfg.embed_dim), dtype="float32")
    i = 0

    LOG.info("Embedding %d chunks with %s ...", N, cfg.openai_embed_model)
    while i < N:
        batch = chunks[i:i + cfg.batch_size]
        inputs = [(ch["text"].replace("\n", " ").strip() or " ") for ch in batch]

        # retry loop
        for attempt in range(cfg.retries):
            try:
                resp = client.embeddings.create(model=cfg.openai_embed_model, input=inputs)
                break
            except Exception as e:
                wait = cfg.retry_backoff ** attempt
                LOG.warning("Embeddings retry %d due to %s; sleeping %.1fs", attempt + 1, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError("Embeddings failed after retries")

        for j, e in enumerate(resp.data):
            vec = np.array(e.embedding, dtype="float32")
            if vec.shape[0] != cfg.embed_dim:
                raise ValueError(f"Unexpected embedding dim {vec.shape[0]} vs {cfg.embed_dim}")
            E[i + j] = vec

        i += cfg.batch_size

    # L2 normalize for cosine metric
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = E / norms
    LOG.info("Embeddings ready: shape=%s dtype=%s", E.shape, E.dtype)
    return E


def save_index(cfg: Config, chunks: List[Dict], E: np.ndarray) -> None:
    np.save(cfg.emb_path, E)
    with open(cfg.meta_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    LOG.info("Saved embeddings -> %s", cfg.emb_path)
    LOG.info("Saved metadata   -> %s", cfg.meta_path)


def load_index(cfg: Config) -> Tuple[np.ndarray, List[str]]:
    E = np.load(cfg.emb_path)
    with open(cfg.meta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return E, lines


# ---------- Retrieval ----------
class Retriever:
    def __init__(self, E: np.ndarray):
        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn.fit(E)
        self.E = E

    def query(self, q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.nn.kneighbors(q_vec.reshape(1, -1), n_neighbors=min(k, self.E.shape[0]))
        sims = 1.0 - distances[0]
        idxs = indices[0]
        return sims, idxs


def embed_query(cfg: Config, query: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=cfg.openai_api_key)
    for attempt in range(cfg.retries):
        try:
            r = client.embeddings.create(model=cfg.openai_embed_model, input=[query])
            vec = np.array(r.data[0].embedding, dtype="float32")
            break
        except Exception as e:
            wait = cfg.retry_backoff ** attempt
            LOG.warning("Query-embed retry %d due to %s; sleeping %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    else:
        raise RuntimeError("Query embedding failed after retries")

    n = np.linalg.norm(vec)
    if n == 0:
        n = 1.0
    return vec / n


def build_sources_block(meta_lines: List[str], idxs: np.ndarray, trim: int = 900) -> Tuple[str, List[Dict]]:
    passages = []
    lines = []
    for i, idx in enumerate(idxs.tolist(), start=1):
        m = json.loads(meta_lines[int(idx)])
        body = m["text"].replace("\n", " ").strip()
        if len(body) > trim:
            body = body[:trim] + "..."
        passages.append({
            "title": m["title"], "page": m["page_number"],
            "chunk_id": m["chunk_id"], "text": body
        })
        header = f"{i}) [العنوان: {m['title']} | ص.{m['page_number']} | Chunk: {m['chunk_id']}]"
        lines.append(header + "\n" + body)
    return "\n\n".join(lines), passages


# ---------- Generation (DeepSeek + fallback) ----------
def call_deepseek(cfg: Config, system_prompt: str, user_prompt: str) -> Tuple[str, str, str]:
    """Returns (answer, used_model, used_base_url). Raises on hard failure."""
    from openai import OpenAI as DeepSeekClient

    if not cfg.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")

    base = cfg.deepseek_base_url.rstrip("/")
    base_candidates = [base, base + "/v1"] if not base.endswith("/v1") else [base, base[:-3]]

    valid_models = ["deepseek-reasoner", "deepseek-chat"]
    model_env = cfg.deepseek_model.replace("–", "-").replace("—", "-")
    model_candidates = [m for m in [model_env, "deepseek-reasoner", "deepseek-chat"] if m in valid_models]

    last_err = None
    for b in base_candidates:
        try:
            ds = DeepSeekClient(api_key=cfg.deepseek_api_key, base_url=b)
            try:
                live = {m.id for m in ds.models.list().data}
                candidates = [m for m in model_candidates if m in live] or model_candidates
            except Exception:
                candidates = model_candidates

            for model_id in candidates:
                try:
                    comp = ds.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_tokens=cfg.max_answer_tokens
                    )
                    return comp.choices[0].message.content, model_id, b
                except Exception as e_model:
                    last_err = e_model
        except Exception as e_base:
            last_err = e_base

    raise RuntimeError(f"DeepSeek failed: {last_err}")


def call_openai_fallback(cfg: Config, system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=cfg.openai_api_key)
    comp = client.chat.completions.create(
        model=cfg.openai_fallback_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_answer_tokens
    )
    return comp.choices[0].message.content


# ---------- Main pipeline ----------
def run_pipeline(cfg: Config, question: str, reindex: bool = False) -> None:
    # (1) Index (build or reuse)
    if reindex or not (cfg.emb_path.exists() and cfg.meta_path.exists()):
        LOG.info("Building index from DOCX ...")
        pages = read_docx_to_pages(cfg.docx_path)
        chunks = chunk_pages(pages, cfg.target_chars, cfg.overlap_chars)
        E = embed_chunks(cfg, chunks)
        save_index(cfg, chunks, E)
    else:
        LOG.info("Using existing index (no --reindex)")

    # (2) Load index + retriever
    E, meta_lines = load_index(cfg)
    retriever = Retriever(E)

    # (3) Retrieve
    q = question.replace("\n", " ").strip()
    q_vec = embed_query(cfg, q)
    sims, idxs = retriever.query(q_vec, cfg.top_k)

    # (4) Coverage gates
    best_sim = float(sims[0]) if len(sims) else 0.0
    avg_sim = float(np.mean(sims)) if len(sims) else 0.0
    LOG.info("best_sim=%.4f  avg_sim=%.4f", best_sim, avg_sim)

    if best_sim < cfg.coverage_threshold and avg_sim < cfg.avg_threshold:
        print("\n--- Answer (Arabic) ---\n")
        print("لا أستطيع الإجابة اعتمادًا على المستندات المزوّدة.")
        print(f"\n[reason: low retrieval similarity; best_sim={best_sim:.4f} | avg_sim={avg_sim:.4f}]")
        return
    elif best_sim < cfg.coverage_threshold:
        print("\n--- Answer (Arabic) ---\n")
        print("لا أستطيع الإجابة اعتمادًا على المستندات المزوّدة.")
        print(f"\n[reason: low retrieval similarity; best_sim={best_sim:.4f}]")
        return

    # (5) Build sources
    sources_block, _passages = build_sources_block(meta_lines, idxs, trim=900)

    # (6) Strict, synthesized Arabic system prompt
    system_prompt_ar = (
        "أنت مساعد عربي يجيب حصراً من «المقاطع» المزوّدة.\n"
        "المطلوب: صياغة إجابة واحدة موجزة ومنظمة للمستخدم (ملخّص قصير ثم نقاط قليلة)، وليست اقتباسات متسلسلة.\n"
        "القواعد:\n"
        "1) لا تستخدم أي معرفة خارج «المصادر». إذا لم تكفِ، ارفض: \"لا أستطيع الإجابة اعتمادًا على المستندات المزوّدة.\"\n"
        "2) اجعل الإسناد في نهاية كل نقطة/سطر فقط بالشكل: [العنوان، ص.X].\n"
        "3) للأرقام أو الصيغ الحسّاسة، اقتبس عبارة قصيرة \"بين علامتَي اقتباس\" ثم ضع الإسناد.\n"
        "4) استخدم العربية الفصحى المختصرة وتجنّب التكرار.\n"
    )

    user_prompt_ar = (
        f"سؤال المستخدم:\n{question}\n\n"
        f"المصادر (اعتمد عليها فقط):\n{sources_block}\n\n"
        "رجاءً أعد صياغة إجابة واحدة وفق التنسيق أعلاه، مع إسناد مختصر في نهاية كل نقطة/سطر فقط "
        "بالشكل [العنوان، ص.X]، واذكر \"المراجع\" في سطر أخير يضم العنوان وأرقام الصفحات المستخدمة."
    )

    # (7) Generate: DeepSeek → fallback
    answer = None
    provider = ""
    if cfg.deepseek_api_key:
        try:
            answer, used_model, used_base = call_deepseek(cfg, system_prompt_ar, user_prompt_ar)
            provider = f"DeepSeek [{used_model}] {used_base}"
        except Exception as e:
            LOG.warning("DeepSeek failed (%s); switching to OpenAI fallback", e)

    if answer is None:
        answer = call_openai_fallback(cfg, system_prompt_ar, user_prompt_ar)
        provider = f"OpenAI [{cfg.openai_fallback_model}]"

    # (8) Output
    answer = re.sub(r"[ \t]+", " ", answer or "").strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    print("\n--- Arabic Answer ---\n")
    print(answer)
    print(f"\n[provider: {provider} | best_sim={best_sim:.4f} | avg_sim={avg_sim:.4f}]")

    # (9) Save transcript
    out_path = cfg.outputs_dir / "last_answer.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)
        f.write(f"\n\n[provider: {provider} | best_sim={best_sim:.4f} | avg_sim={avg_sim:.4f}]")
    LOG.info("Saved answer -> %s", out_path)


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Arabic Q&A (DOCX knowledge pack)")
    p.add_argument("--question", "-q", type=str, default="اشرح لي سياسة الانسحاب",
                   help="Arabic question")
    p.add_argument("--reindex", action="store_true", help="Rebuild index from the DOCX")
    p.add_argument("--k", type=int, default=None, help="Top-k passages (override config)")
    p.add_argument("--max-tokens", type=int, default=None, help="Answer max tokens (override config)")
    p.add_argument("--debug", action="store_true", help="Verbose logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        LOG.setLevel(logging.DEBUG)

    cfg = load_config()
    if args.k:
        cfg.top_k = int(args.k)
    if args.max_tokens:
        cfg.max_answer_tokens = int(args.max_tokens)

    run_pipeline(cfg, question=args.question, reindex=args.reindex)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user")