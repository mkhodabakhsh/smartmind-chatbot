#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re, time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Configuration
# ---------------------------
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# TXT input priority:
# 1) environment var INPUT_TXT_PATH
# 2) any .txt found under DATA_DIR
# 3) default path PROJECT_ROOT / data / data.txt
TXT_PATH_ENV = os.getenv("INPUT_TXT_PATH", "").strip()
if TXT_PATH_ENV:
    TXT_PATH = Path(TXT_PATH_ENV)
else:
    txts_found = list(DATA_DIR.glob("**/*.txt"))
    if txts_found:
        TXT_PATH = txts_found[0]
    else:
        TXT_PATH = PROJECT_ROOT / "data" / "data.txt"

META_PATH = INDEX_DIR / "meta.jsonl"
EMB_PATH  = INDEX_DIR / "embeddings.npy"

# Load environment variables
# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

# DEEPSEEK API key: default
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
try:
    if not DEEPSEEK_API_KEY and "st" in globals():
        DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", None)
except Exception:
    DEEPSEEK_API_KEY = None

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL_ENV = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()

# OPENAI API key: fallback only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
try:
    if not OPENAI_API_KEY and "st" in globals():
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    OPENAI_API_KEY = None

OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini").strip()

# Check: at least one API key must exist
if not (DEEPSEEK_API_KEY or OPENAI_API_KEY):
    raise RuntimeError("At least one API key must be provided: DEEPSEEK_API_KEY (preferred) or OPENAI_API_KEY (fallback).")

# Constants
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
TOP_K = 5
COVERAGE_THRESHOLD = 0.22
AVG_THRESHOLD = 0.18
MAX_TOKENS = 1200
TEMPERATURE = 0.2
TOP_P = 1.0

# ---------------------------
# Streamlit Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Powered Smart Mind Info Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (kept from original)
st.markdown("""
<style>
/* Hide Streamlit elements */
.stDeployButton {display:none;}
footer {visibility: hidden;}
.stApp > header {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* RTL Support */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}

/* Container styling */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1000px;
    margin: 0 auto;
}

/* Header styling */
.main-header {
    text-align: center;
    margin-bottom: 3rem;
    padding: 2.5rem 1rem;
    background: linear-gradient(135deg, #f8fffe 0%, #e8f4f2 100%);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: #2c5530;
    margin: 1rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    letter-spacing: 1px;
}

.logo-image {
    width: 120px;
    height: auto;
    margin-bottom: 1rem;
}

/* Tips section */
.tips-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f9ff 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 2rem 0;
    border-right: 5px solid #2c5530;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.tips-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #2c5530;
    margin-bottom: 1rem;
}

.tips-content {
    font-size: 1rem;
    color: #444;
    line-height: 1.6;
}

/* Chat section */
.chat-title {
    font-size: 2rem;
    font-weight: bold;
    color: #2c5530;
    text-align: center;
    margin: 2rem 0 1rem 0;
}

.chat-container {
    max-height: 500px;
    overflow-y: auto;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: #fafafa;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
}

/* Chat bubbles - both on right side with icons */
.chat-bubble {
    max-width: 80%;
    margin: 1rem 0;
    padding: 1.2rem 1.5rem;
    border-radius: 20px;
    float: right;
    clear: both;
    font-size: 1.1rem;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    word-wrap: break-word;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.user-bubble {
    background: linear-gradient(135deg, #b0b0b0, #999);
    color: white;
    border-radius: 20px 20px 5px 20px;
    margin-left: 15%;
}

.smartmind-bubble {
    background: linear-gradient(135deg, #66bb6a, #4caf50);
    color: white;
    border-radius: 20px 20px 5px 20px;
    margin-left: 10%;
    margin-top: 0.5rem;
}

.chat-icon {
    font-size: 1.5rem;
    min-width: 1.5rem;
    margin-top: 2px;
    opacity: 0.9;
}

.chat-content {
    flex: 1;
}

.bubble-label {
    font-size: 0.9rem;
    font-weight: bold;
    margin-bottom: 0.8rem;
    opacity: 0.9;
}

.chat-clear {
    clear: both;
    height: 1rem;
}

/* Input styling */
.input-section {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #e0e0e0;
    margin-bottom: 2rem;
}

.stTextInput > div > div > input {
    border-radius: 25px !important;
    padding: 1rem 1.5rem !important;
    border: 2px solid #e0e0e0 !important;
    font-size: 1.1rem !important;
    font-family: 'Arial', sans-serif !important;
}

.stTextInput > div > div > input:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4caf50, #66bb6a) !important;
    color: white !important;
    border-radius: 25px !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3) !important;
}

/* Custom spinner */
.custom-spinner {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    margin: 1rem 0;
}

.spinner-container {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.spinner-text {
    font-size: 1.1rem;
    color: #2c5530;
    font-weight: 500;
}

.spinner-dots {
    display: flex;
    gap: 0.5rem;
}

.spinner-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #2c5530;
    animation: bounce 1.4s ease-in-out infinite both;
}

.spinner-dot:nth-child(1) { animation-delay: -0.32s; }
.spinner-dot:nth-child(2) { animation-delay: -0.16s; }
.spinner-dot:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #666;
    background: #fafafa;
    border-radius: 15px;
    margin: 2rem 0;
}

.empty-state h3 {
    color: #2c5530;
    font-size: 1.8rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Helper Functions
# ---------------------------
def normalize_ar(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u0640]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text.strip()

@st.cache_data(show_spinner=False)
def read_txt_to_pages(txt_path: Path) -> List[Dict]:
    """
    Read a plain text file and split into 'pages'.
    Splitting logic:
      - If the file contains form-feed characters (\f) we split on that.
      - Otherwise split by groups of blank lines (two or more newlines).
    Each page becomes a dict: {doc_id, title, page_number, text}
    """
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

def chunk_pages(pages: List[Dict], target_chars=1800, overlap_chars=250) -> List[Dict]:
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
                        "doc_id": pg["doc_id"], "title": pg["title"], "page_number": pg["page_number"],
                        "chunk_id": f"{pg['doc_id']}_p{pg['page_number']}_c{len(chunks)}",
                        "text": buf
                    })
                overlap = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
                buf = (overlap + " " + seg).strip()
        if buf:
            chunks.append({
                "doc_id": pg["doc_id"], "title": pg["title"], "page_number": pg["page_number"],
                "chunk_id": f"{pg['doc_id']}_p{pg['page_number']}_c{len(chunks)}",
                "text": buf
            })
    return chunks

def embed_texts_openai(texts: List[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    E = np.zeros((len(texts), EMBED_DIM), dtype="float32")
    bs, i = 64, 0
    while i < len(texts):
        batch = texts[i:i+bs]
        clean = [(t or " ").replace("\n", " ").strip() for t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=clean)
        for j, e in enumerate(resp.data):
            E[i+j] = np.array(e.embedding, dtype="float32")
        i += bs
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return E / norms

def save_index(chunks: List[Dict], E: np.ndarray):
    np.save(EMB_PATH, E)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

@st.cache_resource(show_spinner=False)
def load_index() -> Tuple[NearestNeighbors, np.ndarray, List[str]]:
    E = np.load(EMB_PATH)
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(E)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_lines = f.readlines()
    return nn, E, meta_lines

def embed_query(q: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    r = client.embeddings.create(model=EMBED_MODEL, input=[q.replace("\n", " ").strip()])
    v = np.array(r.data[0].embedding, dtype="float32")
    n = np.linalg.norm(v) or 1.0
    return v / n

def build_sources(meta_lines: List[str], idxs: np.ndarray, trim=900) -> str:
    out_lines = []
    for i, idx in enumerate(idxs.tolist(), start=1):
        m = json.loads(meta_lines[int(idx)])
        body = m["text"].replace("\n", " ").strip()
        if len(body) > trim: 
            body = body[:trim] + "..."
        header = f"{i}) [العنوان: {m['title']} | ص.{m['page_number']} | Chunk: {m['chunk_id']}]"
        out_lines.append(header + "\n" + body)
    return "\n\n".join(out_lines)

def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI as DeepSeekClient
    base = DEEPSEEK_BASE_URL.rstrip("/")
    bases = [base, base + "/v1"] if not base.endswith("/v1") else [base, base[:-3]]
    candidates = ["deepseek-reasoner", "deepseek-chat"]

    for b in bases:
        try:
            ds = DeepSeekClient(api_key=DEEPSEEK_API_KEY, base_url=b)
            for model in candidates:
                try:
                    comp = ds.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS
                    )
                    return comp.choices[0].message.content
                except:
                    continue
        except:
            continue
    raise Exception("DeepSeek failed")

def call_openai_fallback(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    comp = client.chat.completions.create(
        model=OPENAI_FALLBACK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    return comp.choices[0].message.content

def clean_answer(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def show_spinner(message: str):
    return st.markdown(f"""
    <div class="custom-spinner">
        <div class="spinner-container">
            <div class="spinner-text">{message}</div>
            <div class="spinner-dots">
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Check if index exists; if not try to build from TXT
# ---------------------------
if not (EMB_PATH.exists() and META_PATH.exists()):
    # try to build index from TXT_PATH (if available)
    if TXT_PATH.exists():
        try:
            st.info(f"الفهرس غير موجود — سيتم بناؤه من الملف النصي: {TXT_PATH}")
            pages = read_txt_to_pages(TXT_PATH)
            chunks = chunk_pages(pages)
            texts = [c["text"] for c in chunks]
            # embeddings can take a while — show spinner
            with st.spinner("جاري إنشاء المتجهات والفهرس — قد يستغرق بعض الوقت..."):
                E = embed_texts_openai(texts)
                save_index(chunks, E)
            st.success("تم إنشاء الفهرس بنجاح من الملف النصي.")
        except Exception as e:
            st.error(f"فشل إنشاء الفهرس من الملف النصي: {e}")
            st.stop()
    else:
        st.error("الفهرس غير موجود وملف TXT المصدر غير موجود. يرجى وضع ملف .txt داخل مجلد data/ أو تحديد INPUT_TXT_PATH في .env.")
        st.stop()

# Load index
nn, E, meta_lines = load_index()

# ---------------------------
# UI Layout
# ---------------------------

# Header with logo and title
st.markdown('<div class="main-header">', unsafe_allow_html=True)

logo_path = PROJECT_ROOT / "logo.png"
if logo_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(logo_path), width=120)

st.markdown('<h1 class="main-title">AI Powered Smart Mind Info Center</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tips section
st.markdown("""
<div class="tips-box">
    <div class="tips-title">💡 نصائح للحصول على أفضل إجابة</div>
    <div class="tips-content">
        • اكتب سؤالك بوضوح عن خدمة أو سياسة، واذكر التفاصيل المهمة (البرنامج/المستوى/الفرع/التاريخ) لتحصل على إجابة أدق<br>
        • أجيب من المستندات المرفقة فقط؛ وإن لم تتوفر الإجابة، سأقول إنني لم أعثر عليها
    </div>
</div>
""", unsafe_allow_html=True)

# Chat section title
st.markdown('<h2 class="chat-title">💬 اطرح سؤالك</h2>', unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for chat in reversed(st.session_state.chat_history):
        # User bubble with icon
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            <div class="chat-icon">👤</div>
            <div class="chat-content">
                <div class="bubble-label">أنت:</div>
                {chat["question"]}
            </div>
        </div>
        <div class="chat-clear"></div>
        """, unsafe_allow_html=True)
        
        # Smart Mind bubble with brain icon
        st.markdown(f"""
        <div class="chat-bubble smartmind-bubble">
            <div class="chat-icon">🧠</div>
            <div class="chat-content">
                <div class="bubble-label">سمارت مايند:</div>
                {chat["answer"]}
            </div>
        </div>
        <div class="chat-clear"></div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input form - matching app copy.py style
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_question = st.text_input(
            label="سؤالك",
            placeholder="اكتب سؤالك هنا...",
            label_visibility="collapsed",
            key="question_input"
        )
    
    with col2:
        submit_button = st.form_submit_button("➤")

# Process question
if submit_button and user_question.strip():
    # Show loading for retrieval
    spinner1 = st.empty()
    spinner1.markdown("""
    <div class="custom-spinner">
        <div class="spinner-container">
            <div class="spinner-text">جاري البحث في المستندات...</div>
            <div class="spinner-dots">
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
                <div class="spinner-dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieve relevant passages
    q_vec = embed_query(user_question)
    n_neighbors = min(TOP_K, E.shape[0])
    distances, indices = nn.kneighbors(q_vec.reshape(1, -1), n_neighbors=n_neighbors)
    sims = 1.0 - distances[0]
    idxs = indices[0]
    
    spinner1.empty()
    
    best_sim = float(sims[0]) if len(sims) else 0.0
    avg_sim = float(np.mean(sims)) if len(sims) else 0.0
    
    # Check coverage
    if best_sim < COVERAGE_THRESHOLD and avg_sim < AVG_THRESHOLD:
        answer = "لا أستطيع الإجابة اعتماداً على المستندات المزوّدة."
    elif best_sim < COVERAGE_THRESHOLD:
        answer = "لا أستطيع الإجابة اعتماداً على المستندات المزوّدة."
    else:
        # Build context
        sources_block = build_sources(meta_lines, idxs, trim=900)
        
        # Prepare prompts
        system_prompt = (
            "أنت مساعد عربي يجيب حصراً من المقاطع المزوّدة. "
            "اكتب إجابة واحدة موجزة ومنظمة. لا تستخدم رموز التنسيق مثل ** أو *. "
            "إذا لم تكفِ المعلومات، قل: لا أستطيع الإجابة اعتماداً على المستندات المزوّدة."
        )
        
        user_prompt = (
            f"سؤال المستخدم:\n{user_question}\n\n"
            f"المصادر:\n{sources_block}\n\n"
            "أجب بإيجاز مع ذكر المراجع بالشكل [العنوان، ص.X]"
        )
        
        # Show loading for generation
        spinner2 = st.empty()
        spinner2.markdown("""
        <div class="custom-spinner">
            <div class="spinner-container">
                <div class="spinner-text">جاري تحضير الإجابة...</div>
                <div class="spinner-dots">
                    <div class="spinner-dot"></div>
                    <div class="spinner-dot"></div>
                    <div class="spinner-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate answer
        answer = None

        # Prefer DeepSeek
        if DEEPSEEK_API_KEY:
            try:
                answer = call_deepseek(system_prompt, user_prompt)
            except Exception as e:
                st.warning(f"DeepSeek API failed: {e}")

        # Fallback to OpenAI if DeepSeek failed or key missing
        if (answer is None or not answer.strip()) and OPENAI_API_KEY:
            try:
                answer = call_openai_fallback(system_prompt, user_prompt)
            except Exception as e:
                st.error(f"OpenAI fallback failed: {e}")
                answer = "عذراً، لم أتمكن من الحصول على إجابة من أي مصدر."

        # Final cleanup
        answer = clean_answer(answer or "لا أستطيع الإجابة اعتماداً على المستندات المزوّدة.")

        
        spinner2.empty()
        answer = clean_answer(answer)
    
    # Add to chat history
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": answer
    })
    
    st.rerun()

# Empty state
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <h3>مرحباً بك في مركز معلومات سمارت مايند الذكي</h3>
        <p>اطرح سؤالك أعلاه للحصول على إجابة دقيقة من مستنداتنا</p>
    </div>
    """, unsafe_allow_html=True)
