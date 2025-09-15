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

# ---------------------------
# UI CSS 
# ---------------------------
st.markdown("""
<style>
/* Force entire page and Streamlit containers to be whitish and dark text */
html, body, .stApp, .main, .block-container, .reportview-container, .viewerBadge_container, .css-1y4p8pa, .css-1outpf7, .css-18e3th9 {
    background: #ffffff !important;
    color: #0b0b0b !important;
    direction: rtl;
}

/* Hide Streamlit default chrome for a clean look */
.stApp > header { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Container styling */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1100px;
    margin: 0 auto;
}

/* Main header card */
.main-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding: 18px 12px;
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 8px 30px rgba(16,24,40,0.06);
    border: 1px solid rgba(16,24,40,0.04);
}

/* Title styling */
.main-title {
    font-size: 28px;
    font-weight: 700;
    color: #0b0b0b;
    margin: 6px 0;
}

/* Tips section */
.tips-box {
    background: #ffffff;
    padding: 14px;
    border-radius: 12px;
    margin: 0.8rem 0;
    border: 1px solid rgba(16,24,40,0.04);
    box-shadow: 0 6px 20px rgba(16,24,40,0.03);
    color: #0b0b0b;
}
.tips-title { font-size: 1rem; font-weight:700; color: #0b0b0b; margin-bottom:8px; }
.tips-content { font-size: 0.98rem; color: #222; line-height: 1.6; }

/* Chat container */
.chat-title { font-size:20px; font-weight:700; color:#0b0b0b; text-align:center; margin:10px 0 8px 0; }
.chat-container {
    max-height: 560px;
    overflow-y: auto;
    margin-bottom: 1rem;
    padding: 12px;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid rgba(16,24,40,0.04);
    box-shadow: 0 6px 20px rgba(16,24,40,0.03);
}

/* Chat bubbles */
.chat-bubble {
    max-width: 82%;
    margin: 12px 0;
    padding: 12px 14px;
    border-radius: 14px;
    float: right;
    clear: both;
    font-size: 1rem;
    line-height: 1.6;
    box-shadow: 0 6px 18px rgba(16,24,40,0.03);
    word-wrap: break-word;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #ffffff;
    color: #0b0b0b;
    border: 1px solid rgba(16,24,40,0.03);
}

.user-bubble {
    background: #fbfbfb;
    border-radius: 14px 14px 6px 14px;
    margin-left: 12%;
    color: #0b0b0b;
}

.smartmind-bubble {
    background: #ffffff;
    border-radius: 14px 14px 6px 14px;
    margin-left: 8%;
    margin-top: 6px;
    color: #0b0b0b;
    border-left: 4px solid rgba(10,10,10,0.04);
}

.chat-icon { font-size: 1.4rem; min-width: 1.4rem; margin-top:2px; opacity:0.95; }
.chat-content { flex: 1; }
.bubble-label { font-size:0.95rem; font-weight:700; margin-bottom:6px; color:#0b0b0b; }
.chat-clear { clear: both; height: 1rem; }

/* Input styling */
.input-section { background: #ffffff; padding: 12px; border-radius: 12px; border: 1px solid rgba(16,24,40,0.04); margin-bottom:1rem; }
.stTextInput > div > div > input {
    border-radius: 12px !important;
    padding: 12px 14px !important;
    border: 1px solid rgba(16,24,40,0.06) !important;
    font-size: 1rem !important;
    color: #0b0b0b !important;
    background: #ffffff !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(11,11,11,0.12) !important;
    box-shadow: 0 6px 20px rgba(16,24,40,0.06) !important;
}

/* Buttons white with dark text */
.stButton > button {
    background: #ffffff !important;
    color: #0b0b0b !important;
    border-radius: 12px !important;
    border: 1px solid rgba(16,24,40,0.08) !important;
    padding: 8px 14px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 6px 18px rgba(16,24,40,0.02) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(16,24,40,0.06) !important;
}

/* --- SPECIFIC FIX: submit button default (not hover) appearance --- */
/* Target the form submit button used in the app (data-testid as observed) */
button[data-testid="stBaseButton-secondaryFormSubmit"],
button[class*="st-emotion-cache"][data-testid="stBaseButton-secondaryFormSubmit"],
.stButton>button[aria-label=""] {
    background: #ffffff !important;
    color: #0b0b0b !important;
    border: 1px solid rgba(16,24,40,0.06) !important;
    box-shadow: 0 6px 18px rgba(16,24,40,0.02) !important;
    opacity: 1 !important;
}

/* Also general fallback for any emotion-generated button classes to ensure consistent default look */
button[class*="st-emotion-cache"] {
    background: #ffffff !important;
    color: #0b0b0b !important;
    border: 1px solid rgba(16,24,40,0.06) !important;
}

/* Custom spinner - white surface */
.custom-spinner {
    display:flex; justify-content:center; align-items:center;
    padding:12px; background:#ffffff; border-radius:10px; border:1px solid rgba(16,24,40,0.04);
    color:#0b0b0b;
}
.spinner-dots { display:flex; gap:6px; }
.spinner-dot { width:10px; height:10px; border-radius:50%; background:#0b0b0b; animation:bounce 1.2s infinite; }
@keyframes bounce { 0%,80%,100%{transform:scale(0);}40%{transform:scale(1);} }

/* Empty state */
.empty-state { text-align:center; padding:28px 18px; color:#0b0b0b; background:#ffffff; border-radius:12px; margin:1rem 0; border:1px solid rgba(16,24,40,0.03); }

/* Keep markdown/code readable */
code, pre { background:#fbfbfb !important; color:#0b0b0b !important; padding:6px; border-radius:6px; }

/* Ensure images contained */
img { max-width:100%; height:auto; display:block; }
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

# Display chat history (chronological order: older first, new appended below)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
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
        
        # Smart Mind bubble with stylish inline SVG icon
        assistant_icon_svg = '''
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <rect x="3" y="3" width="18" height="14" rx="3" fill="#ffffff" stroke="rgba(10,10,10,0.06)" stroke-width="1"/>
          <circle cx="8.5" cy="9" r="1.3" fill="#0b0b0b"/>
          <circle cx="15.5" cy="9" r="1.3" fill="#0b0b0b"/>
          <path d="M8.8 13c.8.7 2.4.7 3.2 0" stroke="#0b0b0b" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
          <rect x="10" y="2" width="4" height="1.6" rx="0.4" fill="#e9eef2" />
        </svg>
        '''
        st.markdown(f"""
        <div class="chat-bubble smartmind-bubble">
            <div class="chat-icon">{assistant_icon_svg}</div>
            <div class="chat-content">
                <div class="bubble-label">سمارت مايند:</div>
                {chat["answer"]}
            </div>
        </div>
        <div class="chat-clear"></div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <h3>مرحباً بك في مركز معلومات سمارت مايند الذكي</h3>
        <p>اطرح سؤالك أعلاه للحصول على إجابة دقيقة من مستنداتنا</p>
    </div>
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
        if (answer is None or not str(answer).strip()) and OPENAI_API_KEY:
            try:
                answer = call_openai_fallback(system_prompt, user_prompt)
            except Exception as e:
                st.error(f"OpenAI fallback failed: {e}")
                answer = "عذراً، لم أتمكن من الحصول على إجابة من أي مصدر."

        # Final cleanup
        answer = clean_answer(answer or "لا أستطيع الإجابة اعتماداً على المستندات المزوّدة.")
        
        spinner2.empty()
        answer = clean_answer(answer)
    
    # Add to chat history (append keeps chronological order)
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": answer
    })
    
    # Safely rerun to refresh UI; wrapped to avoid environment-specific crash
    try:
        st.rerun()
    except Exception:
        pass

# End of file
