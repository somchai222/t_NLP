"""ไกด์ท่องเที่ยวปราจีนบุรี — RAG Chatbot (Streamlit + FAISS + Groq)"""
import os
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen/qwen3.8-27b"
CHUNK_SIZE = 500      # ตัวอักษรต่อ chunk (โดยประมาณ)
CHUNK_OVERLAP = 100   # ตัวอักษรที่ซ้อนกันระหว่าง chunk
NOT_FOUND = "ไม่พบข้อมูลในเอกสาร"

SYSTEM_PROMPT = f"""คุณคือ "ไกด์ท่องเที่ยวปราจีนบุรี" ผู้ช่วยตอบคำถามเกี่ยวกับแหล่งท่องเที่ยวในจังหวัดปราจีนบุรี
กฎที่ต้องทำตามอย่างเคร่งครัด:
1. ตอบจาก [เอกสารอ้างอิง] ที่ให้มาเท่านั้น ห้ามใช้ความรู้นอกเอกสารและห้ามเดาข้อมูล
2. ถ้าเอกสารไม่มีคำตอบ ให้ตอบว่า "{NOT_FOUND}" แล้วบอกสั้น ๆ ว่าเอกสารมีข้อมูลเรื่องอะไรที่ใกล้เคียง
3. ถ้าเอกสารระบุข้อมูลไม่ตรงกัน (เช่น เวลาเปิด-ปิด) ให้บอกว่าข้อมูลไม่ตรงกันและแจ้งทุกค่า
4. ท้ายคำตอบให้อ้างอิงแหล่งที่มาในรูปแบบ (แหล่งที่มา: ชื่อไฟล์)
5. ตอบเป็นภาษาเดียวกับที่ผู้ใช้ถาม (ไทยหรืออังกฤษ) กระชับ สุภาพ และใช้ตัวเลขเวลา/ราคาตามเอกสารเป๊ะ ๆ
6. เตือนสั้น ๆ ให้ตรวจสอบเวลาและราคากับสถานที่จริงก่อนเดินทาง เมื่อตอบเรื่องเวลาเปิด-ปิดหรือค่าเข้าชม"""

REWRITE_PROMPT = """จากประวัติการสนทนาและคำถามล่าสุด ให้เขียนคำถามล่าสุดใหม่ให้เป็นประโยคที่สมบูรณ์ในตัวเอง
(แทนคำสรรพนามหรือคำที่อ้างถึงด้วยชื่อสถานที่จริง) ตอบเฉพาะคำถามที่เขียนใหม่ ไม่ต้องอธิบายเพิ่ม
ถ้าคำถามสมบูรณ์อยู่แล้วให้ตอบคำถามเดิม"""


# ---------- 1) Document Loading & Chunking ----------
def clean_text(text: str) -> str:
    text = normalize(text)                 # ปรับสระ/วรรณยุกต์ซ้ำของ PyThaiNLP
    text = re.sub(r"[#*`>]+", "", text)    # ลบสัญลักษณ์ Markdown
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, title: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """รวมบรรทัดเป็น chunk ขนาดไม่เกิน size และเติมหัวข้อเอกสารไว้หน้าทุก chunk"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks, buf = [], ""
    for ln in lines:
        if buf and len(buf) + len(ln) + 1 > size:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf = f"{buf}\n{ln}".strip()
    if buf:
        chunks.append(buf)
    return [f"[{title}] {c}" for c in chunks]


@st.cache_resource(show_spinner="กำลังโหลดเอกสารและสร้างดัชนี...")
def build_index():
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        title = raw.strip().split("\n")[0].lstrip("# ").strip() or path.stem
        for i, chunk in enumerate(chunk_text(clean_text(raw), title)):
            docs.append({"text": chunk, "source": path.name, "title": title, "chunk_id": i})

    # ---------- 2) Embedding & Vector Search ----------
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                           show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity (เวกเตอร์ normalize แล้ว)
    index.add(vectors)
    return model, index, docs


def retrieve(query: str, k: int):
    model, index, docs = build_index()
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [{**docs[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]


# ---------- 3) LLM ----------
def get_client() -> Groq:
    key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    key = key or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("ไม่พบ GROQ_API_KEY — ตั้งค่าใน Streamlit Secrets (หรือไฟล์ .streamlit/secrets.toml ตอนรันในเครื่อง)")
        st.stop()
    return Groq(api_key=key)


def rewrite_query(client: Groq, history: list, question: str) -> str:
    """ทำให้คำถามต่อเนื่อง (เช่น "แล้วค่าเข้าล่ะ") ค้นหาเจอ โดยเติมชื่อสถานที่จากบทสนทนาก่อนหน้า"""
    if not history:
        return question
    recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0, max_tokens=150,
            messages=[{"role": "system", "content": REWRITE_PROMPT},
                      {"role": "user", "content": f"ประวัติ:\n{recent}\n\nคำถามล่าสุด: {question}"}],
        )
        return resp.choices[0].message.content.strip() or question
    except Exception:
        return question


def answer(client: Groq, history: list, question: str, contexts: list) -> str:
    context_text = "\n\n".join(f"[{i}] (ไฟล์: {c['source']})\n{c['text']}" for i, c in enumerate(contexts, 1))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user",
                     "content": f"[เอกสารอ้างอิง]\n{context_text}\n\n[คำถาม]\n{question}"})
    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                          temperature=0.1, max_tokens=700)
    return resp.choices[0].message.content.strip()


# ---------- 4) Chatbot Interface ----------
st.set_page_config(page_title="ไกด์ท่องเที่ยวปราจีนบุรี", page_icon="🌿")
st.title("🌿 ไกด์ท่องเที่ยวปราจีนบุรี")
st.caption("ถามเรื่องสถานที่เที่ยว เวลาเปิด-ปิด ค่าเข้าชม (ไทย/English) — ตอบจากคลังเอกสารเท่านั้น")

with st.sidebar:
    st.header("ตั้งค่า")
    top_k = st.slider("จำนวนเอกสารที่ดึงมา (top-k)", 2, 8, 4)
    st.markdown("**ตัวอย่างคำถาม**")
    examples = ["พิพิธภัณฑสถานแห่งชาติ ปราจีนบุรี เปิดวันไหน",
                "ค่าเข้าอุทยานแห่งชาติทับลานเท่าไร",
                "ทุ่งดอกหงอนนาคเปิดช่วงเดือนไหน",
                "Which places are free to enter?"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.info("ข้อมูลรวบรวมจากเว็บท่องเที่ยวสาธารณะ โปรดตรวจสอบเวลาและราคากับสถานที่จริงก่อนเดินทาง")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 เอกสารอ้างอิงที่ใช้ตอบ"):
                for s in m["sources"]:
                    st.markdown(f"**{s['title']}** · `{s['source']}` · ความคล้าย {s['score']:.2f}")
                    st.caption(s["text"])

question = st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวปราจีนบุรี...") or st.session_state.pop("pending", None)

if question:
    client = get_client()
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาและเรียบเรียงคำตอบ..."):
            search_query = rewrite_query(client, history, question)
            contexts = retrieve(search_query, top_k)
            try:
                reply = answer(client, history, question, contexts)
            except Exception as e:
                reply = f"เกิดข้อผิดพลาดในการเรียก LLM: {e}"
        st.markdown(reply)
        with st.expander("📚 เอกสารอ้างอิงที่ใช้ตอบ"):
            st.caption(f"คำค้นที่ใช้: {search_query}")
            for s in contexts:
                st.markdown(f"**{s['title']}** · `{s['source']}` · ความคล้าย {s['score']:.2f}")
                st.caption(s["text"])
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": contexts})
