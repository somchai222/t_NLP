# 🌿 ไกด์ท่องเที่ยวปราจีนบุรี (RAG Chatbot)

แชตบอตตอบคำถามเกี่ยวกับแหล่งท่องเที่ยวจังหวัดปราจีนบุรี (เวลาเปิด-ปิด ค่าเข้าชม จุดเด่น การวางแผนเส้นทาง) ด้วยเทคนิค **Retrieval-Augmented Generation** รองรับคำถามไทย/อังกฤษ

- 🔗 Streamlit App: `ใส่ลิงก์ที่ deploy แล้ว`
- 🔗 GitHub: `ใส่ลิงก์ repository`

## แนวคิดของ Domain
นักท่องเที่ยวมักต้องเปิดหลายเว็บเพื่อเช็กว่า "สถานที่นี้เปิดกี่โมง ค่าเข้าเท่าไร ปิดวันไหน" และข้อมูลแต่ละเว็บมักไม่ตรงกัน แชตบอตนี้รวบรวมข้อมูลไว้ในคลังเดียว ตอบจากเอกสารพร้อมอ้างอิงแหล่งที่มา และบอก "ไม่พบข้อมูลในเอกสาร" เมื่อไม่มีข้อมูล แทนการเดา ซึ่งสำคัญมากกับข้อมูลที่ผิดแล้วทำให้เสียเที่ยว เช่น เวลาเปิด-ปิด

## สถาปัตยกรรม
| ขั้นตอน | เทคนิค |
|---|---|
| Document Loading & Chunking | อ่านไฟล์ `data/*.md` → ทำความสะอาดด้วย PyThaiNLP `normalize` + ลบสัญลักษณ์ Markdown → แบ่ง chunk ~500 ตัวอักษร overlap 100 และใส่ชื่อสถานที่นำหน้าทุก chunk |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (รองรับไทย/อังกฤษ) |
| Vector Search | FAISS `IndexFlatIP` บนเวกเตอร์ที่ normalize แล้ว (cosine similarity), top-k ปรับได้ที่ sidebar |
| Prompt Engineering | System prompt บังคับตอบจาก context เท่านั้น อ้างอิงไฟล์ ตอบ "ไม่พบข้อมูลในเอกสาร" และแจ้งเมื่อข้อมูลขัดแย้งกัน |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Chatbot Interface | `st.chat_input` + `st.session_state` คุยต่อเนื่องได้ มีขั้นตอน query rewrite ให้คำถามต่อเนื่อง เช่น "แล้วค่าเข้าล่ะ" ค้นหาเจอ และแสดงเอกสารอ้างอิงทุกคำตอบ |

## โครงสร้างไฟล์
```
app.py
requirements.txt
README.md
test_questions.csv
data/                 # เอกสารความรู้ 17 ไฟล์ (~15,500 ตัวอักษร)
.streamlit/secrets.toml.example
```

## วิธีรันในเครื่อง
```bash
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# แก้ไฟล์ secrets.toml ใส่ GROQ_API_KEY (ขอฟรีที่ https://console.groq.com)
streamlit run app.py
```

## วิธี Deploy บน Streamlit Community Cloud
1. สร้าง GitHub repository (private) แล้ว push โค้ดทั้งหมด **ห้ามมีไฟล์ `secrets.toml` จริง** (ถูก `.gitignore` ไว้แล้ว)
2. เข้า https://share.streamlit.io → New app → เลือก repo, branch, และไฟล์ `app.py` (หาก repo เป็น private ต้องอนุญาตให้ Streamlit เข้าถึง repo)
3. Advanced settings → Secrets → ใส่
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy (ครั้งแรกจะช้าเพราะดาวน์โหลดโมเดล embedding)

## แหล่งที่มาของเอกสาร
ข้อมูลใน `data/` เรียบเรียงจากเว็บไซต์ท่องเที่ยวสาธารณะ (ปี 2024-2026) ได้แก่ TNN Thailand, TrueID Travel, Wongnai, Traveloka, Trip.com, ชิลไปไหน, Kapook Travel และ CBT Thailand (DASTA) เวลาเปิด-ปิดและค่าเข้าชมอาจเปลี่ยนแปลง บางสถานที่แหล่งข้อมูลไม่ตรงกัน (เช่น วัดต้นโพธิ์ศรีมหาโพธิ) ซึ่งระบุไว้ในเอกสารแล้ว ควรตรวจสอบกับสถานที่จริงก่อนเดินทาง

## ชุดทดสอบ
`test_questions.csv` มี 13 ข้อ (10 ข้อมีคำตอบในเอกสาร, 3 ข้อไม่มีคำตอบ เพื่อทดสอบว่าบอตตอบ "ไม่พบข้อมูลในเอกสาร")

## ตัวอย่าง Prompt ที่ใช้สั่ง AI
**Prompt สร้างโครงโปรเจกต์**
> ช่วยเขียน Streamlit RAG chatbot ไกด์ท่องเที่ยวปราจีนบุรี ใช้ sentence-transformers (multilingual) + FAISS + Groq API เก็บ key ใน st.secrets โหลดเอกสารจาก data/ ทำความสะอาดด้วย pythainlp แบ่ง chunk มี overlap แสดงเอกสารอ้างอิงทุกคำตอบ และเก็บประวัติแชตด้วย session_state

**Prompt สร้างเอกสารความรู้**
> รวบรวมข้อมูลสถานที่ท่องเที่ยวปราจีนบุรีจากเว็บท่องเที่ยว แยกไฟล์ละสถานที่ ระบุที่ตั้ง เวลาเปิด-ปิด ค่าเข้าชม และจุดเด่น ถ้าแหล่งข้อมูลไม่ตรงกันให้ระบุว่าไม่ตรงกัน ถ้าไม่มีข้อมูลให้เขียนว่าไม่พบข้อมูล ห้ามเดา

**System prompt ของแอป (ใน `app.py`)**
> ตอบจากเอกสารอ้างอิงเท่านั้น อ้างอิงชื่อไฟล์ท้ายคำตอบ ถ้าไม่มีคำตอบให้ตอบ "ไม่พบข้อมูลในเอกสาร" และแจ้งเมื่อข้อมูลขัดแย้งกัน
