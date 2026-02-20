import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
from PyPDF2 import PdfReader
from PIL import Image
import torch
from supabase import create_client

# LangChain (stable)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Transformers
from transformers import BlipProcessor, BlipForQuestionAnswering

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")
USERS_FILE = "users.json"

# -------------------- SUPABASE --------------------
SUPABASE_URL = "https://dwtklgdlykgwustgocba.supabase.co"
SUPABASE_KEY = "sb_publishable_RCdzFID7M8IePq0KpQAF3A_8-yvTNDB"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- HELPERS --------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_lottie(url):
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def type_text(text, speed=0.03):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### {out}")
        time.sleep(speed)

# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    return processor, model, device

# -------------------- SESSION DEFAULTS --------------------
defaults = {
    "authenticated": False,
    "users": load_users(),
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "user_id": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- AUTH UI --------------------
def login_ui():
    col1, col2 = st.columns(2)

    with col1:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"),
            height=300
        )

    with col2:
        type_text("🔐 Welcome to SlideSense")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u in st.session_state.users and st.session_state.users[u] == hash_password(p):
                    st.session_state.authenticated = True
                    st.session_state.user_id = hashlib.md5(u.encode()).hexdigest()[:32]
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                if nu in st.session_state.users:
                    st.warning("User already exists")
                else:
                    st.session_state.users[nu] = hash_password(np)
                    save_users(st.session_state.users)
                    st.success("Account created")

# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=20, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = f"""
Question: {question}
Vision Answer: {short_answer}
Convert into a clear natural sentence.
"""
    return llm.invoke(prompt).content

# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- HERO --------------------
col1, col2 = st.columns([1, 2])

with col1:
    st_lottie(
        load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
        height=250
    )

with col2:
    type_text("📘 SlideSense AI Platform")
    st.markdown("### Smart Learning | Smart Vision | Smart AI")

st.divider()

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )
                chunks = splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        q = st.text_input("Ask a question")

        if q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(q, k=5)

            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context": docs, "question": q})
            answer = res if isinstance(res, str) else res.get("output_text", "")

            # ---------- SAVE TO SUPABASE ----------
            supabase.table("pdf_chats").insert({
                "user_id": st.session_state.user_id,
                "question": q,
                "answer": answer
            }).execute()

            st.session_state.chat_history.append((q, answer))

        st.markdown("## 💬 Conversation")

        # ---------- LOAD FROM DB ----------
        db_data = supabase.table("pdf_chats")\
            .select("question,answer")\
            .eq("user_id", st.session_state.user_id)\
            .order("created_at", desc=True)\
            .limit(20)\
            .execute()

        for row in reversed(db_data.data):
            st.markdown(f"🧑 **You:** {row['question']}")
            st.markdown(f"🤖 **AI:** {row['answer']}")
            st.divider()

# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analyzing image..."):
                answer = answer_image_question(img, question)

                # ---------- SAVE TO SUPABASE ----------
                supabase.table("pdf_chats").insert({
                    "user_id": st.session_state.user_id,
                    "question": f"[IMAGE] {question}",
                    "answer": answer
                }).execute()

                st.success(answer)
