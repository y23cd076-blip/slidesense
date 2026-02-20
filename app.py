import streamlit as st
from supabase import create_client, Client
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uuid
import time

# ==========================
# CONFIG
# ==========================
SUPABASE_URL = "https://dwtklgdlykgwustgocba.supabase.co"
SUPABASE_KEY = "sb_publishable_RCdzFID7M8IePq0KpQAF3A_8-yvTNDB"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="SlideSense AI", layout="wide")

# ==========================
# SESSION STATE
# ==========================
if "user" not in st.session_state:
    st.session_state.user = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ==========================
# AUTH UI
# ==========================
def auth_ui():
    st.title("🔐 SlideSense Login")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                user = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state.user = user.user
                st.success("Logged in successfully")
                st.rerun()
            except Exception as e:
                st.error("Login failed")

    with tab2:
        email = st.text_input("Signup Email")
        password = st.text_input("Signup Password", type="password")
        if st.button("Create Account"):
            try:
                supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                st.success("Account created. Please login.")
            except Exception as e:
                st.error("Signup failed")

# ==========================
# PDF PROCESSING
# ==========================
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

def build_faiss(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, model

# ==========================
# MAIN APP
# ==========================
def main_app():
    st.sidebar.title("📂 SlideSense")
    st.sidebar.write(f"Logged in as: {st.session_state.user.email}")

    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)
            index, model = build_faiss(chunks)

            st.session_state.faiss_index = index
            st.session_state.chunks = chunks
            st.session_state.model = model

        st.success("PDF processed successfully ✅")

    if st.session_state.faiss_index:
        st.subheader("💬 Ask your PDF")
        question = st.text_input("Ask a question")

        if st.button("Ask"):
            q_embed = st.session_state.model.encode([question])
            D, I = st.session_state.faiss_index.search(np.array(q_embed).astype("float32"), k=3)

            context = ""
            for idx in I[0]:
                context += st.session_state.chunks[idx] + "\n"

            answer = f"Context:\n{context}\n\nAnswer:\nAI Response based on content."

            st.write(answer)

            # Store in DB (user isolated)
            supabase.table("pdf_chats").insert({
                "id": str(uuid.uuid4()),
                "user_id": st.session_state.user.id,
                "question": question,
                "answer": answer,
                "pdf_name": uploaded_file.name
            }).execute()

    # ==========================
    # HISTORY
    # ==========================
    st.subheader("📜 Your Chat History")

    data = supabase.table("pdf_chats")\
        .select("*")\
        .eq("user_id", st.session_state.user.id)\
        .order("created_at", desc=True)\
        .execute()

    if data.data:
        for row in data.data:
            st.markdown(f"""
**📄 {row['pdf_name']}**  
**Q:** {row['question']}  
**A:** {row['answer']}  
---  
""")
    else:
        st.info("No chat history yet")

# ==========================
# ROUTER
# ==========================
if st.session_state.user is None:
    auth_ui()
else:
    main_app()
