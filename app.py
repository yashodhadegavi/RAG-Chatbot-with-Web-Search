import os
import tempfile

import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# duckduckgo-search was renamed to ddgs in newer versions; support both
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot (RAG + Web Search)")

# ---------- API key ----------
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()


# ---------- Model selection ----------
@st.cache_data(ttl=3600)
def list_chat_models(key):
    """Ask Groq which models this API key can actually use."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
    except Exception:
        return []
    skip = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "compound")
    return sorted(i for i in ids if not any(x in i.lower() for x in skip))


PREFERRED = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

available = list_chat_models(api_key)
if available:
    default_model = next((m for m in PREFERRED if m in available), available[0])
else:
    available = PREFERRED
    default_model = "llama-3.1-8b-instant"

with st.sidebar:
    MODEL_NAME = st.selectbox(
        "Groq model", available, index=available.index(default_model)
    )

llm = ChatGroq(api_key=api_key, model=MODEL_NAME, temperature=0.2)


# ---------- PDF / RAG helpers ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(pdf_file):
    """Returns (index, chunks, n_pages). index is None if no text was found."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    docs = [d for d in docs if d.page_content.strip()]  # drop empty pages
    if not docs:
        return None, [], 0  # scanned/image-only PDF

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    index = FAISS.from_documents(chunks, get_embeddings())
    return index, chunks, len(docs)


def web_search(query, n=4):
    try:
        results = DDGS().text(query, max_results=n)
        return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
    except Exception:
        return ""


# ---------- Sidebar: PDF upload ----------
with st.sidebar:
    st.header("📄 Documents")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf and st.session_state.get("pdf_name") != pdf.name:
        with st.spinner("Reading and indexing PDF..."):
            try:
                index, chunks, n_pages = build_index(pdf)
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.pdf_name = pdf.name
                st.session_state.n_pages = n_pages
            except Exception as e:
                st.session_state.index = None
                st.session_state.pdf_name = None
                st.error(f"Could not read this PDF: {e}")
    if not pdf:
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.pdf_name = None

    if st.session_state.get("index"):
        st.success(
            f"✅ {st.session_state.pdf_name}\n\n"
            f"{st.session_state.n_pages} pages, {len(st.session_state.chunks)} chunks indexed"
        )
    elif pdf and st.session_state.get("pdf_name") == pdf.name:
        st.warning(
            "No text found in this PDF. It is probably a scanned/image PDF. "
            "Try a PDF where you can select the text."
        )

    if st.button("Clear chat"):
        st.session_state.messages = []

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# ---------- Answer logic ----------
WEB_THRESHOLD = 1.6  # only search the web if the PDF match is really poor
SUMMARY_WORDS = ("summar", "overview", "about this", "what is this", "this pdf",
                 "this document", "this file", "explain the document")


def answer(question):
    parts, sources = [], []
    index = st.session_state.get("index")
    best = None

    # 1) If a PDF is loaded, ALWAYS use its content first
    if index:
        hits = index.similarity_search_with_score(question, k=5)
        docs = [d for d, _ in hits]
        best = min(score for _, score in hits) if hits else None

        # For "summarize this PDF" style questions, include the start of the document
        if any(w in question.lower() for w in SUMMARY_WORDS):
            docs = st.session_state.chunks[:6] + docs

        pdf_text = "\n\n".join(
            f"[Page {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs
        )
        parts.append("PDF CONTENT:\n" + pdf_text)
        sources.append("📄 your PDF")

    # 2) Use the web if there is no PDF, or the PDF match is poor
    if (not index) or (best is not None and best > WEB_THRESHOLD):
        web = web_search(question)
        if web:
            parts.append("WEB RESULTS:\n" + web)
            sources.append("🌐 web search")

    if not sources:
        sources.append("general knowledge")

    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]
    )

    prompt = f"""You are a helpful assistant. Answer using the context below.
If PDF CONTENT is provided, prefer it and mention page numbers when useful.
If the context doesn't contain the answer, say so briefly and answer from general knowledge.

Conversation so far:
{history}

Context:
{chr(10).join(parts)}

Question: {question}"""
    reply = llm.invoke(prompt).content
    return f"{reply}\n\n*Source: {' + '.join(sources)}*"


# ---------- Chat input ----------
if q := st.chat_input("Ask me anything..."):
    st.chat_message("user").write(q)
    with st.spinner("Thinking..."):
        a = answer(q)
    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("assistant").write(a)
    st.session_state.messages.append({"role": "assistant", "content": a})