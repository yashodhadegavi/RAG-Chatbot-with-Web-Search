# 🤖 RAG Chatbot with Web Search

**Chat with your PDFs. When the document doesn't have the answer, the bot searches the web, and always tells you where its answer came from.**

---

## 🎯 The Problem

Language models don't know the contents of your private documents, and their knowledge is frozen in time. Chatbots built only on a document fail on anything outside it; chatbots built only on a model make things up.

## 💡 The Solution

A hybrid **Retrieval-Augmented Generation (RAG)** chatbot:

1. **Reads your PDF** and answers from its content, with page references.
2. **Falls back to live web search** when the document has no good match.
3. **Labels every answer** with its source (📄 PDF, 🌐 web, or general knowledge), so answers can be verified.

## ✨ Key Features

- 📄 PDF upload with automatic text extraction, chunking and indexing
- 🔎 Semantic search over document chunks using FAISS
- 🌐 Automatic web search fallback (DuckDuckGo)
- 🏷️ Source attribution on every response
- 💬 Conversation memory for follow-up questions
- 🔀 Model selector that lists only models available to your API key
- ☁️ Deployed on Streamlit Community Cloud

## 🏗️ Architecture

```
User question
      │
      ▼
 PDF uploaded? ──Yes──► FAISS similarity search ──► Top relevant chunks ─┐
      │                                                                   │
      No / weak match                                                     ▼
      │                                                            Groq LLM ──► Answer + source
      ▼                                                                   ▲
 DuckDuckGo web search ───────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Groq API (Llama / GPT-OSS models) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| PDF Parsing | PyPDF |
| Web Search | DuckDuckGo Search |
| Deployment | Streamlit Community Cloud |

## 🧠 Engineering Decisions & Challenges

Real problems I hit while building this and how I solved them:

| Challenge | Solution |
|---|---|
| Strict similarity threshold caused the bot to ignore the PDF and search the web instead | Changed the logic so a loaded PDF is always used first; web search is added only when the best match is weak |
| Questions like "summarize this PDF" don't match any single chunk | Detected summary-style questions and included the start of the document as context |
| Hardcoded model name broke when the provider retired it (404 error) | Fetch the list of models available to the API key at runtime and let the user choose |
| Scanned/image-only PDFs produced empty indexes and crashes | Detect empty extraction and show a clear warning instead of failing |
| Search library was renamed between versions | Added an import fallback to support both package names |
| Large ML dependencies exceed free cloud limits | Used CPU-only PyTorch in `requirements.txt` |

## 🧰 Skills Demonstrated

`Generative AI` · `RAG` · `Vector Search` · `Embeddings` · `Prompt Design` · `LLM API Integration` · `Fallback Logic` · `Error Handling` · `Python` · `Cloud Deployment`

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/RAG-Chatbot-with-Web-Search.git
cd RAG-Chatbot-with-Web-Search

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)

pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` and add your free key from [console.groq.com](https://console.groq.com):

```toml
GROQ_API_KEY = "your_api_key_here"
```

```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py              # Streamlit app: RAG pipeline, web fallback, chat UI
├── requirements.txt    # Dependencies
└── README.md
```

## 🔮 Roadmap

- [ ] Multiple PDF support
- [ ] Streaming responses
- [ ] Page-level citations with highlights
- [ ] DOCX and TXT support


⭐ If you found this project useful, consider giving it a star.
