# 📄 StudyMate AI

**StudyMate AI** is a Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it in natural language. Answers are grounded strictly in the document's content and cited with page numbers — no hallucinated responses.

Built as a study/research companion: upload lecture notes, textbooks, or research papers, and get instant, accurate answers instead of manually searching through pages.

---

## ✨ Features

- 📂 **Upload any PDF** and get it indexed in seconds
- 💬 **Chat interface** to ask natural-language questions about the document
- 🎯 **Grounded answers only** — responses are generated strictly from retrieved context, reducing hallucinations
- 📌 **Page-number citations** for every answer, so you can verify the source instantly
- ⚡ **Low-latency streamed responses** powered by Groq
- 🧠 **Semantic search** via vector embeddings, not simple keyword matching

---

## 🏗️ How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage → Semantic Search → LLM Response
```

1. **Text Extraction** — The uploaded PDF is parsed page-by-page using **PyMuPDF**.
2. **Chunking** — Extracted text is split into overlapping chunks (1000 characters, 400 overlap) using LangChain's `RecursiveCharacterTextSplitter`, preserving context across chunk boundaries.
3. **Embedding** — Each chunk is converted into a vector using the `all-MiniLM-L6-v2` HuggingFace sentence-transformer model (runs locally, no API cost).
4. **Vector Storage** — Embeddings are stored in **Qdrant**, a high-performance vector database.
5. **Retrieval** — On each user query, the most semantically relevant chunks are retrieved from Qdrant via similarity search.
6. **Generation** — Retrieved context + the user's question are passed to **Groq's LLM API**, which streams back a concise, cited answer in real time.

---

## 🛠️ Tech Stack

| Layer            | Technology                                  |
|-------------------|----------------------------------------------|
| Frontend / UI     | Streamlit                                    |
| Orchestration     | LangChain                                    |
| PDF Parsing       | PyMuPDF (fitz)                               |
| Embeddings        | HuggingFace `all-MiniLM-L6-v2`               |
| Vector Database   | Qdrant                                       |
| LLM Inference     | Groq (`groq/compound-mini`)                  |
| Environment Mgmt  | python-dotenv                                |
| Containerization  | Docker (Qdrant service via docker-compose)   |

---

## 📦 Prerequisites

- Python 3.10+
- Docker & Docker Compose (for running Qdrant locally) — *or* a free [Qdrant Cloud](https://cloud.qdrant.io/) instance
- A free [Groq API key](https://console.groq.com/keys)
