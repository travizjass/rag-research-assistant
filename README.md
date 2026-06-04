# 🔍 Multi-Agent RAG Research Assistant

A production-style **Retrieval-Augmented Generation (RAG)** pipeline built with **LangGraph**, featuring a multi-agent architecture with relevance grading, hallucination detection, and full observability via **LangFuse**.

---

## 🧠 Architecture

The pipeline is orchestrated as a **stateful LangGraph graph** where each agent is a dedicated node connected via conditional edges and retry loops.

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Analyzer │  ── Decomposes & classifies the query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retriever    │  ── Fetches top-k chunks from Qdrant
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Relevance      │  ── Grades each chunk: relevant / irrelevant
│  Grader         │
└────────┬────────┘
         │
    relevant?
    ├── No  ──► Retry Retrieval (max 2 loops)
    │
    ▼ Yes
┌─────────────────┐
│   Generator     │  ── Synthesizes answer from graded context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hallucination  │  ── Checks if answer is grounded in context
│  Checker        │
└────────┬────────┘
         │
    grounded?
    ├── No  ──► Re-generate (max 2 loops)
    │
    ▼ Yes
 Final Answer
```

> All nodes are traced end-to-end via **LangFuse** — latency, token usage, prompt versions, and agent decisions are logged automatically.

---

## ✨ Features

- **Multi-agent LangGraph pipeline** with conditional edges and retry loops
- **Relevance grading** — filters irrelevant retrieved chunks before generation
- **Hallucination detection** — verifies answer is grounded in retrieved context before returning
- **LangFuse observability** — full tracing of every node, token usage, latency, and prompt versions
- **Qdrant vector store** — dense semantic retrieval using Sentence Transformers
- **Groq inference** — fast, free-tier LLM using Llama 3.3-70B
- **FastAPI backend** — streaming REST API ready for integration
- **Document ingestion** — supports PDF, TXT, and DOCX via CLI ingestion script

---

## 🗂️ Project Structure

```
rag-research-assistant/
├── app/
│   ├── agents/
│   │   ├── query_analyzer.py
│   │   ├── retriever.py
│   │   ├── grader.py
│   │   ├── generator.py
│   │   └── hallucination_checker.py
│   │
│   ├── schemas/
│   │   ├── query.py              # Request schemas
│   │   ├── response.py           # API response schemas
│   │   ├── state.py              # Graph state models
│   │   └── grading.py            # Relevance & hallucination outputs
│   │
│   ├── graph/
│   │   ├── state.py
│   │   └── pipeline.py
│   │
│   ├── vectorstore/
│   │   ├── embeddings.py
│   │   └── qdrant_client.py
│   │
│   ├── observability/
│   │   └── langfuse_tracer.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   └── main.py
│
├── ingest/
│   └── document_loader.py
│
├── docs/
│   └── architecture.png
│
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
│
├── .env.example
├── docker-compose.yml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph 0.2+ |
| LLM | Groq — Llama 3.3-70B (free tier) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | Qdrant (local via Docker) |
| Observability | LangFuse (cloud free tier) |
| API Framework | FastAPI + Uvicorn |
| Document Parsing | pdfplumber, python-docx |
| Language | Python 3.11+ |

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/travizjass/rag-research-assistant.git
cd rag-research-assistant
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in your `.env`:

```env
GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=research_docs
```

### 4. Start Qdrant via Docker

```bash
docker-compose up -d
```

### 5. Ingest documents

```bash
python -m ingest.document_loader --path ./your-documents/
```

### 6. Run the API

```bash
uvicorn app.main:app --reload
```

API will be live at `http://localhost:8000`

---

## 📡 API Usage

### Query endpoint

```bash
POST /query
Content-Type: application/json

{
  "query": "What are the key findings in the research paper?",
  "top_k": 5
}
```

### Response

```json
{
  "answer": "The key findings include...",
  "sources": ["doc1.pdf, page 3", "doc2.pdf, page 7"],
  "grading_score": 0.91,
  "hallucination_check": "grounded",
  "retry_count": 0
}
```

---

## 📊 Observability with LangFuse

Every pipeline run is fully traced in LangFuse:

- **Node-level latency** — see which agent is the bottleneck
- **Token usage per node** — track costs across Retriever, Generator, Checker
- **Prompt versioning** — compare prompt changes across runs
- **Retry tracking** — monitor how often the graph loops back

Visit your [LangFuse dashboard](https://cloud.langfuse.com) to view traces after running queries.

---

## 🔁 Graph Flow — Conditional Logic

```python
# Retry retrieval if no relevant chunks found
graph.add_conditional_edges(
    "grader",
    decide_after_grading,   # returns "generate" or "retry_retrieval"
    {"generate": "generator", "retry_retrieval": "retriever"}
)

# Re-generate if hallucination detected
graph.add_conditional_edges(
    "hallucination_checker",
    decide_after_check,     # returns "end" or "regenerate"
    {"end": END, "regenerate": "generator"}
)
```

---

## 📦 Requirements

```
langgraph>=0.2.0
langchain>=0.2.0
langchain-groq
langfuse
qdrant-client
sentence-transformers
fastapi
uvicorn
pdfplumber
python-docx
python-dotenv
```

---

## 📄 License

MIT License — free to use and modify.

---

## 👤 Author

**Jasmin Bheda**
[LinkedIn](https://www.linkedin.com/in/jasmin-bheda/) · [GitHub](https://github.com/travizjass)