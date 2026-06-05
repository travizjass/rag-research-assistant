# 🔍 Multi-Agent RAG Research Assistant

A production-style **Retrieval-Augmented Generation (RAG)** pipeline built with **LangGraph**, featuring a multi-agent architecture with **relevance grading**, **hallucination detection**, end-to-end **observability via LangFuse**, and a **live web UI that streams each agent's thinking in real time**.

Each agent is a node in a stateful graph connected by conditional edges and retry loops, so the system can re-retrieve when context is irrelevant and re-generate when an answer isn't grounded — before anything is returned to the user.

---

## 📖 How It Works (Complete Overview)

The project has **three moving parts** that work together:

```
   ┌──────────────┐         ┌─────────────────────────────┐         ┌──────────────┐
   │   Web UI     │  HTTP   │     FastAPI Backend         │  TCP    │    Qdrant    │
   │ (/ui/, SSE)  │ ──────► │  (LangGraph agent pipeline) │ ──────► │ (vector DB,  │
   │  browser     │ ◄────── │   uvicorn :8000             │ ◄────── │  Docker :6333)│
   └──────────────┘ stream  └─────────────────────────────┘         └──────────────┘
```

1. **Qdrant** (vector database) runs in Docker on port `6333`. It stores your documents as **embeddings** (numeric vectors) so they can be searched by meaning, not keywords.

2. **The FastAPI backend** (port `8000`) does two jobs:
   - **Ingestion** (a one-time CLI step): reads your PDF/TXT/DOCX files, splits them into overlapping chunks, embeds each chunk with Sentence-Transformers, and stores them in Qdrant.
   - **Serving the agent**: on each question it runs a **LangGraph pipeline** of five agents, and also **serves the web UI** as static files.

3. **The Web UI** (served by the backend at `/ui/`) lets you type a question and **watch each agent think step-by-step**, streamed live over Server-Sent Events (SSE).

### What happens on every question

```
Your question
  → Query Analyzer    : classifies it and rewrites it into a clean search query
  → Retriever         : embeds the query, pulls the top-k most similar chunks from Qdrant
  → Relevance Grader  : asks the LLM to keep only the chunks that truly help
        └─ if nothing relevant → retry retrieval (up to 2x, widening the search)
  → Generator         : writes an answer grounded ONLY in the kept chunks, with citations
  → Hallucination Chk : verifies every claim is supported by the context
        └─ if not grounded → re-generate (up to 2x)
  → Final answer + sources + scores
```

The whole point of the multi-agent design is **trustworthiness**: bad context is filtered out *before* generation, and ungrounded answers are caught *after* generation — so the model is far less likely to make things up.

---

## 🧠 Architecture

The pipeline is orchestrated as a **stateful LangGraph graph**. Every agent is a dedicated node; conditional edges decide whether to advance, retry retrieval, or regenerate.

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Analyzer │  ── Classifies the query & rewrites it for retrieval
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retriever    │  ── Fetches top-k chunks from Qdrant (widens on retry)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relevance       │  ── Grades each chunk: relevant / irrelevant
│ Grader          │
└────────┬────────┘
         │
    relevant?
    ├── No  ──► Retry Retrieval (max 2 loops, then proceed)
    │
    ▼ Yes
┌─────────────────┐
│   Generator     │  ── Synthesizes an answer from the graded context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hallucination   │  ── Verifies the answer is grounded in context
│ Checker         │
└────────┬────────┘
         │
    grounded?
    ├── No  ──► Re-generate (max 2 loops, then return)
    │
    ▼ Yes
 Final Answer
```

> When LangFuse keys are configured, every node is traced automatically — latency, token usage, and agent decisions. If keys are absent, tracing is silently skipped and the pipeline runs unchanged.

### The agents

| Node | File | Responsibility |
|---|---|---|
| **Query Analyzer** | [app/agents/query_analyzer.py](app/agents/query_analyzer.py) | Classifies the question and rewrites it into an optimised search query |
| **Retriever** | [app/agents/retriever.py](app/agents/retriever.py) | Dense semantic search over Qdrant; widens `top_k` on each retry |
| **Relevance Grader** | [app/agents/grader.py](app/agents/grader.py) | LLM-grades each chunk; keeps only relevant ones and computes a score |
| **Generator** | [app/agents/generator.py](app/agents/generator.py) | Writes the answer grounded **only** in the graded context, with citations |
| **Hallucination Checker** | [app/agents/hallucination_checker.py](app/agents/hallucination_checker.py) | Verifies every claim is supported by the context |

---

## ✨ Features

- **Multi-agent LangGraph pipeline** with conditional edges and bounded retry loops
- **Live streaming web UI** — watch each agent "think" in real time via Server-Sent Events
- **Relevance grading** — filters irrelevant retrieved chunks before generation
- **Hallucination detection** — verifies the answer is grounded in context before returning
- **LangFuse observability** — optional, graceful, full tracing of every node
- **Qdrant vector store** — dense semantic retrieval with Sentence Transformers
- **Groq inference** — fast, free-tier LLM (Llama 3.3-70B by default)
- **FastAPI backend** — JSON REST API + SSE streaming + auto-generated `/docs`
- **Document ingestion** — PDF, TXT, and DOCX via a CLI ingestion script
- **Fully configurable** — model, embeddings, retry budgets, and collection via env vars

---

## 🗂️ Project Structure

```
rag-research-assistant/
├── app/
│   ├── config.py                  # Central settings + cached Groq LLM factory
│   ├── main.py                    # FastAPI entrypoint; mounts the UI + CORS
│   │
│   ├── agents/
│   │   ├── query_analyzer.py
│   │   ├── retriever.py
│   │   ├── grader.py
│   │   ├── generator.py
│   │   └── hallucination_checker.py
│   │
│   ├── schemas/
│   │   └── models.py              # Request, response, grading & analysis models
│   │
│   ├── graph/
│   │   ├── state.py               # GraphState (LangGraph TypedDict)
│   │   └── pipeline.py            # Node wiring, conditional edges, retry logic
│   │
│   ├── vectorstore/
│   │   ├── embeddings.py          # Sentence-Transformers wrapper
│   │   └── qdrant_client.py       # Qdrant collection mgmt, upsert, search
│   │
│   ├── observability/
│   │   └── langfuse_tracer.py     # LangFuse v3 client + LangChain callback handler
│   │
│   └── api/
│       └── routes.py              # POST /query, POST /query/stream (SSE), GET /health
│
├── ui/                            # Static web UI (served by the backend at /ui/)
│   ├── index.html                 # Page structure
│   ├── styles.css                 # Dark theme + per-agent styling
│   └── app.js                     # Fetch-based SSE client; renders agent steps live
│
├── ingest/
│   └── document_loader.py         # CLI: parse → chunk → embed → upsert to Qdrant
│
├── sample_documents/              # Example PDF / TXT / DOCX to ingest & test
│
├── tests/
│   └── test_pipeline.py           # Offline unit tests (routing + schemas)
│
├── .env.example
├── .dockerignore
├── docker-compose.yml             # Qdrant service
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Groq — Llama 3.3-70B (`llama-3.3-70b-versatile`, free tier) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Store | Qdrant (local via Docker) |
| Observability | LangFuse v3 (cloud free tier, optional) |
| API Framework | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Web UI | Vanilla HTML + CSS + JavaScript (no build step) |
| Document Parsing | pdfplumber, python-docx |
| Language | Python 3.9+ |

---

## 🚀 Quick Start

> Prerequisites: **Python 3.9+**, **Docker Desktop** (running), and a free **Groq API key** from <https://console.groq.com>.

### 1. Install dependencies

```bash
git clone <repository-url>
cd rag-research-assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at least your Groq key (LangFuse keys are optional):

```env
GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key      # optional
LANGFUSE_SECRET_KEY=your_langfuse_secret_key      # optional
LANGFUSE_HOST=https://cloud.langfuse.com          # optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=research_docs
```

### 3. Start Qdrant (the vector database)

```bash
docker compose up -d
```

Verify it's healthy:

```bash
curl http://localhost:6333/healthz      # -> "healthz check passed"
```

Qdrant's dashboard is at <http://localhost:6333/dashboard>.

### 4. Ingest documents

Use the included samples (or point `--path` at your own folder of PDF/TXT/DOCX):

```bash
python -m ingest.document_loader --path ./sample_documents/
```

> ⚠️ The **first run downloads the embedding model (~90 MB)** — it will pause for a bit, that's normal. You'll see `Done. N chunks indexed.` when finished.

### 5. Run the backend (which also serves the UI)

```bash
uvicorn app.main:app --reload
```

### 6. Open the UI 🎉

```
http://localhost:8000/ui/
```

Type a question (or click an example chip) and **watch the agents work step by step**.

---

## 🧩 Running the Three Components

A common point of confusion: **you do not run a separate server for the UI** — the FastAPI backend serves it. So there are only **two processes** to run: Qdrant (Docker) and the backend (uvicorn).

| Component | What it is | How to start | Port |
|---|---|---|---|
| **Qdrant** | Vector database (Docker container) | `docker compose up -d` | `6333` (REST), `6334` (gRPC) |
| **Backend + UI** | FastAPI app running the LangGraph agent and serving the UI | `uvicorn app.main:app --reload` | `8000` |
| **Web UI** | Static files served by the backend | *(open in browser)* `http://localhost:8000/ui/` | — |

### Useful commands

```bash
# Qdrant
docker compose up -d        # start
docker compose ps           # status
docker compose logs -f      # logs
docker compose stop         # stop (keeps your data)
docker compose down         # remove container (keeps the data volume)
docker compose down -v      # remove container AND wipe all ingested data

# Backend
uvicorn app.main:app --reload                 # dev (auto-reloads on code change)
uvicorn app.main:app --host 0.0.0.0 --port 8000   # expose on your network

# Stop the backend if it's running in the background
lsof -ti:8000 | xargs kill
```

### Typical order of operations

```
docker compose up -d                                    # 1. start Qdrant
python -m ingest.document_loader --path ./sample_documents/   # 2. load data (once)
uvicorn app.main:app --reload                           # 3. start backend + UI
# 4. open http://localhost:8000/ui/ and ask questions
```

> Ingestion and querying must use the **same `COLLECTION_NAME`** (default `research_docs`). They do by default, since both read [app/config.py](app/config.py).

---

## 🖥️ The Web UI — "Agent Thinking" View

Open <http://localhost:8000/ui/> and ask a question. As the pipeline runs, each agent appears in a live timeline:

| Step | Shows |
|---|---|
| 🧭 **Query Analyzer** | The rewritten search query, query type, and any sub-questions |
| 📚 **Retriever** | How many chunks were pulled (and on which attempt), with snippets + scores |
| ⚖️ **Relevance Grader** | How many chunks were kept, the relevance score, and the surviving chunks |
| ✍️ **Generator** | The drafted answer + sources + attempt number |
| 🛡️ **Hallucination Checker** | The `grounded` / `ungrounded` verdict |

A final **Answer panel** then shows the consolidated result with badges (grounded status, relevance score, retry count). Retry loops appear naturally in the timeline — if the graph re-retrieves or re-generates, you'll see those nodes show up again.

**How it streams:** the UI calls `POST /query/stream`, and the backend uses LangGraph's `graph.stream(stream_mode="updates")` to emit a Server-Sent Event after **each node finishes**. Because SSE-over-POST can't use the browser's `EventSource`, [ui/app.js](ui/app.js) reads the `fetch` response body stream and parses the SSE frames manually.

> 💡 **Want to see it think longer?** The Relevance Grader makes **one LLM call per retrieved chunk**, so raising `top_k` (up to 20) makes the grading step take noticeably longer. To see the **retry loop** fire, ask an off-topic question — the grader rejects the irrelevant chunks and the graph retries retrieval up to 2× before giving up.

---

## 📡 API Usage

### `POST /query` — non-streaming

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?", "top_k": 5}'
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | The natural-language question (required) |
| `top_k` | int | `5` | Number of chunks to retrieve (1–20) |

**Response**

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is...",
  "sources": ["introduction_to_rag.txt", "transformer_architecture.pdf, page 1"],
  "grading_score": 0.8,
  "hallucination_check": "grounded",
  "retry_count": 0
}
```

| Field | Description |
|---|---|
| `answer` | The generated, context-grounded answer |
| `sources` | De-duplicated citations (`source, page N`) backing the answer |
| `grading_score` | Fraction of retrieved chunks judged relevant (0–1) |
| `hallucination_check` | `grounded` or `ungrounded` |
| `retry_count` | Total extra loops taken (retrieval + generation) |

### `POST /query/stream` — streaming (SSE)

Same request body as `/query`, but the response is a `text/event-stream` of events:

- `start` — the run has begun
- `step`  — one per node execution (includes the node's label and details)
- `done`  — the consolidated final result
- `error` — something failed

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How does self-attention work?", "top_k": 4}'
```

This is what the web UI consumes.

### `GET /health`

Liveness probe → `{"status": "ok"}`.

### Other endpoints

- `GET /` — JSON banner with links
- `GET /docs` — interactive Swagger UI
- `GET /ui/` — the streaming web UI

---

## 🔧 Configuration

All settings are read from the environment (via `.env`) in [app/config.py](app/config.py).

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key for the LLM |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq chat model id |
| `LLM_TEMPERATURE` | `0` | Base sampling temperature |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `COLLECTION_NAME` | `research_docs` | Qdrant collection name |
| `LANGFUSE_PUBLIC_KEY` | — | LangFuse public key (optional) |
| `LANGFUSE_SECRET_KEY` | — | LangFuse secret key (optional) |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | LangFuse host (`LANGFUSE_BASE_URL` also accepted) |
| `MAX_RETRIEVAL_RETRIES` | `2` | Max retrieval retry loops on irrelevant context |
| `MAX_GENERATION_RETRIES` | `2` | Max regeneration loops on hallucination |

---

## 🔁 Graph Flow — Conditional Logic

The retry behaviour lives in [app/graph/pipeline.py](app/graph/pipeline.py):

```python
# Retry retrieval if no relevant chunks survived grading
graph.add_conditional_edges(
    "grader",
    decide_after_grading,   # -> "generate" or "retry_retrieval"
    {"generate": "generator", "retry_retrieval": "retriever"},
)

# Re-generate if the answer isn't grounded in the context
graph.add_conditional_edges(
    "hallucination_checker",
    decide_after_check,     # -> "end" or "regenerate"
    {"end": END, "regenerate": "generator"},
)
```

Each loop is **bounded** by its retry budget (`MAX_RETRIEVAL_RETRIES` / `MAX_GENERATION_RETRIES`); once exhausted, the graph proceeds with the best available result instead of looping forever.

---

## 📊 Observability with LangFuse

When LangFuse keys are set, every run is traced via the LangChain callback handler (LangFuse v3 — see [app/observability/langfuse_tracer.py](app/observability/langfuse_tracer.py)):

- **Node-level latency** — see which agent is the bottleneck
- **Token usage per node** — track costs across Retriever, Generator, Checker
- **Retry tracking** — monitor how often the graph loops back
- **Full trace tree** — analyzer → retriever → grader → generator → checker

View traces in your [LangFuse dashboard](https://cloud.langfuse.com) after running queries.

---

## 🧪 Running Tests

The test suite mocks all network / LLM / vector-store calls, so it runs fully offline:

```bash
python -m unittest tests.test_pipeline -v
# or, if pytest is installed:
pytest -v
```

It covers the graph routing decisions (retry vs. proceed) and the core Pydantic schemas.

---

## 🩺 Troubleshooting

| Symptom | Fix |
|---|---|
| UI loads but queries hang/fail | Make sure the backend is running (`uvicorn ...`) and Qdrant is up (`docker compose ps`) |
| `Connection refused` to `:6333` | Qdrant isn't running — `docker compose up -d` |
| Empty / "no context" answers | You haven't ingested anything into `COLLECTION_NAME` yet — run the ingest step |
| First query/ingest is very slow | The embedding model (~90 MB) downloads once on first use; subsequent runs are fast |
| `GROQ_API_KEY` errors / 401 | Set a valid key in `.env` and restart the backend |
| Qdrant client/server version warning | Harmless; optionally `pip install -U qdrant-client` to match the server |
| Port 8000 already in use | `lsof -ti:8000 \| xargs kill`, then restart |

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
