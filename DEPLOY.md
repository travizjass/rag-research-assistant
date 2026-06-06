# 🚀 Deploying the RAG Research Assistant

This app is **not deployable on Vercel** — it loads PyTorch / sentence-transformers
(well over Vercel's 250 MB serverless function limit) and needs a long-lived
Qdrant vector store, neither of which fits Vercel's stateless serverless model.

Instead it deploys cleanly to a **container platform** (Render, Railway, Fly.io)
with everything intact: local embeddings, the embedded Qdrant vector store, the
LangGraph multi-agent pipeline, and the live SSE-streaming UI.

The whole stack runs as **one Docker container**:

```
┌─────────────────────────────────────────────┐
│  one container                                │
│   ├─ Qdrant server (localhost:6333)           │
│   └─ FastAPI + UI (uvicorn on $PORT)          │
│        └─ ingests sample_documents/ on boot   │
└─────────────────────────────────────────────┘
```

See [Dockerfile](Dockerfile) and [start.sh](start.sh).

---

## Prerequisites

- A free **Groq API key** → <https://console.groq.com>
- Your code pushed to **GitHub** (Render/Railway deploy from a git repo)
- LangFuse keys are **optional** (tracing is skipped if absent)

> ⚠️ **Commit the previously-ignored files.** The old `.gitignore` excluded
> `__init__.py` and `sample_documents/`. This has been fixed — make sure they're
> committed, or the deploy will fail (missing packages / nothing to ingest):
>
> ```bash
> git add .gitignore Dockerfile start.sh render.yaml DEPLOY.md \
>         app/**/__init__.py ingest/__init__.py sample_documents/
> git commit -m "Add container deploy (Docker + Render) and track packages + samples"
> git push
> ```

---

## Option A — Render (recommended, free tier)

A [render.yaml](render.yaml) Blueprint is included, so it's almost one click.

1. Push to GitHub (see the commit step above).
2. In Render: **New +** → **Blueprint** → select this repo.
3. Render reads `render.yaml` and creates the service. When prompted, set the
   secret **`GROQ_API_KEY`** (and optionally the LangFuse keys).
4. Click **Apply**. The first build takes a few minutes (it installs PyTorch and
   pre-downloads the embedding model).
5. Open the service URL → the UI is at **`https://<your-app>.onrender.com/ui/`**.

**Notes**
- The free plan has **no persistent disk**, so the sample docs are re-ingested
  on each cold start (quick). Free services also **sleep after ~15 min idle** —
  the first request after sleeping is slow while it wakes + re-ingests.
- To **persist** the vector store and add your own documents permanently, switch
  `plan: free` → a paid tier in `render.yaml` and uncomment the `disk:` block.
- **RAM:** the free instance is 512 MB. The MiniLM embedder + embedded Qdrant fit,
  but it's tight; if you see out-of-memory restarts, use a larger instance.

---

## Option B — Railway

Railway auto-detects the `Dockerfile` — no extra config needed.

1. Push to GitHub.
2. In Railway: **New Project** → **Deploy from GitHub repo** → select this repo.
3. **Variables** tab → add `GROQ_API_KEY` (and optional LangFuse keys).
4. Railway builds the Dockerfile and injects `$PORT` (handled by `start.sh`).
5. Under **Settings → Networking**, generate a domain → open **`/ui/`** on it.

For persistence, add a **Volume** mounted at `/data` and set
`QDRANT_STORAGE_PATH=/data/qdrant` (already the default).

---

## Option C — Fly.io

```bash
fly launch --no-deploy          # detects the Dockerfile; edit fly.toml internal_port = 8000
fly secrets set GROQ_API_KEY=your_key
fly volumes create qdrant_data --size 1     # optional, for persistence
fly deploy
```

Mount the volume at `/data` in `fly.toml` to persist the vector store.

---

## Verify your deployment

```bash
# Liveness
curl https://<your-app>/health            # -> {"status":"ok"}

# Full pipeline end-to-end
curl -X POST https://<your-app>/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?", "top_k": 5}'
```

Then open **`https://<your-app>/ui/`** and watch the agents stream live.

---

## Test the container locally (same image the platform builds)

```bash
docker build -t rag-app .
# Pass the key UNQUOTED. Docker's --env-file keeps quotes literally (unlike
# python-dotenv), so a quoted GROQ_API_KEY in .env would 401 against Groq.
docker run --rm -p 8080:8000 \
  -e GROQ_API_KEY="$(grep '^GROQ_API_KEY=' .env | sed -E 's/^GROQ_API_KEY=//; s/^\"//; s/\"$//')" \
  -e PORT=8000 rag-app
# open http://localhost:8080/ui/
```

> On Render/Railway/Fly you paste the **raw** key value into the dashboard
> (no surrounding quotes), so this quoting caveat only affects the local
> `--env-file` shortcut.

---

## Production upgrade path

The single-container setup is ideal for a demo/portfolio deploy. For heavier
production use, split Qdrant into its own service with a persistent disk and
point the app at it via `QDRANT_HOST` / `QDRANT_PORT` (or use **Qdrant Cloud**) —
no app code changes required, since the connection is already env-driven
([app/vectorstore/qdrant_client.py](app/vectorstore/qdrant_client.py)).
