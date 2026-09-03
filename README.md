# Doc Q&A Agent — LangChain + FastAPI + CI/CD

A small, complete example of the stack behind production GenAI applications:
a LangChain-based RAG chain, wrapped in a FastAPI service, with a CI/CD
pipeline that tests it on every push.

Builds directly on an earlier "Ask My Docs" RAG mini-project — this version
wraps that retrieval + generation logic in a proper LangChain chain and
exposes it as a real API instead of a command-line script.

## Why this project

Written to close three specific gaps: hands-on LangChain usage, a working
API (FastAPI), and a real CI/CD pipeline (GitHub Actions) — the core stack
behind most "GenAI Engineer" job postings, beyond just RAG/embeddings alone.

## Architecture

```
Documents (.txt)
      │
      ▼
 DocRetriever          — chunks docs, embeds with sentence-transformers,
                          stores/retrieves via ChromaDB
      │
      ▼
 LangChain RAG chain    — RunnableLambda (retrieve) → prompt template →
                          Gemini LLM → StrOutputParser
      │
      ▼
 FastAPI service        — POST /ask wraps the chain as a REST endpoint
      │
      ▼
 GitHub Actions CI      — runs pytest on every push, verifying the app
                          starts and responds correctly
```

## Tech stack

| Component        | Tool                                    |
|-------------------|-------------------------------------------|
| Agent framework   | LangChain (LCEL — chain composition)      |
| Embedding model   | sentence-transformers (all-MiniLM-L6-v2)  |
| Vector database   | ChromaDB                                  |
| LLM               | Google Gemini (`gemini-1.5-flash`) via `langchain-google-genai` |
| API framework     | FastAPI                                   |
| Testing           | pytest + FastAPI TestClient               |
| CI/CD             | GitHub Actions                            |

## Setup

```bash
pip install -r requirements.txt

# Get a free API key at https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your_key_here"
```

## Running locally

```bash
uvicorn main:app --reload
```

Then in another terminal:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Ask a question
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What tool was used for dashboards in the Tokyo Olympics project?"}'
```

## Running tests

```bash
pytest test_api.py -v
```

These tests don't require a Gemini API key — they verify the app starts
correctly and the health endpoint responds, which is exactly what the CI
pipeline checks on every push.

## CI/CD pipeline

See `.github/workflows/ci.yml`. On every push or pull request to `main`,
GitHub Actions:
1. Checks out the repo
2. Installs dependencies
3. Runs the test suite

This is a genuine, working CI/CD setup — not a description of one. Push
this repo to GitHub and the "Actions" tab will show it running.

## Design notes / what I'd improve next

- **Chain complexity**: this uses a single retrieve → generate chain. A true
  "agent" (as opposed to a fixed chain) would add tool-use and decision-making
  — e.g., a LangGraph agent that decides whether to retrieve, ask a
  clarifying question, or call an external tool.
- **Observability**: no tracing/evaluation tooling yet (e.g., LangSmith).
  Next step would be adding basic tracing to inspect retrieval quality and
  LLM latency per request.
- **Deployment**: this runs locally/via `uvicorn`. A production version
  would containerize it (Dockerfile) and deploy behind a proper ASGI server
  (e.g., Gunicorn + Uvicorn workers) with the CI pipeline extended to build
  and push a Docker image.
- **State**: the vector store is rebuilt in-memory on every app startup.
  A persistent ChromaDB instance (or hosted vector DB) would avoid
  re-embedding documents on every restart.

## Sample docs included

- `docs/tokyo_olympics_project.txt`
- `docs/customer_segmentation_project.txt`

Try asking: *"What tool was used for dashboards in the Tokyo Olympics project?"*
