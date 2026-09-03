"""
main.py — FastAPI service exposing the LangChain RAG chain.

Endpoints:
  GET  /health        — health check (used by CI/CD)
  POST /ask            — ask a question, get a grounded answer

Run locally:
  pip install -r requirements.txt
  export GEMINI_API_KEY="your_key"
  uvicorn main:app --reload

Then:
  curl -X POST http://127.0.0.1:8000/ask \
       -H "Content-Type: application/json" \
       -d '{"question": "What tool was used for dashboards in the Tokyo Olympics project?"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag_chain import DocRetriever, build_rag_chain

# Global state, initialized once at startup (avoids re-embedding docs per request)
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build the retriever and chain once
    retriever = DocRetriever()
    state["retriever"] = retriever
    try:
        state["chain"] = build_rag_chain(retriever)
    except EnvironmentError:
        # Allow the app to start even without an API key set,
        # so /health still works — /ask will report the issue clearly.
        state["chain"] = None
    yield
    state.clear()


app = FastAPI(title="Doc Q&A Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """Simple health check — used by the CI/CD pipeline to verify the app starts correctly."""
    return {"status": "ok", "chain_ready": state.get("chain") is not None}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if state.get("chain") is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialized — GEMINI_API_KEY is not set on the server.",
        )

    result = state["chain"].invoke({"question": request.question})
    return AskResponse(answer=result["answer"], sources=result["sources"])
