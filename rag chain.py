"""
rag_chain.py — LangChain-based RAG chain.

Wraps the retrieval + generation logic (from the earlier "Ask My Docs"
mini-project) inside a LangChain chain, so it can be reused as a proper
agent component rather than raw function calls.

Chain structure:
  retriever (ChromaDB + sentence-transformers embeddings)
        │
        ▼
  prompt template (injects retrieved context + question)
        │
        ▼
  LLM (Gemini, via LangChain's ChatGoogleGenerativeAI)
        │
        ▼
  output parser (plain string answer)
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


DOCS_FOLDER = "docs"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"


class DocRetriever:
    """Loads, chunks, and embeds documents; retrieves top-k relevant chunks for a query."""

    def __init__(self, folder_path=DOCS_FOLDER):
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("doc_qa")
        except Exception:
            pass
        self.collection = self.client.create_collection("doc_qa")
        self._load_and_index(folder_path)

    def _load_and_index(self, folder_path):
        chunks, metadata = [], []
        filepaths = glob.glob(os.path.join(folder_path, "*.txt"))
        if not filepaths:
            raise FileNotFoundError(f"No .txt files found in '{folder_path}'.")

        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            words = text.split()
            start = 0
            while start < len(words):
                end = start + CHUNK_SIZE
                chunks.append(" ".join(words[start:end]))
                metadata.append({"source": os.path.basename(filepath)})
                start += CHUNK_SIZE - CHUNK_OVERLAP

        embeddings = self.embed_model.encode(chunks, show_progress_bar=False).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_embedding = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        docs = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]
        return docs, sources


def format_context(retrieval_result):
    docs, sources = retrieval_result
    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(docs, sources)
    )
    return {"context": context, "sources": sources}


def build_rag_chain(retriever: DocRetriever):
    """Builds a LangChain runnable chain: retrieve -> prompt -> LLM -> parse."""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    )

    def retrieve_and_format(inputs):
        query = inputs["question"]
        result = retriever.retrieve(query)
        formatted = format_context(result)
        return {"context": formatted["context"], "question": query, "sources": formatted["sources"]}

    chain = (
        RunnableLambda(retrieve_and_format)
        | RunnablePassthrough.assign(
            answer=(lambda x: {"context": x["context"], "question": x["question"]})
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return chain
