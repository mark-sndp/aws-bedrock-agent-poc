"""
Ingestion Service — FastAPI
Responsibilities:
  - POST /ingest/text  → chunk, embed, store in FAISS
  - POST /ingest/file  → parse PDF/TXT, chunk, embed, store
  - POST /search       → embed query, return top-k similar chunks
  - GET  /health
"""
import os
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from shared.config import get_settings
from shared.utils import get_logger, get_bedrock_runtime_client

logger = get_logger("ingestion-service")
settings = get_settings()

VECTOR_STORE_PATH = Path(settings.vector_store_path)
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
INDEX_FILE = VECTOR_STORE_PATH / "index.faiss"
META_FILE = VECTOR_STORE_PATH / "metadata.json"

EMBED_DIM = 1024  # Titan Embed v2 dimension
CHUNK_SIZE = 512   # characters per chunk
CHUNK_OVERLAP = 64


# ---------------------------------------------------------------------------
# In-memory store (loaded from disk at startup)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self):
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(EMBED_DIM)
        self.metadata: list[dict] = []  # parallel list to FAISS vectors

    def load(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            self.metadata = json.loads(META_FILE.read_text())
            logger.info("Loaded %d vectors from disk", self.index.ntotal)

    def save(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(self.metadata))

    def add(self, vectors: np.ndarray, metas: list[dict]):
        self.index.add(vectors)
        self.metadata.extend(metas)
        self.save()

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "score": float(dist)})
        return results


store = VectorStore()


# ---------------------------------------------------------------------------
# Bedrock embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using Amazon Titan Embed v2 via Bedrock."""
    client = get_bedrock_runtime_client()
    vectors = []
    for text in texts:
        body = json.dumps({"inputText": text})
        resp = client.invoke_model(
            modelId=settings.bedrock_embed_model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(resp["body"].read())
        vectors.append(result["embedding"])
    return np.array(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str = "unknown") -> list[dict]:
    """Split text into overlapping chunks and attach metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({"text": chunk, "source": source, "start": start})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(title="Ingestion Service", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class IngestTextRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ingestion-service", "vectors_indexed": store.index.ntotal}


@app.post("/ingest/text")
async def ingest_text(req: IngestTextRequest):
    chunks = chunk_text(req.text, source=req.source)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text to ingest")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    store.add(vectors, chunks)
    logger.info("Ingested %d chunks from source '%s'", len(chunks), req.source)
    return {"ingested_chunks": len(chunks), "total_vectors": store.index.ntotal}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or "upload"

    if filename.lower().endswith(".pdf"):
        try:
            import pypdf
            from io import BytesIO
            reader = pypdf.PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(status_code=501, detail="pypdf not installed; only .txt supported")
    else:
        text = content.decode("utf-8", errors="replace")

    chunks = chunk_text(text, source=filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from file")
    vectors = embed_texts([c["text"] for c in chunks])
    store.add(vectors, chunks)
    logger.info("Ingested file '%s': %d chunks", filename, len(chunks))
    return {"filename": filename, "ingested_chunks": len(chunks), "total_vectors": store.index.ntotal}


@app.post("/search")
async def search(req: SearchRequest):
    vector = embed_texts([req.query])
    results = store.search(vector, top_k=req.top_k)
    return {"query": req.query, "results": results}
