# Ingestion Service

Provides the RAG (Retrieval-Augmented Generation) pipeline for the system. Documents are chunked, embedded using Amazon Titan Embed v2 via AWS Bedrock, and stored in an in-memory FAISS index that is persisted to disk. The MCP Server calls `/search` at query time to retrieve relevant context for the agent.

## Responsibilities

- Accept raw text or uploaded files (PDF, TXT) for indexing
- Split content into overlapping chunks to preserve context across chunk boundaries
- Call Amazon Bedrock to embed each chunk using Titan Embed v2 (1024-dimensional vectors)
- Store vectors in a FAISS flat L2 index alongside chunk metadata
- Persist the index and metadata to disk so data survives container restarts
- Expose a `/search` endpoint that embeds a query and returns the top-k most similar chunks

## Request / Response flow

### Ingestion path

```
Client
  │
  │  POST /ingest/text  {"text": "...", "source": "aws-docs"}
  ▼
ingestion-service  (port 8001)
  │
  ├─① chunk_text(text)            → list of {text, source, start} dicts
  ├─② embed_texts(chunks)         → Bedrock InvokeModel (Titan Embed v2) × N calls
  │        └─ returns float32 numpy array  shape (N, 1024)
  ├─③ store.add(vectors, metas)   → FAISS IndexFlatL2.add + write to disk
  └─④ Return {"ingested_chunks": N, "total_vectors": total}
```

### Search path

```
MCP Server (or any caller)
  │
  │  POST /search  {"query": "What is Bedrock?", "top_k": 5}
  ▼
ingestion-service
  │
  ├─① embed_texts([query])        → single Bedrock embedding call
  ├─② store.search(vector, top_k) → FAISS index.search → distances + indices
  └─③ Return {"query": "...", "results": [{text, source, start, score}, ...]}
```

## Code walkthrough — `app.py`

### Constants

```python
EMBED_DIM    = 1024   # Titan Embed v2 output dimension
CHUNK_SIZE   = 512    # characters per chunk
CHUNK_OVERLAP = 64    # characters shared between consecutive chunks
```

Overlapping chunks ensure that sentences that fall across a chunk boundary are represented in both adjacent chunks, preventing information loss.

### `VectorStore` class

The class wraps a FAISS `IndexFlatL2` index (exact brute-force L2 search — suitable for PoC scale) together with a parallel Python list of metadata dicts.

```python
self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(EMBED_DIM)
self.metadata: list[dict] = []   # index i in metadata matches vector i in FAISS
```

**Why a parallel list instead of a FAISS payload?**  
FAISS only stores float vectors; it does not store arbitrary metadata. Keeping a Python list where `metadata[i]` corresponds to `index.reconstruct(i)` is the standard lightweight pattern for small-scale PoC use.

**Persistence**  
`save()` calls `faiss.write_index()` and writes metadata as JSON. `load()` is called once at startup via `lifespan`. Vectors are stored in the Docker volume `faiss_data` (see `docker-compose.yml`).

### `embed_texts(texts)`

```python
body = json.dumps({"inputText": text})
resp = client.invoke_model(modelId=settings.bedrock_embed_model_id, ...)
result = json.loads(resp["body"].read())
vectors.append(result["embedding"])
```

Each chunk is embedded with a separate Bedrock call (serial loop). For production, consider batching or parallelising these calls. The model is configurable via `BEDROCK_EMBED_MODEL_ID` (default: `amazon.titan-embed-text-v2:0`).

### `chunk_text(text, source)`

```
text = "ABCDE...Z"   (length > CHUNK_SIZE)

chunk 0: chars   0 → 512     (start=0)
chunk 1: chars 448 → 960     (start = 512 - 64 = 448)
chunk 2: chars 896 → 1408    (start = 960 - 64 = 896)
...
```

Each chunk dict also carries `source` (file name or label) and `start` (character offset) so search results can be traced back to the original document.

### Startup (`lifespan`)

```python
async def lifespan(app: FastAPI):
    store.load()   # reads index.faiss + metadata.json if they exist
    yield
```

If no persisted index exists (first run), `store.load()` is a no-op and the FAISS index starts empty.

### `POST /search` — scoring

FAISS returns L2 (squared Euclidean) distances. A **lower score means more similar**. Results are returned in ascending score order (best match first).

## API reference

### `GET /health`

```json
{"status": "ok", "service": "ingestion-service", "vectors_indexed": 42}
```

### `POST /ingest/text`

**Request**
```json
{"text": "Full document text here...", "source": "my-doc"}
```

**Response**
```json
{"ingested_chunks": 3, "total_vectors": 10}
```

### `POST /ingest/file`

Multipart form upload. Supports `.pdf` (via `pypdf`) and any plain-text file.

```bash
curl -X POST http://localhost:8001/ingest/file \
  -F "file=@report.pdf"
```

**Response**
```json
{"filename": "report.pdf", "ingested_chunks": 12, "total_vectors": 22}
```

### `POST /search`

**Request**
```json
{"query": "What is Bedrock?", "top_k": 3}
```

**Response**
```json
{
  "query": "What is Bedrock?",
  "results": [
    {"text": "AWS Bedrock is...", "source": "aws-docs", "start": 0, "score": 0.12},
    {"text": "Bedrock provides...", "source": "aws-docs", "start": 448, "score": 0.34}
  ]
}
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Embedding model |
| `VECTOR_STORE_PATH` | `/app/data/faiss_index` | Directory for `index.faiss` + `metadata.json` |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock calls |

## Running locally (without Docker)

```bash
cd ingestion-service
pip install -r requirements.txt
VECTOR_STORE_PATH=./data/faiss_index uvicorn app:app --port 8001 --reload
```

## Scaling notes (beyond PoC)

| Concern | PoC approach | Production upgrade |
|---|---|---|
| Vector store | In-process FAISS flat index | Amazon OpenSearch Serverless (k-NN plugin) |
| Embedding calls | Serial per-chunk | Async batch calls or Bedrock batch inference |
| File parsing | `pypdf` in-process | Pre-processing Lambda with S3 trigger |
| Index persistence | Docker volume | EFS or S3-backed store |
