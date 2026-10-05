# Chat Service

The user-facing entry point for the conversational AI system. It manages conversation sessions, persists history to DynamoDB, and delegates all AI reasoning to the Agent Orchestrator.

## Responsibilities

- Accept chat messages over HTTP (`POST /chat`)
- Auto-assign a `session_id` for new conversations or reuse one supplied by the caller
- Load the last 20 messages of conversation history from DynamoDB and forward them to the agent
- Save every user message and assistant reply to DynamoDB for future turns
- Return the agent's reply together with the `session_id` so the caller can continue the conversation

## Request / Response flow

```
Caller
  │
  │  POST /chat  {"message": "...", "session_id": "optional"}
  ▼
chat-service  (port 8080)
  │
  ├─① Generate or reuse session_id
  ├─② load_history(session_id)      → DynamoDB query (last 20 messages)
  ├─③ save_message(user, message)   → DynamoDB put_item
  ├─④ POST agent-orchestrator/invoke  {message, session_id, history}
  │        │
  │        └─ (agent thinks, may call tools, returns reply)
  │
  ├─⑤ save_message(assistant, reply) → DynamoDB put_item
  └─⑥ Return {"session_id": "...", "reply": "..."}
```

## Code walkthrough — `app.py`

### Startup (`lifespan`)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    dynamodb = get_dynamodb_resource()
    ensure_table(dynamodb)
    yield
```

FastAPI's `lifespan` context runs `ensure_table` once on startup. `ensure_table` calls `dynamodb.create_table(...)` and silently swallows `ResourceInUseException` so the service is idempotent whether or not the table already exists.

### DynamoDB schema

| Attribute | Type | Role |
|---|---|---|
| `session_id` | String | Partition key (HASH) — groups all messages for one conversation |
| `timestamp` | String (ISO-8601) | Sort key (RANGE) — orders messages chronologically |
| `role` | String | `"user"` or `"assistant"` |
| `content` | String | The message text |

`BillingMode: PAY_PER_REQUEST` means no capacity planning is needed for a PoC.

### `load_history(session_id, limit=20)`

Queries DynamoDB with `ScanIndexForward=True` (oldest → newest) and `Limit=20`. Returns a plain list of `{"role": ..., "content": ...}` dicts — exactly the format the Agent Orchestrator expects.

### `save_message(session_id, role, content)`

Writes a single item to DynamoDB with `datetime.utcnow().isoformat()` as the sort key. Because ISO-8601 strings sort lexicographically in the correct chronological order, no secondary index is needed.

### `POST /chat` handler

```
session_id = req.session_id or str(uuid.uuid4())
```

If the caller omits `session_id` a new UUID is generated, starting a fresh conversation. Both DynamoDB calls (`load_history`, `save_message`) are wrapped in `try/except` so a DynamoDB outage degrades gracefully rather than crashing the request.

The agent call uses `httpx.AsyncClient(timeout=60.0)` — 60 seconds allows the LangGraph loop to make several tool calls before timing out.

## API reference

### `GET /health`

```json
{"status": "ok", "service": "chat-service"}
```

### `POST /chat`

**Request**
```json
{
  "message": "What is AWS Bedrock?",
  "session_id": "optional-uuid-to-continue-a-conversation"
}
```

**Response**
```json
{
  "session_id": "3f2a1b4c-...",
  "reply": "AWS Bedrock is a fully managed service..."
}
```

**Error codes**

| Code | Meaning |
|---|---|
| `502` | Agent Orchestrator returned a non-2xx response |
| `503` | Agent Orchestrator is unreachable (network error / not running) |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `AGENT_ORCHESTRATOR_URL` | `http://agent-orchestrator:8002` | Where to forward chat requests |
| `DYNAMODB_TABLE_NAME` | `chat-history` | Table name |
| `DYNAMODB_ENDPOINT_URL` | _(empty = real AWS)_ | Set to `http://dynamodb-local:8000` for local dev |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |

## Running locally (without Docker)

```bash
cd chat-service
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in credentials
uvicorn app:app --port 8080 --reload
```

> The service will log a warning if DynamoDB is unreachable and continue without persistence.
