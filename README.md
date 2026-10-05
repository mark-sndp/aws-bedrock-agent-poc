# AWS Bedrock AgentCore — PoC

A proof-of-concept conversational AI system with RAG pipeline and agent architecture built on **AWS Bedrock**.

## Architecture

```
User
 │
 ▼
Chat Service (8080)
 │  stores history in DynamoDB
 │
 ▼
Agent Orchestrator (8002)  ← LangGraph ReAct (Claude 3 Sonnet)
 │
 ▼
MCP Server (8003)          ← tool registry & dispatcher
 │
 ▼
Ingestion Service (8001)   ← FAISS vector store + Titan Embed v2
```

| Service | Port | Responsibility |
|---|---|---|
| `chat-service` | 8080 | User-facing HTTP API, conversation history |
| `agent-orchestrator` | 8002 | LangGraph ReAct agent, Bedrock LLM |
| `mcp-server` | 8003 | Model Context Protocol tool server |
| `ingestion-service` | 8001 | Document ingestion, embeddings, vector search |
| `dynamodb-local` | 8000 | Local DynamoDB (dev only) |

## Quick Start

### 1. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_DEFAULT_REGION (default: us-east-1)
```

Make sure the IAM user/role has these permissions:
- `bedrock:InvokeModel` on `anthropic.claude-3-sonnet-*` and `amazon.titan-embed-text-*`
- `dynamodb:*` on the chat-history table (or use `dynamodb-local` for dev)

### 2. Start all services

```bash
docker compose up --build
```

### 3. Ingest some documents

```bash
# Ingest raw text
curl -X POST http://localhost:8001/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "AWS Bedrock is a fully managed service that makes foundation models available via API.", "source": "aws-docs"}'

# Ingest a file
curl -X POST http://localhost:8001/ingest/file \
  -F "file=@/path/to/your/document.pdf"
```

### 4. Chat

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AWS Bedrock?"}'
```

Continue a conversation by passing `session_id` from the previous response:

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you elaborate?", "session_id": "<session-id-from-above>"}'
```

## API Reference

### Chat Service (`http://localhost:8080`)

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Health check |
| POST | `/chat` | `{message, session_id?}` | Send a message |

### Ingestion Service (`http://localhost:8001`)

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Health + vector count |
| POST | `/ingest/text` | `{text, source?}` | Ingest raw text |
| POST | `/ingest/file` | multipart `file` | Ingest PDF or TXT |
| POST | `/search` | `{query, top_k?}` | Semantic search |

### MCP Server (`http://localhost:8003`)

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Lists available tools |
| POST | `/tools/list` | — | Full tool schemas |
| POST | `/tools/call` | `{name, arguments}` | Execute a tool |

### Agent Orchestrator (`http://localhost:8002`)

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Health + loaded tools |
| POST | `/invoke` | `{message, session_id, history?}` | Run agent |

## Project Structure

```
AWS-Bedrock/
├── shared/                  # Shared config and boto3 helpers
│   ├── config.py
│   ├── utils.py
│   └── requirements.txt
├── chat-service/            # User-facing API
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── ingestion-service/       # RAG pipeline
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── mcp-server/              # Tool server (MCP protocol)
│   ├── server.py
│   ├── requirements.txt
│   └── Dockerfile
├── agent-orchestrator/      # LangGraph ReAct agent
│   ├── agent.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Extending

- **Add a new tool**: Register it in `mcp-server/server.py` (`TOOLS` list + handler in `TOOL_HANDLERS`).
- **Swap the LLM**: Change `BEDROCK_LLM_MODEL_ID` in `.env`.
- **Production vector store**: Replace FAISS in `ingestion-service/app.py` with Amazon OpenSearch Serverless.
- **Production history**: Remove `DYNAMODB_ENDPOINT_URL` to use real DynamoDB.

## License

No license is granted for this project. It is provided for educational and
reference purposes only. See [LICENSE](LICENSE).
