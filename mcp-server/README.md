# MCP Server

A lightweight **Model Context Protocol (MCP)** server. It acts as the tool registry and dispatcher for the Agent Orchestrator: the agent asks "what tools exist?" via `/tools/list`, then invokes them via `/tools/call`. Adding a new capability to the entire system only requires adding one entry here.

## What is MCP?

The Model Context Protocol defines a standard way for LLM agents to discover and call external tools. Each tool has:

- A **name** (the identifier the agent uses)
- A **description** (what the LLM reads to decide whether to use it)
- An **inputSchema** (JSON Schema describing the required arguments)

The MCP Server decouples tool *definition* from tool *execution*, so any downstream system (or a human developer) can enumerate available capabilities without knowing how they are implemented.

## Responsibilities

- Maintain a registry of available tools with their JSON Schema definitions
- Serve the tool catalogue via `POST /tools/list`
- Dispatch `POST /tools/call` requests to the correct handler function
- Forward `search_knowledge_base` calls to the Ingestion Service
- Provide a simple `echo` tool for smoke-testing the agent-tools pipeline
- Provide a `web_search` stub that can be wired to a real search API

## Request / Response flow

```
Agent Orchestrator
  │
  │─── startup ────────────────────────────────────────────────────────
  │  POST /tools/list
  │  ← [{name, description, inputSchema}, ...]
  │
  │─── per agent turn (when the LLM decides to use a tool) ──────────
  │  POST /tools/call  {"name": "search_knowledge_base", "arguments": {"query": "..."}}
  │        │
  │        ├─ tool_search_knowledge_base(args)
  │        │       └─ POST ingestion-service/search {"query": ..., "top_k": ...}
  │        │              └─ returns [{text, source, score}, ...]
  │        └─ Return {"tool": "search_knowledge_base", "result": {...}}
```

## Code walkthrough — `server.py`

### Tool registry (`TOOLS`)

```python
TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "...",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", ...},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    ...
]
```

`TOOLS` is a plain Python list of dicts. It is the single source of truth for what the agent knows it can call. The `description` field is particularly important — the LLM reads it verbatim when deciding which tool to invoke, so it should be action-oriented and specific.

### Tool handlers (`TOOL_HANDLERS`)

```python
TOOL_HANDLERS = {
    "search_knowledge_base": tool_search_knowledge_base,
    "echo":                  tool_echo,
    "web_search":            tool_web_search,
}
```

A simple dict maps tool names to their async handler functions. Every handler has the same signature:

```python
async def tool_xxx(args: dict) -> dict: ...
```

This makes it trivial to add new tools — define the handler function, add it to `TOOLS` and `TOOL_HANDLERS`.

### `tool_search_knowledge_base`

```python
async with httpx.AsyncClient(timeout=30.0) as client:
    resp = await client.post(
        f"{settings.ingestion_service_url}/search",
        json={"query": query, "top_k": top_k},
    )
```

Proxies the call to the Ingestion Service and returns its JSON response unchanged. The 30-second timeout accommodates multiple Bedrock embedding calls inside the Ingestion Service.

### `tool_echo`

```python
return {"echo": args.get("message", "")}
```

Trivial reflector. Its purpose is to let you verify the entire agent → MCP → tool pipeline works end-to-end before integrating real tools.

### `tool_web_search`

Returns a stub message noting that a real integration is needed. To replace it, swap the stub body with a call to a provider such as:

- **Tavily** (`pip install tavily-python`)
- **Brave Search API** (HTTP call with API key)
- **SerpAPI**

### `POST /tools/call` dispatcher

```python
handler = TOOL_HANDLERS.get(req.name)
if handler is None:
    raise HTTPException(status_code=404, detail=f"Unknown tool: {req.name}")
result = await handler(req.arguments)
```

- Returns `404` for unknown tool names (prevents silent failures)
- Returns `502` when an upstream service (e.g. Ingestion Service) returns an error
- Returns `500` for unexpected handler exceptions

## API reference

### `GET /health`

```json
{"status": "ok", "service": "mcp-server", "tools": ["search_knowledge_base", "echo", "web_search"]}
```

### `POST /tools/list`

No request body required.

**Response**
```json
{
  "tools": [
    {
      "name": "search_knowledge_base",
      "description": "Search the internal knowledge base...",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "The search query"},
          "top_k": {"type": "integer", "description": "Number of results to return", "default": 5}
        },
        "required": ["query"]
      }
    }
  ]
}
```

### `POST /tools/call`

**Request**
```json
{"name": "search_knowledge_base", "arguments": {"query": "What is Bedrock?", "top_k": 3}}
```

**Response**
```json
{
  "tool": "search_knowledge_base",
  "result": {
    "query": "What is Bedrock?",
    "results": [{"text": "...", "source": "...", "score": 0.12}]
  }
}
```

**Error codes**

| Code | Meaning |
|---|---|
| `404` | Tool name not found in registry |
| `502` | Upstream service (Ingestion Service) returned an error |
| `500` | Handler raised an unexpected exception |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `INGESTION_SERVICE_URL` | `http://ingestion-service:8001` | Where `search_knowledge_base` forwards calls |

## Adding a new tool (step-by-step)

1. **Write the handler** in `server.py`:

   ```python
   async def tool_get_weather(args: dict) -> dict:
       city = args.get("city", "")
       # call a weather API here
       return {"temperature": "22°C", "condition": "Sunny"}
   ```

2. **Register the schema** in `TOOLS`:

   ```python
   {
       "name": "get_weather",
       "description": "Get current weather for a city. Use when the user asks about weather.",
       "inputSchema": {
           "type": "object",
           "properties": {
               "city": {"type": "string", "description": "City name"}
           },
           "required": ["city"],
       },
   },
   ```

3. **Register the handler** in `TOOL_HANDLERS`:

   ```python
   "get_weather": tool_get_weather,
   ```

4. **Restart the MCP Server.** The Agent Orchestrator fetches the tool list on startup, so restart it too so it picks up the new tool.

## Running locally (without Docker)

```bash
cd mcp-server
pip install -r requirements.txt
INGESTION_SERVICE_URL=http://localhost:8001 uvicorn server:app --port 8003 --reload
```
