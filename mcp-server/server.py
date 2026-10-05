"""
MCP Server — FastAPI implementing a lightweight Model Context Protocol.

Exposes tools the Agent Orchestrator can call:
  POST /tools/list          → enumerate available tools
  POST /tools/call          → execute a named tool with arguments

Tools provided:
  search_knowledge_base(query, top_k)  → calls Ingestion Service /search
  echo(message)                         → reflects message back (smoke-test tool)
  web_search(query)                     → stub (returns placeholder; replace with real API)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger("mcp-server")
settings = get_settings()

app = FastAPI(title="MCP Server", version="0.1.0")

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal knowledge base for information relevant to a query. "
            "Use this when the user asks factual questions that may be answered by ingested documents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "echo",
        "description": "Reflect the given message back unchanged. Useful for testing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information not available in the knowledge base. "
            "Returns a stub response in this PoC — replace with a real search API integration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def tool_search_knowledge_base(args: dict) -> dict:
    query = args.get("query", "")
    top_k = int(args.get("top_k", 5))
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.ingestion_service_url}/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()


async def tool_echo(args: dict) -> dict:
    return {"echo": args.get("message", "")}


async def tool_web_search(args: dict) -> dict:
    query = args.get("query", "")
    logger.info("web_search stub called with query: %s", query)
    return {
        "stub": True,
        "message": (
            f"Web search for '{query}' is not yet implemented. "
            "Integrate a search provider (e.g. Brave Search, Tavily) here."
        ),
    }


TOOL_HANDLERS = {
    "search_knowledge_base": tool_search_knowledge_base,
    "echo": tool_echo,
    "web_search": tool_web_search,
}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "mcp-server", "tools": [t["name"] for t in TOOLS]}


@app.post("/tools/list")
async def list_tools():
    return {"tools": TOOLS}


@app.post("/tools/call")
async def call_tool(req: ToolCallRequest):
    handler = TOOL_HANDLERS.get(req.name)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {req.name}")
    try:
        result = await handler(req.arguments)
    except httpx.HTTPStatusError as exc:
        logger.error("Tool '%s' upstream error: %s", req.name, exc)
        raise HTTPException(status_code=502, detail=f"Tool upstream error: {exc}")
    except Exception as exc:
        logger.error("Tool '%s' execution error: %s", req.name, exc)
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {exc}")
    return {"tool": req.name, "result": result}
