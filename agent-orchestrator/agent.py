"""
Agent Orchestrator — FastAPI + LangGraph ReAct Agent
Responsibilities:
  - POST /invoke  → run the ReAct agent loop using Claude via Bedrock
  - Agent fetches available tools from MCP Server at startup
  - Each tool call is forwarded to MCP Server /tools/call
  - Conversation history from Chat Service is fed into the agent context
"""
import os
import sys
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger("agent-orchestrator")
settings = get_settings()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base and web search.
When answering questions, first check the knowledge base. Only use web_search for very recent events.
Be concise and factual. If you don't know something, say so."""

# ---------------------------------------------------------------------------
# MCP tool loader — fetches tool definitions from the MCP server
# ---------------------------------------------------------------------------

async def fetch_mcp_tools() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{settings.mcp_server_url}/tools/list")
            resp.raise_for_status()
            return resp.json().get("tools", [])
    except Exception as exc:
        logger.warning("Could not fetch MCP tools: %s", exc)
        return []


async def call_mcp_tool(name: str, arguments: dict) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.mcp_server_url}/tools/call",
            json={"name": name, "arguments": arguments},
        )
        resp.raise_for_status()
        return json.dumps(resp.json().get("result", {}))


# ---------------------------------------------------------------------------
# Dynamic LangChain tool factory from MCP schema
# ---------------------------------------------------------------------------

def make_langchain_tool(tool_def: dict):
    """Create a LangChain StructuredTool from an MCP tool definition."""
    from langchain_core.tools import StructuredTool
    from pydantic import create_model, Field

    name: str = tool_def["name"]
    description: str = tool_def.get("description", "")
    schema = tool_def.get("inputSchema", {}).get("properties", {})
    required = tool_def.get("inputSchema", {}).get("required", [])

    # Build a Pydantic model for the tool's input schema
    fields = {}
    for prop_name, prop_schema in schema.items():
        annotation = str  # default to str
        if prop_schema.get("type") == "integer":
            annotation = int
        field_default = ... if prop_name in required else prop_schema.get("default", None)
        fields[prop_name] = (annotation, Field(default=field_default, description=prop_schema.get("description", "")))

    input_model = create_model(f"{name}_input", **fields)

    async def _async_fn(**kwargs):
        return await call_mcp_tool(name, kwargs)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=input_model,
        coroutine=_async_fn,
    )


# ---------------------------------------------------------------------------
# LangGraph agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Agent factory — rebuilt on each request (tools are cached at startup)
# ---------------------------------------------------------------------------

_langchain_tools: list = []


def build_graph(tools: list):
    """Build a LangGraph ReAct graph wired to the given tools."""
    llm_kwargs = dict(region_name=settings.aws_default_region)
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        llm_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        llm_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

    llm = ChatBedrock(
        model_id=settings.bedrock_llm_model_id,
        model_kwargs={"max_tokens": 2048, "temperature": 0.2},
        **llm_kwargs,
    ).bind_tools(tools)

    tool_node = ToolNode(tools)

    def call_model(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Lifespan: load tools once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _langchain_tools
    mcp_tools = await fetch_mcp_tools()
    _langchain_tools = [make_langchain_tool(t) for t in mcp_tools]
    logger.info("Loaded %d MCP tools: %s", len(_langchain_tools), [t.name for t in _langchain_tools])
    yield


app = FastAPI(title="Agent Orchestrator", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class InvokeRequest(BaseModel):
    message: str
    session_id: str
    history: list[dict] = []


class InvokeResponse(BaseModel):
    session_id: str
    reply: str


def history_to_messages(history: list[dict]) -> list[BaseMessage]:
    msgs = []
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "agent-orchestrator",
        "tools_loaded": [t.name for t in _langchain_tools],
    }


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest):
    graph = build_graph(_langchain_tools)

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history_to_messages(req.history))
    messages.append(HumanMessage(content=req.message))

    try:
        result = await graph.ainvoke({"messages": messages})
    except Exception as exc:
        logger.error("Agent graph error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    # Extract the final AI reply (last AIMessage without tool_calls)
    reply = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return InvokeResponse(session_id=req.session_id, reply=reply)
