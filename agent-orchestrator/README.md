# Agent Orchestrator

The brain of the system. It runs a **LangGraph ReAct agent** powered by **Claude 3 Sonnet via AWS Bedrock**. On each request it builds a graph, runs the Reason-Act loop until the LLM produces a final answer, and returns that answer to the Chat Service.

## How ReAct works

The ReAct (Reason + Act) pattern lets an LLM iteratively:

1. **Reason** — decide whether to answer directly or use a tool
2. **Act** — call a tool and observe its output
3. **Repeat** — reason again with the new observation until confident

```
System prompt
User message (+ history)
      │
      ▼
  ┌──────────┐
  │  agent   │  ←─────────────────────────────────────┐
  │ (Claude) │                                         │
  └────┬─────┘                                         │
       │  tool_calls present?                          │
      YES                                              │
       │                                               │
       ▼                                               │
  ┌──────────┐    POST /tools/call                     │
  │  tools   │ ──────────────────► MCP Server          │
  │  (node)  │ ◄────────────────── {result}            │
  └────┬─────┘                                         │
       └───────────────── ToolMessage ─────────────────┘
                                          │
                                     NO tool_calls
                                          │
                                          ▼
                                     Final reply → caller
```

## Responsibilities

- Fetch tool definitions from MCP Server once at startup
- Convert MCP JSON Schema tool definitions into LangChain `StructuredTool` objects
- Build a LangGraph `StateGraph` for each request
- Inject conversation history from Chat Service as prior `HumanMessage` / `AIMessage` nodes
- Dispatch tool calls to MCP Server via HTTP
- Extract and return the final plain-text reply

## Request / Response flow

```
Chat Service
  │
  │  POST /invoke  {message, session_id, history: [{role, content}, ...]}
  ▼
agent-orchestrator  (port 8002)
  │
  ├─① Build message list:
  │       SystemMessage(SYSTEM_PROMPT)
  │       HumanMessage(turn 1) / AIMessage(turn 1)   ← from history
  │       ...
  │       HumanMessage(current message)
  │
  ├─② graph.ainvoke(messages)    ← LangGraph async run
  │       │
  │       ├─ agent node: Claude decides → tool call
  │       ├─ tools node: call MCP Server → ToolMessage
  │       ├─ agent node: Claude reasons with result → another tool call (optional)
  │       ├─ tools node: ...
  │       └─ agent node: Claude produces final reply (no tool_calls) → END
  │
  ├─③ Walk result["messages"] in reverse to find last AIMessage without tool_calls
  └─④ Return {session_id, reply}
```

## Code walkthrough — `agent.py`

### Startup (`lifespan`)

```python
mcp_tools = await fetch_mcp_tools()
_langchain_tools = [make_langchain_tool(t) for t in mcp_tools]
```

At startup the orchestrator calls `POST mcp-server/tools/list`, receives the JSON Schema tool definitions, and converts them into LangChain `StructuredTool` objects stored in the module-level `_langchain_tools` list. If MCP is unreachable (e.g. first boot race), a warning is logged and the agent starts with no tools.

### `fetch_mcp_tools()` and `call_mcp_tool()`

`fetch_mcp_tools` does a single HTTP call to retrieve all tool definitions.  
`call_mcp_tool` is the coroutine that fires when LangChain decides to execute a tool. It serialises the arguments to JSON and POSTs them to `mcp-server/tools/call`, returning the result as a JSON string (which LangGraph wraps in a `ToolMessage`).

### `make_langchain_tool(tool_def)`

```python
def make_langchain_tool(tool_def: dict):
    input_model = create_model(f"{name}_input", **fields)   # dynamic Pydantic model
    async def _async_fn(**kwargs):
        return await call_mcp_tool(name, kwargs)
    return StructuredTool(name=name, description=description,
                          args_schema=input_model, coroutine=_async_fn)
```

This factory dynamically builds a Pydantic model from the MCP JSON Schema (`properties` + `required`). LangChain uses the model for argument validation and for generating the tool-call payload it sends to Claude. The async coroutine is the actual executor.

**Type mapping:**

| MCP JSON Schema type | Python type |
|---|---|
| `"integer"` | `int` |
| anything else | `str` |

### `AgentState`

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

LangGraph's `add_messages` reducer appends new messages to the list rather than replacing it — this is what makes the conversation accumulate across graph nodes.

### `build_graph(tools)`

```python
llm = ChatBedrock(model_id=..., model_kwargs={...}).bind_tools(tools)
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)   # → "tools" or END
graph.add_edge("tools", "agent")                        # always return to agent
```

- `ChatBedrock.bind_tools(tools)` informs Claude which tools are available by injecting their schemas into the system/user message.
- `ToolNode` is LangGraph's built-in node that dispatches `tool_calls` from the last `AIMessage` to the matching `StructuredTool` coroutines.
- `should_continue` checks whether the last message has `tool_calls`; if yes, route to `"tools"`, otherwise terminate.

### `should_continue` — the routing function

```python
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END
```

This is the only branching logic in the graph. The loop runs until Claude emits an `AIMessage` with an empty `tool_calls` list.

### `history_to_messages(history)`

```python
for item in history:
    if role == "user":      msgs.append(HumanMessage(content=content))
    elif role == "assistant": msgs.append(AIMessage(content=content))
```

Converts the flat `[{role, content}]` list from Chat Service / DynamoDB into typed LangChain message objects, preserving the full conversation context for Claude.

### Extracting the final reply

```python
for msg in reversed(result["messages"]):
    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
        reply = msg.content
        break
```

Walking backwards through the accumulated messages finds the last `AIMessage` that is *not* a tool-call request — that is Claude's final natural-language answer.

## API reference

### `GET /health`

```json
{
  "status": "ok",
  "service": "agent-orchestrator",
  "tools_loaded": ["search_knowledge_base", "echo", "web_search"]
}
```

### `POST /invoke`

**Request**
```json
{
  "message": "What does AWS Bedrock support?",
  "session_id": "abc-123",
  "history": [
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Response**
```json
{
  "session_id": "abc-123",
  "reply": "AWS Bedrock supports foundation models from Anthropic, Amazon, Meta, and others..."
}
```

**Error codes**

| Code | Meaning |
|---|---|
| `500` | LangGraph/Bedrock error (see logs for detail) |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `BEDROCK_LLM_MODEL_ID` | `anthropic.claude-3-sonnet-20240229-v1:0` | Claude model to use |
| `MCP_SERVER_URL` | `http://mcp-server:8003` | Where to fetch/call tools |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock calls |

## Changing the LLM

Any Bedrock model that supports the Converse API (tool use) works. Set `BEDROCK_LLM_MODEL_ID` to:

| Model | ID |
|---|---|
| Claude 3 Haiku (faster/cheaper) | `anthropic.claude-3-haiku-20240307-v1:0` |
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20240620-v1:0` |
| Amazon Nova Pro | `amazon.nova-pro-v1:0` |
| Llama 3 70B | `meta.llama3-70b-instruct-v1:0` |

## System prompt

```
You are a helpful AI assistant with access to a knowledge base and web search.
When answering questions, first check the knowledge base. Only use web_search for very recent events.
Be concise and factual. If you don't know something, say so.
```

Edit `SYSTEM_PROMPT` in `agent.py` to change the agent's persona, priorities, or tool usage strategy.

## Running locally (without Docker)

```bash
cd agent-orchestrator
pip install -r requirements.txt
MCP_SERVER_URL=http://localhost:8003 uvicorn agent:app --port 8002 --reload
```

## Scaling notes (beyond PoC)

| Concern | PoC approach | Production upgrade |
|---|---|---|
| Graph rebuild per request | `build_graph()` called every `/invoke` | Cache compiled graph; recreate only when tools change |
| Bedrock calls | Synchronous inside async handler | Already async via `ainvoke`; add retry logic with exponential back-off |
| Tool timeout | 30s per MCP call | Per-tool configurable timeouts; circuit breaker pattern |
| Observability | `logging` module | LangSmith tracing or AWS CloudWatch + X-Ray |
