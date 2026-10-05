"""
Chat Service — FastAPI
Responsibilities:
  - Accept user messages via POST /chat
  - Persist conversation history in DynamoDB
  - Forward request to Agent Orchestrator and return the reply
"""
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from botocore.exceptions import ClientError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import get_settings
from shared.utils import get_logger, get_dynamodb_resource

logger = get_logger("chat-service")
settings = get_settings()

# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def ensure_table(dynamodb):
    """Create the chat-history table if it doesn't already exist."""
    try:
        table = dynamodb.create_table(
            TableName=settings.dynamodb_table_name,
            KeySchema=[
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        logger.info("DynamoDB table created: %s", settings.dynamodb_table_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            logger.info("DynamoDB table already exists: %s", settings.dynamodb_table_name)
        else:
            raise


def save_message(session_id: str, role: str, content: str):
    import datetime
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(settings.dynamodb_table_name)
    table.put_item(Item={
        "session_id": session_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "role": role,
        "content": content,
    })


def load_history(session_id: str, limit: int = 20) -> list[dict]:
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(settings.dynamodb_table_name)
    resp = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(session_id),
        ScanIndexForward=True,
        Limit=limit,
    )
    return [{"role": item["role"], "content": item["content"]} for item in resp.get("Items", [])]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        dynamodb = get_dynamodb_resource()
        ensure_table(dynamodb)
    except Exception as exc:
        logger.warning("Could not initialise DynamoDB (running without persistence): %s", exc)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Chat Service", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chat-service"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Load conversation history
    try:
        history = load_history(session_id)
    except Exception as exc:
        logger.warning("Could not load history: %s", exc)
        history = []

    # Persist user message
    try:
        save_message(session_id, "user", req.message)
    except Exception as exc:
        logger.warning("Could not save user message: %s", exc)

    # Forward to agent orchestrator
    payload = {
        "message": req.message,
        "session_id": session_id,
        "history": history,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.agent_orchestrator_url}/invoke",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Agent orchestrator error: %s", exc)
        raise HTTPException(status_code=502, detail="Agent orchestrator returned an error")
    except Exception as exc:
        logger.error("Could not reach agent orchestrator: %s", exc)
        raise HTTPException(status_code=503, detail="Agent orchestrator unavailable")

    reply = data.get("reply", "")

    # Persist assistant reply
    try:
        save_message(session_id, "assistant", reply)
    except Exception as exc:
        logger.warning("Could not save assistant message: %s", exc)

    return ChatResponse(session_id=session_id, reply=reply)
