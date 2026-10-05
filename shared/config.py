"""Shared configuration loaded from environment variables."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"

    # Bedrock models
    bedrock_llm_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    # DynamoDB
    dynamodb_table_name: str = "chat-history"
    dynamodb_endpoint_url: str = ""  # empty = real AWS

    # Service URLs
    agent_orchestrator_url: str = "http://agent-orchestrator:8002"
    mcp_server_url: str = "http://mcp-server:8003"
    ingestion_service_url: str = "http://ingestion-service:8001"

    # Vector store
    vector_store_path: str = "/app/data/faiss_index"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
