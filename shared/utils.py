"""Shared utilities: Bedrock client factory and logging setup."""
import logging
import boto3
from shared.config import get_settings


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def get_bedrock_runtime_client():
    """Return a boto3 Bedrock Runtime client using env credentials."""
    settings = get_settings()
    kwargs = dict(region_name=settings.aws_default_region)
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("bedrock-runtime", **kwargs)


def get_dynamodb_resource():
    """Return a boto3 DynamoDB resource, pointing to local endpoint when configured."""
    settings = get_settings()
    kwargs = dict(region_name=settings.aws_default_region)
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.dynamodb_endpoint_url:
        kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
    return boto3.resource("dynamodb", **kwargs)
