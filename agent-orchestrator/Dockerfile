FROM python:3.11-slim

WORKDIR /app
# Build context is the project root
COPY agent-orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY agent-orchestrator/ .

EXPOSE 8002
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8002"]
