# Stage 1: Build base environment
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies if required (e.g. for building c extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies into a virtual environment or directly
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn pydantic

# Stage 2: Final runtime image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn pydantic

# Copy application source
COPY src/ /app/src/
COPY main.py /app/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI inference service using uvicorn
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]
