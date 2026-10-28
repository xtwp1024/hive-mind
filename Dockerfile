# Hive Mind Swarm System - Dockerfile

FROM python:3.11-slim

LABEL maintainer="Hive Team"
LABEL version="2.1"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

EXPOSE 5000

CMD ["python", "main.py"]
