# syntax=docker/dockerfile:1

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    binutils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY pyproject.toml ./
COPY app.py config.py ingestion.py rag.py ./
RUN pip install .

# Fail loudly if console scripts are missing
RUN test -x /opt/venv/bin/bbc-ingest || (echo "ERROR: bbc-ingest missing" >&2; exit 1) && \
    test -x /opt/venv/bin/bbc-rag    || (echo "ERROR: bbc-rag missing"    >&2; exit 1)

# Prune
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true


# ==============================================================================
# Stage 2: Runtime
# CLI only — runs as root so the named volume is writable without chown tricks
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

RUN mkdir -p /app/chroma_db

COPY config.py app.py ingestion.py rag.py pyproject.toml bbc_news.csv ./

# No USER directive — container runs as root (uid 0).
# Acceptable for one-shot batch jobs (bbc-ingest / bbc-rag exit after running).
CMD ["bbc-rag"]