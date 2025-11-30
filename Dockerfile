# LiminalLM Production Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Copy application code
COPY liminallm/ ./liminallm/

# Production stage
FROM python:3.11-slim AS production

# Security: Run as non-root user
RUN groupadd -r liminallm && useradd -r -g liminallm liminallm

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/liminallm ./liminallm
COPY frontend/ ./frontend/
COPY sql/ ./sql/
COPY scripts/ ./scripts/

# Create data directories with proper permissions
RUN mkdir -p /srv/liminallm && \
    chown -R liminallm:liminallm /srv/liminallm /app

# Environment variables with secure defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SHARED_FS_ROOT=/srv/liminallm \
    HOST=0.0.0.0 \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Switch to non-root user
USER liminallm

EXPOSE 8000

# Use uvicorn with production settings
CMD ["python", "-m", "uvicorn", "liminallm.api.routes:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
