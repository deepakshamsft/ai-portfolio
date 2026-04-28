# UnifiedAI - Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 unifiedai && \
    chown -R unifiedai:unifiedai /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/unifiedai/.local

# Copy application code
COPY --chown=unifiedai:unifiedai src/ ./src/
COPY --chown=unifiedai:unifiedai config.yaml .

# Create directories for models and logs
RUN mkdir -p models logs && \
    chown -R unifiedai:unifiedai models logs

# Switch to non-root user
USER unifiedai

# Add .local/bin to PATH
ENV PATH=/home/unifiedai/.local/bin:$PATH

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Run API server
CMD ["python", "-m", "src.api"]
