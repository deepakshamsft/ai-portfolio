# FaceAI - Production Dockerfile
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
RUN useradd -m -u 1000 faceai && \
    chown -R faceai:faceai /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/faceai/.local

# Copy application code
COPY --chown=faceai:faceai src/ ./src/
COPY --chown=faceai:faceai config.yaml .

# Create directories for models and logs
RUN mkdir -p models logs && \
    chown -R faceai:faceai models logs

# Switch to non-root user
USER faceai

# Add .local/bin to PATH
ENV PATH=/home/faceai/.local/bin:$PATH

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Run application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "30", "--access-logfile", "-", "--error-logfile", "-", "src.api:app"]
