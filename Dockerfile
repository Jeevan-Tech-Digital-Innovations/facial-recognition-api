# Facial Recognition API - Docker Image
# Python 3.11 + FastAPI + DeepFace + pgvector
#
# Build:   docker build -t face-api .
# Run:     docker run -p 8000:8000 --env-file .env face-api

FROM python:3.11-slim

ARG BUILD_ENV=staging
ARG APP_VERSION=latest

# Labels
LABEL maintainer="JeevanTech <support@jeevantech.com>"
LABEL version="${APP_VERSION}"
LABEL description="Facial Recognition API - Employee face registration and recognition"
LABEL environment="${BUILD_ENV}"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    wget \
    curl \
    tzdata \
    && cp /usr/share/zoneinfo/Asia/Kolkata /etc/localtime \
    && echo "Asia/Kolkata" > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Create directories for face images and DeepFace model cache
RUN mkdir -p /app/face_images /app/.deepface/weights /app/logs /app/tmp \
    && chown -R appuser:appgroup /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY setup_db.py .

# Create version file
RUN echo "${APP_VERSION}" > /app/version.txt

# Set writable home directory for model caches (DeepFace, TensorFlow, etc.)
ENV HOME=/app \
    DEEPFACE_HOME=/app/.deepface \
    XDG_CACHE_HOME=/app/.cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider --timeout=5 http://127.0.0.1:8000/api/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
