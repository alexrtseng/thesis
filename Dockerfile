# Lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps (if any future native libs needed, add here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (only needed dirs to reduce image size)
COPY data ./data
COPY deterministic ./deterministic
COPY Dockerfile .

# Default environment variables (override in ECS task definition)
ENV PJM_WORKERS=3 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Entrypoint will run the PJM data fetcher
ENTRYPOINT ["python", "data/helpers/get_pjm_data.py"]
