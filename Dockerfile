# =========================
# 1. Base Image
# =========================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Avoid writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================
# 2. System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 3. Copy Files
# =========================
# Copy dependency file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container
COPY . .

# =========================
# 4. Environment Variables
# =========================
# MLflow tracking & project paths
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV PYTHONPATH=/app

# Streamlit config (so it runs on Docker correctly)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# =========================
# 5. Expose Ports
# =========================
EXPOSE 8501 5000

# =========================
# 6. Default Command
# =========================
# Choose the service mode via build arg or override at runtime
# Example: docker run -e MODE=train ekopower:latest
ARG MODE=app
ENV MODE=${MODE}

# =========================
# 7. Entrypoint
# =========================
# Train pipeline or run Streamlit UI depending on mode
CMD if [ "$MODE" = "train" ]; then \
        echo "🚀 Running training pipeline..." && \
        python scripts/run_pipeline.py ; \
    else \
        echo "🚀 Launching Streamlit App..." && \
        streamlit run app/app_streamlit.py --server.port=8501 --server.address=0.0.0.0 ; \
    fi
