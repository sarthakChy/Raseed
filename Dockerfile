# syntax=docker/dockerfile:1

# --- Base image ---
FROM python:3.10-slim

# --- Set working directory ---
WORKDIR /app

# --- Install system dependencies (for Firebase, etc.) ---
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies ---
COPY ./req.txt .
RUN pip install --no-cache-dir -r req.txt

# --- Copy server-side project folders ---
COPY server/ ./server/
COPY core/ ./core/
COPY agents/ ./agents/
COPY models/ ./models/
COPY ui/ ./ui/
COPY utils/ ./utils/
COPY config/ ./config/
COPY wallet/ ./wallet/

# --- Optional: Copy env or secrets if needed ---
# COPY .env .

# --- Expose FastAPI port ---
EXPOSE 8000

# --- Run FastAPI server ---
CMD ["sh", "-c", "uvicorn server.run:app --host 0.0.0.0 --port ${PORT:-8000}"]
