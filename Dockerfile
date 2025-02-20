# Use official Python base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run site-package-check.py to setup environment
RUN python site-package-check.py

ENV PGVECTOR_CONN="postgresql+psycopg://calmadrduser:HelloWeMeetAgain#1020@172.17.0.1:5432/calmadrddb"
ENV OLLAMA_HOST="172.17.0.1"

# Start command
CMD ["uvicorn", "main_graph:fastapi_app", "--host", "0.0.0.0", "--port", "8000", "--reload"]