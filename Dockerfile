FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will override with PORT env var)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server - it reads PORT from environment
CMD ["python", "http_server.py"]
