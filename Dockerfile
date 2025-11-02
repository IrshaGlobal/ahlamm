# Dockerfile for Ahlamm Application
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY model/ ./model/
COPY data/ ./data/
COPY server.py .

# Expose port
EXPOSE 8000

# Health check (no external deps)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Run application
CMD ["python", "server.py"]
