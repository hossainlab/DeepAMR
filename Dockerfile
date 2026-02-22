FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/
COPY run_server.py .

# Create demo directory if needed
COPY demo/ demo/

EXPOSE 8000

CMD ["python", "run_server.py"]
