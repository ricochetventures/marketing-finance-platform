# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY configs/ ./configs/
COPY frontend/ ./frontend/

# Create necessary directories
RUN mkdir -p logs data/external models/saved data/processed

# Expose both API and Streamlit ports
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload &\n\
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Start both services
CMD ["/app/start.sh"]