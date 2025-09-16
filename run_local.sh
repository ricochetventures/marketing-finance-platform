#!/bin/bash

echo "🚀 Starting Marketing-Finance Platform (Local Mode)..."

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# Start Redis (if installed locally with Homebrew)
if command -v redis-server &> /dev/null; then
    redis-server --daemonize yes
    echo "✅ Redis started"
fi

# Start the API server in background
echo "Starting API server..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "✅ API running on http://localhost:8000"

# Start the Streamlit frontend
echo "Starting frontend..."
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!
echo "✅ Frontend running on http://localhost:8501"

echo "
=================================
Platform is running!
Frontend: http://localhost:8501
API: http://localhost:8000
API Docs: http://localhost:8000/docs

Press Ctrl+C to stop all services
=================================
"

# Wait and handle shutdown
trap "kill $API_PID $FRONTEND_PID; exit" INT
wait
