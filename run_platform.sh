#!/bin/bash

echo "🚀 Starting Marketing-Finance Platform..."

# Kill any existing processes
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Set environment to suppress warnings
export PYTHONWARNINGS="ignore"

# Start API silently
python test_api.py > api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 2

# Check if API started
if lsof -i:8000 > /dev/null 2>&1; then
    echo "✅ API running on http://localhost:8000"
else
    echo "❌ API failed to start. Check api.log for details"
    exit 1
fi

echo "✅ Starting frontend on http://localhost:8501"
echo ""
echo "============================================"
echo "Platform is ready!"
echo "Open your browser to: http://localhost:8501"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Start Streamlit with suppressed deprecation warnings
PYTHONWARNINGS="ignore" streamlit run test_frontend.py \
    --server.headless true \
    --logger.level error \
    2>/dev/null

# Cleanup
trap "kill $API_PID 2>/dev/null; echo 'Services stopped'" EXIT
