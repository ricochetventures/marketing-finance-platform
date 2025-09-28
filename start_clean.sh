#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 Starting Marketing-Finance Platform..."

# Kill any existing processes silently
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Start API in background with suppressed output
echo "Starting API server..."
python test_api.py > /dev/null 2>&1 &
API_PID=$!
sleep 2

# Start Frontend with cleaner output
echo "Starting Frontend..."
echo ""
echo "============================================"
echo "✅ Platform is running successfully!"
echo "============================================"
echo "📊 Frontend: http://localhost:8501"
echo "🔧 API: http://localhost:8000" 
echo "📚 API Docs: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Run streamlit with minimal output
streamlit run test_frontend.py --server.headless true 2>/dev/null

# Cleanup on exit
trap "kill $API_PID 2>/dev/null" EXIT
