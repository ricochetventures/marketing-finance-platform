#!/bin/bash

echo "🚀 Starting Marketing-Finance Platform..."

# Clean up any existing processes
pkill -f "python test_api.py"
pkill -f "streamlit run"
sleep 1

# Start API first
echo "Starting API server..."
python test_api.py &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        echo "✅ API is running on http://localhost:8000"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ API failed to start after 10 seconds"
        kill $API_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend..."
streamlit run test_frontend.py --server.headless true --logger.level error &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "✅ Platform is running!"
echo "Frontend: http://localhost:8501"
echo "API: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo "============================================"

# Handle shutdown
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped all services'" EXIT

# Keep script running
wait
