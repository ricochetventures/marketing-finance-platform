#!/bin/bash
# run.sh

echo "🚀 Starting Marketing-Finance Platform..."

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
docker-compose -f docker-compose.dev.yml ps

echo "✅ Platform is running!"
echo "📊 Frontend: http://localhost:8501"
echo "🔧 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
