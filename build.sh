#!/bin/bash
# build.sh

echo "🔨 Building Marketing-Finance Platform..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please create it from .env.example"
    exit 1
fi

# Build Docker images
docker-compose -f docker-compose.dev.yml build

echo "✅ Build complete!"
