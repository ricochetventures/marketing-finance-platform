#!/bin/bash
# stop.sh

echo "🛑 Stopping Marketing-Finance Platform..."

docker-compose -f docker-compose.dev.yml down

echo "✅ Platform stopped!"

