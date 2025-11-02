#!/bin/bash

echo "🧪 Testing Neo4j Backend API..."

# Check if the API is running
echo "📡 Checking API health..."
curl -s http://localhost:8000/health | jq '.' || echo "❌ API not responding"

echo ""
echo "📊 Getting current graph stats..."
curl -s http://localhost:8000/api/graph/stats | jq '.' || echo "❌ Failed to get stats"

echo ""
echo "📁 Storing test file structure..."
curl -X POST "http://localhost:8000/api/graph/store" \
     -H "Content-Type: application/json" \
     -d @test_payload.json | jq '.' || echo "❌ Failed to store data"

echo ""
echo "📊 Getting updated graph stats..."
curl -s http://localhost:8000/api/graph/stats | jq '.' || echo "❌ Failed to get updated stats"

echo ""
echo "🔍 Searching for files..."
curl -s "http://localhost:8000/api/graph/search?name=Enterprise" | jq '.' || echo "❌ Failed to search"

echo ""
echo "✅ Test completed!"
