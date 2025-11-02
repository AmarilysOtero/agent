#!/bin/bash
# restore-working-state.sh

echo "Restoring working Neo4j integration state..."

# Check if Neo4j Desktop is running
echo "Checking Neo4j Desktop..."
if ! pgrep -f "Neo4j Desktop" > /dev/null; then
    echo "⚠️  Neo4j Desktop is not running. Please start it first."
    exit 1
fi

# Start backend
echo "Starting Neo4j backend..."
cd neo4j_backend
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Test backend health
echo "Testing backend health..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
    kill $BACKEND_PID
    exit 1
fi

# Start frontend
echo "Starting frontend..."
cd ..
npm run dev &
FRONTEND_PID=$!

echo "✅ Working state restored!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8000"
echo "Neo4j Browser: http://localhost:7474"

echo "Press Ctrl+C to stop all services"
wait
