@echo off
echo 🧪 Testing Neo4j Backend API...

echo 📡 Checking API health...
curl -s http://localhost:8000/health

echo.
echo 📊 Getting current graph stats...
curl -s http://localhost:8000/api/graph/stats

echo.
echo 📁 Storing test file structure...
curl -X POST "http://localhost:8000/api/graph/store" ^
     -H "Content-Type: application/json" ^
     -d @test_payload.json

echo.
echo 📊 Getting updated graph stats...
curl -s http://localhost:8000/api/graph/stats

echo.
echo 🔍 Searching for files...
curl -s "http://localhost:8000/api/graph/search?name=Enterprise"

echo.
echo ✅ Test completed!
pause
