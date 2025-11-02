# Neo4j Backend for RAG File Scanner

This Python backend receives JSON file structures from the frontend and stores them in Neo4j as a graph database.

## Features

- **FastAPI REST API** for receiving file structure JSON
- **Neo4j Graph Database** integration
- **Automatic Graph Creation** from hierarchical file structures
- **Search and Query** capabilities
- **Statistics and Analytics** for the graph
- **Data Validation** with Pydantic models

## Prerequisites

1. **Python 3.8+** installed
2. **Neo4j Database** running (local or cloud)
3. **pip** package manager

## Installation

1. **Navigate to backend directory:**

   ```bash
   cd neo4j_backend
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**

   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your Neo4j credentials:

   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

## Neo4j Setup

### Option 1: Local Neo4j Desktop

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database
3. Start the database
4. Note the connection details (URI, username, password)

### Option 2: Neo4j AuraDB (Cloud)

1. Go to [Neo4j AuraDB](https://console.neo4j.io/)
2. Create a free account
3. Create a new database
4. Copy the connection URI and credentials

### Option 3: Docker

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

## Running the Backend

1. **Start the API server:**

   ```bash
   python main.py
   ```

2. **The API will be available at:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

- `POST /api/graph/store` - Store file structure JSON in Neo4j
- `GET /api/graph/stats` - Get graph statistics
- `POST /api/graph/clear` - Clear all data from the graph
- `GET /api/graph/search` - Search files by name
- `GET /api/graph/tree/{directory_id}` - Get directory tree structure

### Utility Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check with Neo4j connection status

## Graph Schema

### Node Types

**Directory Nodes:**

- Label: `Directory`
- Properties: `id`, `name`, `fullPath`, `relativePath`, `modifiedTime`, `createdAt`, `source`, `lastUpdated`

**File Nodes:**

- Label: `File`
- Properties: `id`, `name`, `fullPath`, `relativePath`, `size`, `extension`, `modifiedTime`, `createdAt`, `source`, `lastUpdated`

### Relationships

**CONTAINS Relationship:**

- From: Directory nodes
- To: File or Directory nodes
- Represents parent-child containment

## Usage Examples

### 1. Store File Structure

```bash
curl -X POST "http://localhost:8000/api/graph/store" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "id": "root-123",
         "type": "directory",
         "name": "MyProject",
         "fullPath": "/path/to/MyProject",
         "relativePath": ".",
         "modifiedTime": "2024-01-01T00:00:00Z",
         "createdAt": "2024-01-01T00:00:00Z",
         "source": "local",
         "children": [...]
       }
     }'
```

### 2. Get Graph Statistics

```bash
curl "http://localhost:8000/api/graph/stats"
```

### 3. Search Files

```bash
curl "http://localhost:8000/api/graph/search?name=document&source=local"
```

### 4. Get Directory Tree

```bash
curl "http://localhost:8000/api/graph/tree/root-123?max_depth=3"
```

## Integration with Frontend

The frontend can send JSON data to this backend:

```javascript
// From the React frontend
const response = await fetch('http://localhost:8000/api/graph/store', {
	method: 'POST',
	headers: {
		'Content-Type': 'application/json',
	},
	body: JSON.stringify({
		data: scanResults.data,
		metadata: {
			scanTimestamp: new Date().toISOString(),
			source: scanResults.source,
		},
	}),
});
```

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing with curl

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with sample data
curl -X POST "http://localhost:8000/api/graph/store" \
     -H "Content-Type: application/json" \
     -d @sample_data.json
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**

   - Check Neo4j is running
   - Verify credentials in `.env`
   - Check firewall settings

2. **Port Already in Use**

   - Change `API_PORT` in `.env`
   - Or kill process using port 8000

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version (3.8+ required)

### Logs

Check the console output for detailed logs. The application logs:

- Neo4j connection status
- API requests and responses
- Error details

## Next Steps

After storing data in Neo4j:

1. **Query the Graph** using Cypher queries
2. **Visualize** with Neo4j Browser or other tools
3. **Integrate** with the React frontend for graph visualization
4. **Add Analytics** for file system insights

## License

MIT
