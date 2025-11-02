# News Reporter — Microsoft Agent Framework (Python)

This project uses **Microsoft Agent Framework** (Python) with:

- `WorkflowBuilder` for orchestration
- `AzureChatClient` for LLM-backed agents
- LLM-based **intent triage** → sequential or concurrent routing
- Keeps `print()` for quick local feedback + adds structured logging

## Quick start

```bash
# 1) Create venv & install deps
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux
. .venv/bin/activate && python -m pip install -r requirements.txt

# 2) Copy env and fill values
cp .env.example .env

# 3) (optional) pre-commit
pre-commit install

# 4) Run
python src/news_reporter/app.py
# or hit F5 in VS Code

python -m pip install --upgrade --pre agent-framework agent-framework-azure-ai
python -m src.news_reporter.app



```

## Environment

See `.env.example` for required variables:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY` (or use Azure CLI auth)
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT` (default chat model)
- `ROUTING_DEPLOYMENTS` (semicolon-separated for fan-out)
- `MULTI_ROUTE_ALWAYS` (`true`/`false`)

## Notes

- This build uses the **Agent Framework** (`agent-framework` + `agent-framework-azure-ai`).
- We _don't_ rely on `azure-ai-projects` here; we run locally using chat clients.
- If you want Azure Agent Service (Foundry) threads/runs, we can add that as a separate path.
  # RAG

## Additional instruction after make an agent connection in Azure

.venv\Scripts\Activate

pip install "azure-ai-projects>=1.1.0b4" "azure-ai-agents>=1.2.0b5"

az login --use-device-code
az account show --output table
python -m src.news_reporter.tools.new_01_create_agent

az login --use-device-code
az account show --output table
python -m src.news_reporter.tools.check_agent_reachability

az login --use-device-code
az account show --output table
python -m src.news_reporter.foundry_runner

az login --use-device-code
az account show --output table
python -m src.news_reporter.app

python -m src.news_reporter.tools.debug_env

#for upload and vectorized PDF
pip install fastapi uvicorn python-dotenv pydantic==2.\* \
 azure-search-documents azure-storage-blob azure-cosmos \
 PyMuPDF openai

# If you'll use Foundry Projects SDK:

pip install azure-ai-projects azure-identity

az login --use-device-code
az account show --output table
python -m src.news_reporter.api

pip install azure-search-documents azure-storage-blob requests

# (only for vector querying with your own embedding:)

# pip install azure-ai-projects azure-identity

---

# RAG File Scanner Application

A modern web application for scanning and extracting file structures from multiple sources (local directories, Google Drive, SharePoint) and exporting them as structured JSON.

## Features

- **Multiple Data Sources**: Connect to local directories, Google Drive, and SharePoint
- **Interactive UI**: Built with React, Tailwind CSS, and modern UI components
- **Authentication**: Secure OAuth flows for Google Drive and SharePoint
- **JSON Export**: Extract complete directory structures as structured JSON
- **Real-time Scanning**: Live progress updates and status indicators
- **Modern Stack**: React, Redux, Tailwind CSS, Vite, Express

## Tech Stack

### Frontend

- **React 18** with Hooks
- **Tailwind CSS** for modern, responsive styling
- **React Router** for navigation
- **React Hook Form** for form management
- **Redux Toolkit** for state management
- **Vite** for fast development
- **Lucide React** for icons

### Backend

- **Node.js** with Express
- **Googleapis** for Google Drive API integration
- **Microsoft Graph** for SharePoint integration
- Local file system scanning

## Quick Start

### 1. Installation

Install all dependencies:

```bash
npm run install-all
```

Or install manually:

```bash
npm install
cd client && npm install
```

### 2. Configuration

Copy the example environment file:

```bash
cp env.example .env
```

Edit `.env` with your credentials (see [SETUP.md](./SETUP.md) for detailed instructions).

### 3. Run Application

Start both frontend and backend servers:

```bash
npm run dev
```

The application will be available at:

- Frontend: http://localhost:3000
- Backend API: http://localhost:3001

## Project Structure

```
RAG/
├── client/                 # React frontend application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API service functions
│   │   ├── store/          # Redux store and slices
│   │   ├── utils/          # Utility functions
│   │   ├── hooks/          # Custom React hooks
│   │   ├── App.jsx         # Main app component
│   │   └── index.css       # Global styles
│   ├── public/             # Static assets
│   └── package.json
├── server/                 # Node.js backend
│   ├── routes/             # API route handlers
│   │   ├── local.js        # Local file scanning
│   │   ├── googleDrive.js  # Google Drive integration
│   │   └── sharepoint.js   # SharePoint integration
│   ├── utils/              # Server utilities
│   └── index.js            # Main server file
├── package.json
├── README.md
└── SETUP.md                # Detailed setup instructions
```

## Usage

1. **Start the application**: Run `npm run dev`
2. **Choose data source**: Select Local, Google Drive, or SharePoint
3. **Authenticate** (for cloud services): Follow OAuth flow
4. **Scan**: Enter path or trigger scan
5. **View results**: See JSON structure with file/folder information
6. **Export**: Download or copy JSON data

## API Endpoints

### Health Check

- `GET /api/health` - Server status

### Local File System

- `POST /api/local/scan` - Scan local directory

### Google Drive

- `POST /api/drive/authenticate` - Start OAuth flow
- `GET /api/drive/callback` - OAuth callback
- `POST /api/drive/scan` - Scan Google Drive

### SharePoint

- `POST /api/sharepoint/authenticate` - Start authentication
- `GET /api/sharepoint/callback` - OAuth callback
- `POST /api/sharepoint/scan` - Scan SharePoint

## Scripts

- `npm run dev` - Start both frontend and backend in development mode
- `npm run server` - Start backend server only
- `npm run client` - Start frontend development server only
- `npm run build` - Build frontend for production
- `npm start` - Start backend server (production)
- `npm run install-all` - Install both root and client dependencies

## Next Phase

This is **Phase 1: JSON Extraction**. Future phases will include:

- **Phase 2**: Neo4j integration for graph storage
- **Phase 3**: Interactive graph visualization with React Flow
- **Phase 4**: Advanced filtering and search capabilities

For detailed setup instructions, see [SETUP.md](./SETUP.md)

## License

MIT
