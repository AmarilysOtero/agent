# RAG File Scanner - Setup Guide

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

## Installation Steps

### 1. Install Root Dependencies

```bash
npm install
```

This will install the backend dependencies and the concurrent package for running both servers.

### 2. Install Client Dependencies

```bash
cd client
npm install
cd ..
```

Or use the convenient script:

```bash
npm run install-all
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
PORT=3001

# Google Drive API (Optional)
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# SharePoint Configuration (Optional)
SHAREPOINT_TENANT_ID=your_tenant_id_here
SHAREPOINT_CLIENT_ID=your_client_id_here
```

### 4. Google Drive API Setup

To use Google Drive integration:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Web application)
5. Add authorized redirect URI: `http://localhost:3001/api/drive/callback`
6. Copy Client ID and Client Secret to `.env` file

### 5. SharePoint API Setup (Optional)

To use SharePoint integration:

1. Register app in Azure Portal
2. Get Tenant ID and Client ID
3. Add to `.env` file
4. Configure SharePoint site URL

## Running the Application

### Development Mode (Runs both servers)

```bash
npm run dev
```

This will start:

- Backend server on `http://localhost:3001`
- Frontend development server on `http://localhost:3000`

### Run Servers Separately

**Backend only:**

```bash
npm run server
```

**Frontend only:**

```bash
npm run client
```

### Production Build

Build the frontend for production:

```bash
npm run build
```

Then start the backend:

```bash
npm start
```

## Project Structure

```
rag-file-scanner/
├── client/              # React frontend
│   ├── src/
│   │   ├── components/ # UI components
│   │   ├── pages/      # Page components
│   │   ├── services/   # API services
│   │   ├── store/      # Redux store
│   │   ├── utils/      # Utility functions
│   │   └── hooks/      # Custom hooks
│   └── public/         # Static assets
├── server/              # Node.js backend
│   ├── routes/         # API routes
│   ├── utils/          # Server utilities
│   └── index.js        # Main server file
└── package.json
```

## Usage

1. **Start the application**: `npm run dev`
2. **Open browser**: Navigate to `http://localhost:3000`
3. **Select data source**: Choose Local, Google Drive, or SharePoint
4. **Authenticate** (if required for cloud services)
5. **Scan**: Trigger the scan operation
6. **View results**: See file structure in JSON format
7. **Export**: Download or copy the JSON data

## API Endpoints

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

## Troubleshooting

### Port Already in Use

Change the port in `.env`:

```env
PORT=3002
```

### Google Drive Authentication Fails

- Check redirect URI is correctly configured
- Verify Client ID and Secret are correct
- Ensure Google Drive API is enabled in Google Cloud Console

### Local Scanning Errors

- Ensure the path exists and is accessible
- Check file system permissions
- Use absolute paths for better reliability

## Next Steps

This is Phase 1: JSON extraction. Future phases will include:

- Neo4j integration for graph storage
- Interactive graph visualization with React Flow
- Advanced filtering and search capabilities

## License

MIT
