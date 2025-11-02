# RAG File Scanner - Usage Guide

## Getting Started

### First Time Setup

1. **Clone and Navigate**

   ```bash
   cd RAG
   ```

2. **Install Dependencies**

   ```bash
   npm run install-all
   ```

3. **Configure Environment**

   ```bash
   cp env.example .env
   ```

   Edit `.env` file with your credentials (optional for local scanning).

4. **Start the Application**

   ```bash
   npm run dev
   ```

5. **Open Browser**
   - Navigate to: http://localhost:3000

## Using the Application

### 1. Local Directory Scanning

**Step 1:** Click on "Local Directory" card or go to `/scanner?source=local`

**Step 2:** Enter the directory path

- Windows: `C:\Users\YourName\Documents`
- Mac/Linux: `/home/user/documents`

**Step 3:** Click "Scan Directory"

**Step 4:** Wait for scan to complete (shows progress)

**Step 5:** View results with statistics and JSON preview

### 2. Google Drive Scanning

**Step 1:** Click on "Google Drive" card or go to `/scanner?source=drive`

**Step 2:** Configure Google Drive API

- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Enable Google Drive API
- Create OAuth 2.0 credentials
- Add redirect URI: `http://localhost:3001/api/drive/callback`
- Add credentials to `.env`:
  ```
  GOOGLE_CLIENT_ID=your_client_id
  GOOGLE_CLIENT_SECRET=your_client_secret
  ```

**Step 3:** Click "Authenticate with Google Drive"

- Pop-up window will appear
- Sign in with Google account
- Grant permissions
- Window closes automatically

**Step 4:** Click "Scan Google Drive"

- Scans your entire Google Drive
- Shows progress

**Step 5:** View and export results

### 3. SharePoint Scanning

**Step 1:** Click on "SharePoint" card or go to `/scanner?source=sharepoint`

**Step 2:** Enter SharePoint credentials

- Site URL: Your SharePoint URL
- Client ID: Azure app registration ID
- Tenant ID: Azure AD tenant ID

**Step 3:** Click "Authenticate and Scan SharePoint"

- Pop-up window appears
- Sign in with Microsoft account
- Grant permissions

**Step 4:** View results

## Understanding the Results

### Statistics Cards

- **Total Files**: Number of files scanned
- **Total Folders**: Number of directories scanned
- **Source Type**: Where the data came from

### Directory Structure View

- Tree representation of your file structure
- Folders shown with folder icon
- Files shown with file icon
- File sizes displayed
- Expandable/collapsible tree view

### JSON Preview

- Complete JSON structure
- Syntax highlighted
- Scrollable for large results
- Shows all file metadata

### Available Actions

1. **Copy JSON**

   - Click "Copy JSON" button
   - Selects all JSON text
   - Copies to clipboard
   - Shows confirmation message

2. **Download JSON**
   - Click "Download JSON" button
   - Downloads as `.json` file
   - Filename includes timestamp

## JSON Output Structure

The application generates structured JSON with the following format:

```json
{
	"id": "unique-uuid",
	"type": "directory",
	"name": "folder-name",
	"fullPath": "/path/to/folder",
	"relativePath": "./folder",
	"modifiedTime": "2024-01-01T00:00:00.000Z",
	"createdAt": "2024-01-01T00:00:00.000Z",
	"source": "local",
	"children": [
		{
			"id": "another-uuid",
			"type": "file",
			"name": "file.txt",
			"fullPath": "/path/to/folder/file.txt",
			"relativePath": "./folder/file.txt",
			"size": 1024,
			"extension": ".txt",
			"modifiedTime": "2024-01-01T00:00:00.000Z",
			"createdAt": "2024-01-01T00:00:00.000Z",
			"source": "local",
			"children": []
		}
	]
}
```

### JSON Fields Explained

| Field          | Type   | Description                                      |
| -------------- | ------ | ------------------------------------------------ |
| `id`           | string | Unique identifier (UUID)                         |
| `type`         | string | "directory" or "file"                            |
| `name`         | string | File/folder name                                 |
| `fullPath`     | string | Absolute path                                    |
| `relativePath` | string | Path relative to root                            |
| `size`         | number | File size in bytes (files only)                  |
| `extension`    | string | File extension (files only)                      |
| `modifiedTime` | string | ISO timestamp of last modification               |
| `createdAt`    | string | ISO timestamp of creation                        |
| `source`       | string | "local", "google-drive", or "sharepoint"         |
| `children`     | array  | Array of child nodes (directories have children) |

## Common Tasks

### Export Data for Neo4j Import

1. Scan your desired data source
2. Click "Download JSON"
3. Save the `.json` file
4. Use it for Neo4j import in Phase 2

### Compare Multiple Sources

1. Scan first source (e.g., Local)
2. Download JSON
3. Clear results
4. Scan second source (e.g., Google Drive)
5. Download JSON
6. Compare the JSON files manually or programmatically

### Handle Large Scans

- Large directories may take time to scan
- Progress is shown during scanning
- Results are paginated in the UI
- JSON export handles large datasets efficiently

## Troubleshooting

### Local Scanning Issues

**Problem:** Path not found
**Solution:** Use absolute paths (full path from root)

**Problem:** Permission denied
**Solution:** Run with appropriate file system permissions

### Google Drive Issues

**Problem:** Authentication fails
**Solution:**

- Check redirect URI matches exactly
- Verify credentials in `.env`
- Ensure Google Drive API is enabled

**Problem:** "Access Denied" error
**Solution:** Grant requested permissions during OAuth flow

### SharePoint Issues

**Problem:** Can't connect to SharePoint
**Solution:**

- Verify site URL is correct
- Check Tenant ID and Client ID
- Ensure app is registered in Azure Portal

### General Issues

**Problem:** Port already in use
**Solution:** Change PORT in `.env` file

**Problem:** API calls fail
**Solution:** Ensure backend server is running (check console)

## Tips & Best Practices

1. **Start Small**: Test with small directories first
2. **Use Absolute Paths**: More reliable than relative paths
3. **Monitor Console**: Check browser and server consoles for errors
4. **Save Regularly**: Download important scan results
5. **Organize Exports**: Use descriptive naming for downloaded JSON files

## Keyboard Shortcuts

- `Escape`: Close modals and popups
- `Ctrl/Cmd + C`: Copy selected text
- `Ctrl/Cmd + S`: Download JSON (when focused)

## Next Steps

After extracting JSON data:

1. **Phase 2**: Import to Neo4j database
2. **Phase 3**: Visualize with React Flow
3. **Phase 4**: Add advanced search and filtering

## Support

For issues or questions:

- Check [SETUP.md](./SETUP.md) for detailed setup instructions
- Review [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) for architecture details
- See [README.md](./README.md) for quick start guide
