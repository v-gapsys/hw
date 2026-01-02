# MCP Server for Decision Search

A Model Context Protocol (MCP) server that provides semantic search capabilities over decision documents using OpenAI embeddings. Built with FastMCP and deployed on Railway.

## 🚀 Quick Start

*Latest update: Deployment fix applied*

### Prerequisites
- Python 3.12+
- OpenAI API key with access to embeddings
- Git repository with decision data

### Installation

```bash
# Clone the repository
git clone https://github.com/v-gapsys/hw.git
cd hw

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ENABLE_SEARCH_TOOLS=1  # Enable semantic search tools
export MCP_DEBUG=1            # Optional: Enable debug logging
```

### Running Locally

```bash
python app.py
```

The server will start on `http://localhost:8000/mcp` using SSE transport.

## 🏗️ Architecture & Structure

### Core Components

```
├── app.py                      # Main entry point
├── hellowworld/               # MCP server package
│   ├── __init__.py
│   ├── config.py              # Environment configuration
│   ├── core.py                # FastMCP instance setup
│   ├── server.py              # Server initialization & routes
│   ├── tools.py               # MCP tools implementation
│   ├── index_loader.py        # Search index loading
│   └── openai_client.py       # OpenAI API client
├── index_builder.py           # Offline index builder
├── requirements.txt           # Python dependencies
├── railway.toml              # Railway deployment config
├── start.sh                  # Railway startup script
└── decisions_index.npz       # Pre-built search index
```

### Data Flow

1. **Index Building** (`index_builder.py`):
   - Reads `tolerated_decisions.jsonl`
   - Chunks text content (800 chars with 100 overlap)
   - Generates embeddings via OpenAI
   - Saves to `decisions_index.npz`

2. **Server Startup** (`app.py`):
   - Loads configuration from environment
   - Initializes MCP server with FastMCP
   - Registers tools and routes
   - Starts HTTP server with SSE transport

3. **Request Handling**:
   - Health checks: `GET /health`
   - MCP communication: `GET /mcp` (SSE)
   - Tool execution via MCP protocol

## 🛠️ MCP Tools

### Available Tools

#### `hello`
A simple greeting tool for testing connectivity.
```python
@mcp.tool()
def hello(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! This MCP server is alive 🎉"
```

#### `search_decisions`
Semantic search over decision documents using vector similarity.
```python
@mcp.tool()
def search_decisions(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search over decisions. Requires ENABLE_SEARCH_TOOLS and a loaded index."""
```

#### `get_decision_chunks`
Retrieve specific decision chunks with optional query-based ranking.
```python
@mcp.tool()
def get_decision_chunks(decision_id: str, query: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return chunks for a decision; optionally rank by query."""
```

## 🔧 Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MCP_PATH` | `/mcp` | MCP endpoint path |
| `INDEX_PATH` | `decisions_index.npz` | Search index file location |
| `ENABLE_SEARCH_TOOLS` | `false` | Enable/disable search functionality |
| `MCP_DEBUG` | `false` | Enable debug logging |
| `OPENAI_API_KEY` | *required* | OpenAI API key for embeddings |
| `PORT` | `8000` | Server port |

## 🚢 Deployment on Railway

### Automatic Deployment
The project is configured for Railway deployment:

```toml
[build]
builder = "NIXPACKS"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "python app.py"
healthcheckPath = "/"
healthcheckTimeout = 300
```

### Deployment Steps
1. Connect GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on git push
4. Access at `https://your-app.railway.app/mcp`

## 🎯 Challenges & Solutions

### Challenge 1: 400 Bad Request with Agent Builders
**Problem**: Railway deployment returned `400 Bad Request` when agent builders tried to connect.

**Root Cause**: Incorrect transport configuration. The server was using `transport="streamable-http"` but agent builders expect Server-Sent Events (SSE).

**Solution**:
```python
# Before (causing 400 errors)
mcp.run(transport="streamable-http", ...)

# After (working with agent builders)
mcp.run(transport="sse", ...)
```

**Impact**: Agent builders can now successfully connect and use MCP tools.

### Challenge 2: Index Loading Performance
**Problem**: Large decision datasets required efficient indexing and loading.

**Solution**: Implemented offline index building with numpy arrays for fast vector search and cosine similarity calculations.

## 📊 Performance Characteristics

- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **Chunk Size**: 800 characters with 100 character overlap
- **Search Algorithm**: Cosine similarity over normalized embeddings
- **Index Format**: NumPy compressed arrays (`.npz`)

## 🔍 Debugging

Enable debug logging:
```bash
export MCP_DEBUG=1
python app.py
```

Check server health:
```bash
curl http://localhost:8000/health
```

## 📝 Development

### Building Search Index
```bash
python index_builder.py \
  --input tolerated_decisions.jsonl \
  --output decisions_index.npz \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --model text-embedding-3-small
```

### Testing Tools
```bash
# Test basic connectivity
curl http://localhost:8000/

# Test MCP endpoint (requires MCP client)
# Use tools like mcp-client or integrate with agent builders
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) - Modern MCP server framework
- [OpenAI](https://openai.com/) - Embedding models and API
- [Railway](https://railway.app/) - Cloud deployment platform
- [Model Context Protocol](https://modelcontextprotocol.io/) - Interoperability standard
