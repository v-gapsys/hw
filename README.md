# MCP Decisions Server

FastMCP server with semantic search over decision documents (OpenAI embeddings) and an offline index builder.

## Project structure
```
app.py                     # Entry point
hellowworld/               # MCP server package
  config.py                # Env flags, paths
  core.py                  # MCP instance
  server.py                # Routes and startup
  tools.py                 # MCP tools (search, chunk retrieval)
  index_loader.py          # Index loading/state
  openai_client.py         # Lazy OpenAI client
index_builder.py           # Offline index builder
decisions_index.npz        # Prebuilt index (embeddings + meta)
tolerated_decisions_sectioned_motives.jsonl  # Paragraph-sectioned source data
requirements.txt
railway.toml
```

## Setup
```bash
cd "/Users/vytautas/MCP 2026/hw"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run server
```bash
source venv/bin/activate
python app.py
```
Env vars:
- `MCP_PATH` (default `/mcp`)
- `INDEX_PATH` (default `decisions_index.npz`)
- `ENABLE_SEARCH_TOOLS=1` to expose search tools
- `SKIP_INDEX_LOAD` / `MCP_SKIP_INDEX` to bypass loading
- `OPENAI_API_KEY` required for search calls
- `MCP_DEBUG=1` for debug logs

Health: `/health` returns status + index info.

## MCP tools
- `search_decisions(query, top_k=5, paragraph_type=None)`: hybrid metadata + semantic search; optional paragraph filter (`reasoning`, `facts`, `law`, `operative`, `header`).
- `search_reasoning_paragraphs`, `search_facts_paragraphs`, `search_law_paragraphs`: shortcuts for filtered search.
- `get_decision_chunks(decision_id, query=None, top_k=5, paragraph_type=None)`: return chunks for a decision, optional rerank/filter.
- `hello(name)`: basic connectivity check.

## Build index (offline)
Paragraph-based (preferred):
```bash
OPENAI_API_KEY=sk-... ./venv/bin/python index_builder.py \
  --input tolerated_decisions_sectioned_motives.jsonl \
  --output decisions_index.npz \
  --use-paragraphs \
  --paragraphs-per-chunk 1
```
Text slicing (fallback):
```bash
OPENAI_API_KEY=sk-... ./venv/bin/python index_builder.py \
  --input tolerated_decisions.jsonl \
  --output decisions_index.npz \
  --chunk-size 800 \
  --chunk-overlap 100
```

## Deploy (Railway/Nixpacks)
- `railway.toml` runs `python app.py` from repo root.
- Healthcheck path should target `/health`.
- MCP clients may issue a GET probe to `MCP_PATH`; this is handled and returns JSON.
- Ensure `decisions_index.npz` is in the image or set `INDEX_PATH` accordingly.
- Set `ENABLE_SEARCH_TOOLS=1`, `OPENAI_API_KEY`, and any optional debug/skip flags.
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
