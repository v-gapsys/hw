import os
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# 1. Create minimal MCP server
mcp = FastMCP("hello_mcp")
MCP_PATH = os.getenv("MCP_PATH", "/mcp")
INDEX_PATH = os.getenv("INDEX_PATH", "decisions_index.npz")
ENABLE_SEARCH_TOOLS = _env_flag("ENABLE_SEARCH_TOOLS")

EMBEDDINGS = None
META = None
INDEX_LOADED = False
_OPENAI_CLIENT = None


def _env_flag(name: str) -> bool:
    """Return True when an env var is set to a truthy value."""
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def debug_log(message: str) -> None:
    """Print debug messages when MCP_DEBUG is enabled."""
    if _env_flag("MCP_DEBUG"):
        print(f"[debug] {message}")


def bootstrap_index() -> None:
    """Load the prebuilt index unless explicitly skipped."""
    global EMBEDDINGS, META, INDEX_LOADED

    if _env_flag("SKIP_INDEX_LOAD") or _env_flag("MCP_SKIP_INDEX"):
        debug_log("Index load skipped via env flag")
        return

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Index file not found at {INDEX_PATH}")

    data = np.load(INDEX_PATH, allow_pickle=True)
    if "embeddings" not in data or "meta" not in data:
        raise RuntimeError("Index file is missing required keys: embeddings, meta")

    EMBEDDINGS = data["embeddings"]
    META = data["meta"]
    INDEX_LOADED = True

    chunk_count = len(META) if META is not None else 0
    debug_log(
        f"Loaded index from {INDEX_PATH}: chunks={chunk_count}, embeddings_shape={getattr(EMBEDDINGS, 'shape', None)}"
    )


def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client, requiring OPENAI_API_KEY to be set."""
    global _OPENAI_CLIENT

    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for search tools")

    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    debug_log("Initialized OpenAI client")
    return _OPENAI_CLIENT


def _ensure_index_ready() -> None:
    if not INDEX_LOADED or EMBEDDINGS is None or META is None:
        raise RuntimeError("Index is not loaded; ensure SKIP flags are off and index file exists")


def _ensure_search_enabled() -> None:
    if not ENABLE_SEARCH_TOOLS:
        raise RuntimeError("Search tools are disabled; set ENABLE_SEARCH_TOOLS=1 to use them")


def _embed_query(query: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    return np.array(resp.data[0].embedding, dtype=float)

# 2. Register one simple tool
@mcp.tool()
def hello(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! This MCP server is alive 🎉"


@mcp.tool()
def search_decisions(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search over decisions. Requires ENABLE_SEARCH_TOOLS and a loaded index."""
    _ensure_search_enabled()
    _ensure_index_ready()

    query_vec = _embed_query(query)
    if EMBEDDINGS.shape[1] != query_vec.shape[0]:
        raise RuntimeError(
            f"Embedding dimension mismatch: index has {EMBEDDINGS.shape[1]}, query has {query_vec.shape[0]}"
        )

    scores = EMBEDDINGS @ query_vec  # cosine-ish assuming normalized vectors
    top_k = max(1, min(top_k, len(scores)))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        meta = META[idx]
        results.append(
            {
                "id": meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None),
                "chunk": meta.get("chunk") if isinstance(meta, dict) else getattr(meta, "chunk", None),
                "score": float(scores[idx]),
            }
        )
    return results


@mcp.tool()
def get_decision_chunks(decision_id: str, query: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return chunks for a decision; optionally rank by query."""
    _ensure_search_enabled()
    _ensure_index_ready()

    chunks: List[Dict[str, Any]] = []
    for meta in META:
        meta_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)
        if meta_id == decision_id:
            chunks.append({"id": meta_id, "chunk": meta.get("chunk") if isinstance(meta, dict) else getattr(meta, "chunk", None)})

    if not chunks:
        raise RuntimeError(f"No chunks found for decision_id={decision_id}")

    if query:
        query_vec = _embed_query(query)
        # Simple scoring: reuse embeddings positions where ids match; if counts mismatch, leave unsorted.
        scores: List[float] = []
        for meta in META:
            meta_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)
            if meta_id == decision_id:
                idx = np.where(META == meta)[0]
                if idx.size > 0 and EMBEDDINGS.shape[0] > idx[0]:
                    scores.append(float(EMBEDDINGS[idx[0]] @ query_vec))
        if scores and len(scores) == len(chunks):
            chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)][:max(1, min(top_k, len(chunks)))]

    return chunks

# Health/ready endpoint for platform probes
@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "MCP server ready", "mcp_path": MCP_PATH})


# Health endpoint with basic server info
@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "mcp_path": MCP_PATH,
            "index_loaded": INDEX_LOADED,
            "chunks": len(META) if META is not None else None,
            "embeddings_shape": EMBEDDINGS.shape if EMBEDDINGS is not None else None,
        }
    )

# 3. Run using HTTP transport
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))  # Railway injects correct port
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}{MCP_PATH}")
    debug_log(f"MCP_DEBUG is enabled; listening on {MCP_PATH}")
    bootstrap_index()

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=MCP_PATH
    )
