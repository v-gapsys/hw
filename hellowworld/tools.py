from typing import Any, Dict, List, Optional

import numpy as np

from .config import ENABLE_SEARCH_TOOLS
from .core import mcp
from .index_loader import EMBEDDINGS, META, ensure_index_ready
from .openai_client import embed_query


@mcp.tool()
def hello(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! This MCP server is alive 🎉"


def _ensure_search_enabled() -> None:
    if not ENABLE_SEARCH_TOOLS:
        raise RuntimeError("Search tools are disabled; set ENABLE_SEARCH_TOOLS=1 to use them")


@mcp.tool()
def search_decisions(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search over decisions. Requires ENABLE_SEARCH_TOOLS and a loaded index."""
    _ensure_search_enabled()
    ensure_index_ready()

    query_vec = embed_query(query)
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
    ensure_index_ready()

    chunks: List[Dict[str, Any]] = []
    for meta in META:
        meta_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)
        if meta_id == decision_id:
            chunks.append({"id": meta_id, "chunk": meta.get("chunk") if isinstance(meta, dict) else getattr(meta, "chunk", None)})

    if not chunks:
        raise RuntimeError(f"No chunks found for decision_id={decision_id}")

    if query:
        query_vec = embed_query(query)
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
