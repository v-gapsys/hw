import os
from typing import Any

import numpy as np
from openai import OpenAI

from .config import debug_log

_OPENAI_CLIENT: OpenAI | None = None


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


def embed_query(query: str) -> np.ndarray:
    client = get_openai_client()
    resp: Any = client.embeddings.create(model="text-embedding-3-small", input=query)
    return np.array(resp.data[0].embedding, dtype=float)
