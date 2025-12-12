import os
from typing import Optional

import numpy as np

from .config import INDEX_PATH, debug_log, env_flag

EMBEDDINGS: Optional[np.ndarray] = None
META: Optional[np.ndarray] = None
INDEX_LOADED = False


def bootstrap_index() -> None:
    """Load the prebuilt index unless explicitly skipped."""
    global EMBEDDINGS, META, INDEX_LOADED

    if env_flag("SKIP_INDEX_LOAD") or env_flag("MCP_SKIP_INDEX"):
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


def ensure_index_ready() -> None:
    if not INDEX_LOADED or EMBEDDINGS is None or META is None:
        raise RuntimeError("Index is not loaded; ensure SKIP flags are off and index file exists")
