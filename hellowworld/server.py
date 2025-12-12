from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import MCP_PATH, debug_log
from .core import mcp
from .index_loader import EMBEDDINGS, INDEX_LOADED, META, bootstrap_index
from . import tools  # noqa: F401 - registers tools


@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "MCP server ready", "mcp_path": MCP_PATH})


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


def startup() -> None:
    debug_log(f"MCP_DEBUG is enabled; listening on {MCP_PATH}")
    bootstrap_index()
