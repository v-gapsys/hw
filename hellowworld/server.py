from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import MCP_PATH, debug_log
from .core import mcp
from . import index_loader
from . import tools  # noqa: F401 - registers tools


@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "MCP server ready", "mcp_path": MCP_PATH})


@mcp.custom_route("/mcp-probe", methods=["GET"])
async def mcp_probe(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "MCP endpoint reachable", "mcp_path": MCP_PATH})


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "mcp_path": MCP_PATH,
            "index_loaded": index_loader.INDEX_LOADED,
            "chunks": len(index_loader.META) if index_loader.META is not None else None,
            "embeddings_shape": index_loader.EMBEDDINGS.shape if index_loader.EMBEDDINGS is not None else None,
        }
    )


def startup() -> None:
    debug_log(f"MCP_DEBUG is enabled; listening on {MCP_PATH}")
    index_loader.bootstrap_index()
