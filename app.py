import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from hellowworld.config import MCP_PATH
from hellowworld.core import mcp
from hellowworld.server import startup


async def mcp_get_probe(_: object) -> JSONResponse:
    # OpenAI MCP clients may probe with GET before initialize; return ok for that.
    return JSONResponse({"status": "ok", "message": "MCP endpoint probe ok", "mcp_path": MCP_PATH})


def build_app() -> Starlette:
    mcp_app = getattr(mcp, "app", None) or getattr(mcp, "asgi_app", None)
    if mcp_app is None:
        raise RuntimeError("FastMCP ASGI app not available")

    return Starlette(
        routes=[
            Route(MCP_PATH, mcp_get_probe, methods=["GET"]),
            Mount("/", app=mcp_app),
        ]
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}{MCP_PATH}")
    startup()

    app = build_app()
    uvicorn.run(app, host=host, port=port)
