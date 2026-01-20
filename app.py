import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount

from hellowworld.config import MCP_PATH
from hellowworld.core import mcp
from hellowworld.server import startup


async def mcp_get_probe(_: object) -> JSONResponse:
    # OpenAI MCP clients may probe with GET before initialize; return ok for that.
    return JSONResponse({"status": "ok", "message": "MCP endpoint probe ok", "mcp_path": MCP_PATH})


class MCPProbeWrapper:
    def __init__(self, mcp_app: object, mcp_path: str) -> None:
        self._mcp_app = mcp_app
        self._mcp_path = mcp_path.rstrip("/") or "/"

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path in (self._mcp_path, f"{self._mcp_path}/"):
                if scope.get("method") == "GET":
                    headers = dict(scope.get("headers") or [])
                    accept = headers.get(b"accept", b"").decode("utf-8").lower()
                    if "text/event-stream" not in accept:
                        response = await mcp_get_probe(Request(scope, receive))
                        await response(scope, receive, send)
                        return
                if path.endswith("/") and self._mcp_path != "/":
                    scope = {**scope, "path": self._mcp_path}
        await self._mcp_app(scope, receive, send)


def resolve_mcp_asgi_app() -> object | None:
    candidates = [
        "app",
        "asgi_app",
        "asgi",
        "_app",
        "_asgi_app",
        "_asgi",
        "application",
    ]
    builders = [
        "http_app",
        "streamable_http_app",
        "sse_app",
        "get_app",
        "build_app",
        "create_app",
        "asgi_app",
        "app",
        "_create_app",
        "_build_app",
        "_get_app",
    ]

    for obj in (mcp, getattr(mcp, "server", None), getattr(mcp, "_server", None)):
        if obj is None:
            continue
        for name in candidates:
            app = getattr(obj, name, None)
            if app is not None:
                return app
        for name in builders:
            builder = getattr(obj, name, None)
            if callable(builder):
                try:
                    app = builder()
                except TypeError:
                    continue
                if app is not None:
                    return app

    if callable(mcp):
        return mcp

    return None


def build_app() -> Starlette | None:
    try:
        mcp_app = mcp.http_app(path=MCP_PATH)
    except Exception:
        mcp_app = resolve_mcp_asgi_app()
    if mcp_app is None:
        return None

    return Starlette(
        routes=[Mount("/", app=MCPProbeWrapper(mcp_app, MCP_PATH))],
        lifespan=getattr(mcp_app, "lifespan", None),
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}{MCP_PATH}")
    startup()

    app = build_app()
    if app is None:
        # Fall back to FastMCP's runner if we cannot resolve its ASGI app.
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            path=MCP_PATH,
        )
    else:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
