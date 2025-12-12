import os
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# 1. Create minimal MCP server
mcp = FastMCP("hello_mcp")
MCP_PATH = os.getenv("MCP_PATH", "/mcp")


def _debug_enabled() -> bool:
    """Return True when verbose logging is requested."""
    return os.getenv("MCP_DEBUG", "").lower() in ("1", "true", "yes", "on")


def debug_log(message: str) -> None:
    """Print debug messages when MCP_DEBUG is enabled."""
    if _debug_enabled():
        print(f"[debug] {message}")

# 2. Register one simple tool
@mcp.tool()
def hello(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! This MCP server is alive 🎉"

# Health/ready endpoint for platform probes
@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "MCP server ready", "mcp_path": MCP_PATH})


# Health endpoint with basic server info
@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "mcp_path": MCP_PATH})

# 3. Run using HTTP transport
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))  # Railway injects correct port
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}{MCP_PATH}")
    debug_log(f"MCP_DEBUG is enabled; listening on {MCP_PATH}")

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=MCP_PATH
    )
