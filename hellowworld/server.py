import os
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# 1. Create minimal MCP server
mcp = FastMCP("hello_mcp")

# 2. Register one simple tool
@mcp.tool()
def hello(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! This MCP server is alive 🎉"

# Health/ready endpoint for platform probes
@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "MCP server ready", "mcp_path": "/mcp"})

# 3. Run using HTTP transport
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))  # Railway injects correct port
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}/mcp")

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        path="/mcp"
    )
