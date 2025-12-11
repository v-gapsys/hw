import os

from hellowworld.server import mcp


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}/mcp")

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        path="/mcp",
    )
