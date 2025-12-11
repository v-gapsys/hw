from fastmcp import FastMCP
import os

# 1. Create minimal MCP server
mcp = FastMCP("hello_mcp")

# 2. Register one simple tool
@mcp.tool()
def hello(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! This MCP server is alive 🎉"

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
