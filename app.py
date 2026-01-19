import os

from hellowworld.config import MCP_PATH
from hellowworld.core import mcp
from hellowworld.server import startup


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}{MCP_PATH}")
    startup()

    mcp.run(
        transport="http",
        host=host,
        port=port,
        path=MCP_PATH,
    )
