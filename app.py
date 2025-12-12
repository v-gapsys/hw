import os

from hellowworld.server import MCP_PATH, bootstrap_index, debug_log, mcp


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0"

    print(f"[server] Starting MCP on http://{host}:{port}{MCP_PATH}")
    debug_log(f"MCP_DEBUG is enabled; listening on {MCP_PATH}")
    bootstrap_index()

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        path=MCP_PATH,
    )
