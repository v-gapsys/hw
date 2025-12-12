import os


def env_flag(name: str) -> bool:
    """Return True when an env var is set to a truthy value."""
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


MCP_PATH = os.getenv("MCP_PATH", "/mcp")
INDEX_PATH = os.getenv("INDEX_PATH", "decisions_index.npz")
ENABLE_SEARCH_TOOLS = env_flag("ENABLE_SEARCH_TOOLS")


def debug_log(message: str) -> None:
    """Print debug messages when MCP_DEBUG is enabled."""
    if env_flag("MCP_DEBUG"):
        print(f"[debug] {message}")
