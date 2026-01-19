from fastmcp import FastMCP
from .config import MCP_PATH

# Shared MCP instance
mcp = FastMCP("hello_mcp", root_path=MCP_PATH)
