#!/usr/bin/env bash
set -euo pipefail

# Install dependencies and launch the MCP server
python3 -m pip install --upgrade pip
python3 -m pip install -r hellowworld/requirements.txt
exec python3 hellowworld/server.py
