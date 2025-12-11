#!/usr/bin/env bash
set -euo pipefail

# Select available Python (prefer python3, fallback to python)
PY_BIN="${PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  else
    echo "No python interpreter found on PATH" >&2
    exit 1
  fi
fi

"$PY_BIN" -m pip install --upgrade pip
# Prefer root requirements; fall back to package path if needed
if [[ -f "requirements.txt" ]]; then
  REQS_PATH="requirements.txt"
elif [[ -f "hellowworld/requirements.txt" ]]; then
  REQS_PATH="hellowworld/requirements.txt"
else
  echo "No requirements file found" >&2
  exit 1
fi

"$PY_BIN" -m pip install -r "$REQS_PATH"
exec "$PY_BIN" hellowworld/server.py
