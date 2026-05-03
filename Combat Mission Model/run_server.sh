#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
echo "URL: http://127.0.0.1:${PORT}/"
echo "If bind fails (address already in use), either:"
echo "  PORT=8001 $0"
echo "  or stop the old server: lsof -iTCP:${PORT} -sTCP:LISTEN"
exec python3 -m uvicorn main:app --host 127.0.0.1 --port "${PORT}"
