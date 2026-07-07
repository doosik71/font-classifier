#!/usr/bin/env bash
# Share the data directory over HTTP.
# Usage: bin/share-data.sh [port] [bind-address]
# Defaults: port=9000, bind-address=0.0.0.0
# Clears VIRTUAL_ENV so an active conda/miniforge environment is ignored
# and the project's own uv .venv environment is always used.

set -u
unset VIRTUAL_ENV
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-9000}"
BIND_ADDRESS="${2:-0.0.0.0}"

cd "$PROJECT_ROOT" || exit 1

if [ ! -d "data" ]; then
    echo "data directory not found: \"$PROJECT_ROOT/data\""
    exit 1
fi

echo "Sharing \"$PROJECT_ROOT/data\" over HTTP."
echo
echo "Local:   http://127.0.0.1:$PORT/"
echo "Network: http://<this-computer-ip>:$PORT/"
echo "Bind:    $BIND_ADDRESS"
echo
echo "Press Ctrl+C to stop."
echo

uv run python -m http.server "$PORT" --bind "$BIND_ADDRESS" --directory "data"
EXIT_CODE=$?

if [ "$EXIT_CODE" != "0" ]; then
    echo
    echo "share-data exited with an error (code $EXIT_CODE)"
    read -r -p "Press Enter to continue..."
fi

exit "$EXIT_CODE"
