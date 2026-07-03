#!/usr/bin/env bash
# Launcher for scripts/font-dataset-browser.py
# Clears VIRTUAL_ENV so an active conda/miniforge environment is ignored
# and the project's own uv .venv environment is always used.

set -u
unset VIRTUAL_ENV
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT" || exit 1
uv run python scripts/font-dataset-browser.py "$@"
EXIT_CODE=$?

if [ "$EXIT_CODE" != "0" ]; then
    echo
    echo "font-dataset-browser exited with an error (code $EXIT_CODE)"
    read -r -p "Press Enter to continue..."
fi

exit "$EXIT_CODE"
