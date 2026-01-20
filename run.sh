#!/bin/bash
# Run both MCP servers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for mock mode
if [ "$1" == "--mock" ]; then
    export FRANKA_MOCK=1
    export CAMERA_MOCK=1
    echo "Running in mock mode (no hardware)"
fi

echo "Starting Franka MCP server..."
python -m franka_mcp.server &
FRANKA_PID=$!

echo "Starting Camera MCP server..."
python -m camera_mcp.server &
CAMERA_PID=$!

echo "Servers started (Franka PID: $FRANKA_PID, Camera PID: $CAMERA_PID)"
echo "Press Ctrl+C to stop"

# Handle shutdown
cleanup() {
    echo "Shutting down servers..."
    kill $FRANKA_PID 2>/dev/null
    kill $CAMERA_PID 2>/dev/null
    wait
    echo "Done"
}

trap cleanup SIGINT SIGTERM

# Wait for both
wait
