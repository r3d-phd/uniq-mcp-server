#!/bin/bash
# UniQ-MCP Server Startup Script

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for required dependencies
python3 -c "import mcp" 2>/dev/null || {
    echo "Installing MCP SDK..."
    pip install "mcp[cli]" httpx
}

python3 -c "import qiskit" 2>/dev/null || {
    echo "Installing Qiskit..."
    pip install qiskit
}

# Run the server
echo "Starting UniQ-MCP Server..."
python3 server.py
