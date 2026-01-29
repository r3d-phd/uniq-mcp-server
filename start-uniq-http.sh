#!/bin/bash
# UniQ-MCP HTTP Server Startup Script
# This script starts both Airlock and UniQ-MCP HTTP server with Cloudflare tunnel

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           UniQ-MCP HTTP Server Startup                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Configuration
UNIQ_DIR="${UNIQ_DIR:-$HOME/Downloads/uniq-mcp-server}"
UNIQ_PORT="${UNIQ_PORT:-8001}"
AIRLOCK_PORT="${AIRLOCK_PORT:-8000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "$UNIQ_DIR/http_server.py" ]; then
    echo -e "${RED}Error: http_server.py not found in $UNIQ_DIR${NC}"
    echo "Please set UNIQ_DIR to your UniQ-MCP installation directory"
    exit 1
fi

cd "$UNIQ_DIR"

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo -e "${YELLOW}Installing missing dependencies...${NC}"
    pip install fastapi uvicorn httpx
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Step 1: Start Airlock (if not already running)
echo -e "\n${GREEN}Step 1: Checking Airlock...${NC}"
if curl -s "http://localhost:$AIRLOCK_PORT/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Airlock already running on port $AIRLOCK_PORT${NC}"
else
    echo -e "${YELLOW}Starting Airlock...${NC}"
    if [ -f "$HOME/start-airlock.sh" ]; then
        $HOME/start-airlock.sh &
        sleep 10
    else
        echo -e "${RED}Warning: Airlock startup script not found${NC}"
        echo "Please start Airlock manually or set AIRLOCK_URL environment variable"
    fi
fi

# Step 2: Start UniQ-MCP HTTP Server
echo -e "\n${GREEN}Step 2: Starting UniQ-MCP HTTP Server on port $UNIQ_PORT...${NC}"
python3 http_server.py --port $UNIQ_PORT &
HTTP_PID=$!
sleep 3

# Check if server started
if curl -s "http://localhost:$UNIQ_PORT/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ UniQ-MCP HTTP Server running${NC}"
else
    echo -e "${RED}✗ Failed to start HTTP server${NC}"
    exit 1
fi

# Step 3: Start Cloudflare Tunnel
echo -e "\n${GREEN}Step 3: Starting Cloudflare Tunnel...${NC}"
echo -e "${YELLOW}The tunnel URL will appear below. Copy it to update your Manus MCP config.${NC}"
echo ""

cloudflared tunnel --url http://localhost:$UNIQ_PORT 2>&1 | while read line; do
    if echo "$line" | grep -q "https://"; then
        TUNNEL_URL=$(echo "$line" | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com')
        if [ -n "$TUNNEL_URL" ]; then
            echo ""
            echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}  TUNNEL URL: $TUNNEL_URL${NC}"
            echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
            echo ""
            echo "To use in Manus, update your MCP config with:"
            echo "  URL: $TUNNEL_URL"
            echo "  Transport: HTTP"
            echo ""
            echo "Or tell Manus:"
            echo "  'UniQ-MCP is available at $TUNNEL_URL'"
            echo ""
        fi
    fi
    echo "$line"
done

# Wait for processes
wait
