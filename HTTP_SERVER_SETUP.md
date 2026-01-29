# UniQ-MCP HTTP Server Setup

This guide explains how to run UniQ-MCP as an HTTP server so it can be accessed from Manus cloud.

## Why HTTP Server?

The default MCP configuration uses **STDIO transport** (local process), which only works when Manus runs on the same machine. Since Manus runs in a cloud sandbox, we need to expose UniQ-MCP via **HTTP transport** through a Cloudflare tunnel.

## Architecture

```
┌─────────────────┐     HTTPS      ┌──────────────────┐     HTTP      ┌─────────────────┐
│   Manus Cloud   │ ◄────────────► │ Cloudflare Tunnel│ ◄───────────► │  UniQ-MCP HTTP  │
│    Sandbox      │                │  (trycloudflare) │               │  (localhost)    │
└─────────────────┘                └──────────────────┘               └─────────────────┘
                                                                              │
                                                                              ▼
                                                                      ┌─────────────────┐
                                                                      │    Airlock      │
                                                                      │  (Mistral 7B)   │
                                                                      └─────────────────┘
```

## Quick Start

### Option 1: All-in-One Script

```bash
cd ~/Downloads/uniq-mcp-server
chmod +x start-uniq-http.sh
./start-uniq-http.sh
```

This script will:
1. Check/start Airlock
2. Start UniQ-MCP HTTP server on port 8001
3. Create Cloudflare tunnel
4. Display the tunnel URL

### Option 2: Manual Steps

#### Step 1: Start Airlock
```bash
~/start-airlock.sh
sleep 15
~/update-airlock-env.sh
```

#### Step 2: Start HTTP Server
```bash
cd ~/Downloads/uniq-mcp-server
source .venv/bin/activate
python http_server.py --port 8001
```

#### Step 3: Create Tunnel (in another terminal)
```bash
cloudflared tunnel --url http://localhost:8001
```

Copy the tunnel URL (e.g., `https://abc-xyz.trycloudflare.com`)

## Using with Manus

### Method 1: Direct HTTP Calls

Once the tunnel is running, tell Manus:

> "UniQ-MCP is available at https://your-tunnel-url.trycloudflare.com"

Then use it:

> "Using UniQ-MCP at https://your-tunnel-url.trycloudflare.com, synthesize a Bell state"

### Method 2: Update MCP Config (Permanent)

If you have access to Manus MCP configuration, update the `uniq-mcp` entry:

```json
{
  "uniq-mcp": {
    "transport": "http",
    "url": "https://your-tunnel-url.trycloudflare.com"
  }
}
```

**Note:** Cloudflare tunnel URLs change each time you restart. For a permanent URL, use a named Cloudflare tunnel or ngrok with a reserved domain.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/health` | GET | Health check |
| `/tools` | GET | List available tools |
| `/tools/{name}` | GET | Get tool info |
| `/invoke` | POST | Invoke a tool |
| `/mcp/tools/list` | POST | MCP-compatible tool list |
| `/mcp/tools/call` | POST | MCP-compatible tool call |

### Example: Invoke a Tool

```bash
curl -X POST "https://your-tunnel-url.trycloudflare.com/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "synthesize_circuit",
    "arguments": {
      "description": "Create a Bell state"
    }
  }'
```

### Example: List Tools

```bash
curl "https://your-tunnel-url.trycloudflare.com/tools"
```

## Available Tools

| Tool | Description |
|------|-------------|
| `check_server_status` | Check server and module status |
| `list_benchmarks` | List curriculum benchmarks |
| `get_curriculum_problem` | Get next problem based on capability |
| `record_learning_attempt` | Record attempt for adaptation |
| `get_curriculum_statistics` | Get learning statistics |
| `execute_on_simulator` | Run circuit on local simulator |
| `execute_on_hardware` | Run on quantum hardware |
| `get_available_hardware` | Check hardware status |
| `list_all_devices` | List all quantum devices |
| `synthesize_circuit` | Generate circuit from description |
| `generate_stepping_stone` | Get Teacher stepping stone |

## Troubleshooting

### "Connection refused"
- Ensure HTTP server is running: `curl http://localhost:8001/health`
- Check if port 8001 is available: `lsof -i :8001`

### "Airlock not configured"
- Set environment variables:
  ```bash
  export AIRLOCK_URL="https://your-airlock-tunnel.trycloudflare.com"
  export AIRLOCK_API_KEY="your-key"
  ```

### "Module not loaded"
- Install dependencies: `pip install fastapi uvicorn httpx chromadb`
- Check imports: `python -c "from curriculum import CurriculumManager"`

### Tunnel URL keeps changing
- Use a named Cloudflare tunnel for persistent URL
- Or use ngrok with reserved domain

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AIRLOCK_URL` | Airlock tunnel URL | Required |
| `AIRLOCK_API_KEY` | Airlock API key | Optional |
| `OPENROUTER_API_KEY` | For Teacher module | Optional |
| `UNIQ_PORT` | HTTP server port | 8001 |

## Files

| File | Purpose |
|------|---------|
| `http_server.py` | HTTP server wrapper |
| `start-uniq-http.sh` | All-in-one startup script |
| `server.py` | Original MCP server (STDIO) |
