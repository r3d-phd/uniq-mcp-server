# UniQ-MCP Server Integration with Manus

This guide explains how to integrate the UniQ-MCP server with Manus so you can use quantum circuit synthesis directly in your conversations.

## Overview

Once integrated, you'll be able to:
- Request quantum circuit synthesis in natural language
- Verify circuit equivalence
- Run benchmarks
- Execute circuits on simulators
- Generate LaTeX tables for publications

## Integration Methods

### Method 1: Local MCP Server (Recommended)

This method runs the UniQ-MCP server on your local machine and connects it to Manus.

#### Step 1: Ensure the server is working

```bash
cd ~/Downloads/uniq-mcp-server
source .venv/bin/activate
python test_server.py
```

#### Step 2: Start the server in STDIO mode

The server runs in STDIO mode (standard input/output), which is how MCP servers communicate:

```bash
python server.py
```

#### Step 3: Configure Manus MCP Settings

To add UniQ-MCP to your Manus configuration, you need to add it to your MCP server list. The exact location depends on your Manus installation, but typically:

**For Manus Desktop App:**
1. Open Manus Settings
2. Navigate to "MCP Servers" or "Integrations"
3. Click "Add Custom MCP Server"
4. Enter the following configuration:

```json
{
  "name": "uniq-mcp",
  "command": "python",
  "args": ["/home/raad/Downloads/uniq-mcp-server/server.py"],
  "env": {
    "AIRLOCK_URL": "https://july-oem-comply-excluding.trycloudflare.com",
    "AIRLOCK_API_KEY": "rwq16ISvBoBr8Y9aSVY0pE5WjSJdZzw_bbRI7-H0w30"
  }
}
```

**For Manus CLI/API:**
Add to your `~/.manus/mcp_config.json`:

```json
{
  "mcpServers": {
    "uniq-mcp": {
      "command": "/home/raad/Downloads/uniq-mcp-server/.venv/bin/python",
      "args": ["/home/raad/Downloads/uniq-mcp-server/server.py"],
      "env": {
        "AIRLOCK_URL": "https://july-oem-comply-excluding.trycloudflare.com",
        "AIRLOCK_API_KEY": "rwq16ISvBoBr8Y9aSVY0pE5WjSJdZzw_bbRI7-H0w30"
      }
    }
  }
}
```

### Method 2: Request Manus Team to Add

If you want UniQ-MCP to be available as a built-in MCP server in Manus (like the quantum MCPs you already have), you can:

1. **Submit a request** to the Manus team to add UniQ-MCP as a supported server
2. **Provide the server code** (the `server.py` file)
3. **Specify the configuration** needed (Airlock credentials, etc.)

This would make it available across all your Manus tasks without manual setup.

## Available Tools After Integration

Once integrated, these tools will be available in your Manus conversations:

| Tool | Description | Example Usage |
|------|-------------|---------------|
| `synthesize_circuit` | Generate quantum circuits | "Create a Bell state circuit" |
| `verify_circuit` | Verify equivalence | "Check if these two circuits are equivalent" |
| `analyze_quantum_circuit` | Analyze properties | "Analyze this circuit's depth and gate count" |
| `execute_on_simulator` | Run simulation | "Execute this circuit with 1000 shots" |
| `run_benchmark` | Run benchmarks | "Run the QFT benchmark" |
| `list_benchmarks` | List available benchmarks | "Show available benchmarks" |
| `get_curriculum_problem` | Get learning problems | "Give me a difficulty 0.5 problem" |
| `generate_latex_table` | Generate LaTeX | "Create a LaTeX table of results" |
| `check_server_status` | Check status | "Check UniQ-MCP status" |

## Example Conversation After Integration

```
You: Create a quantum circuit that prepares a 3-qubit GHZ state

Manus: [calls synthesize_circuit with description="3-qubit GHZ state"]

Result:
from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)

You: Now verify it against the reference implementation

Manus: [calls verify_circuit]

Result: Circuits are equivalent (unitary comparison)

You: Execute it on the simulator with 10000 shots

Manus: [calls execute_on_simulator with shots=10000]

Result: {'000': 5023, '111': 4977}
```

## Troubleshooting

### Server not starting
- Ensure virtual environment is activated
- Check that all dependencies are installed
- Verify Airlock is running if using synthesis

### Airlock connection issues
- Check that Airlock is running: `~/start-airlock.sh`
- Verify the tunnel URL is current (Cloudflare tunnels can expire)
- Update `.env` with new credentials if needed

### MCP not recognized
- Restart Manus after adding the configuration
- Check the path to `server.py` is correct
- Ensure Python path points to the virtual environment

## Security Notes

- Airlock credentials are stored locally in `.env`
- The MCP server runs locally on your machine
- No data is sent to external servers except Airlock (your local GPU)
- AWS credentials (if configured) are only used for Braket access

## Support

For issues with:
- **UniQ-MCP Server**: Check the test output and logs
- **Manus Integration**: Contact Manus support
- **Airlock**: Check `~/.airlock/server.log`
