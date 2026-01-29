# UniQ-MCP Server

A Model Context Protocol (MCP) server that exposes quantum circuit synthesis capabilities for seamless integration with AI assistants like Manus.

## Overview

UniQ-MCP Server converts the UniQ-MCP v13 quantum circuit synthesis system into an MCP server that can be called directly from AI conversations. This enables:

- **Natural language circuit synthesis**: "Generate a Bell state circuit"
- **Automatic verification**: Circuits are validated before returning
- **Benchmark execution**: Run standardised quantum computing benchmarks
- **Hardware execution**: Execute circuits on simulators or AWS Braket
- **Publication-ready exports**: Generate LaTeX tables for papers

## Features

| Tool | Description |
|------|-------------|
| `synthesize_circuit` | Generate quantum circuits from natural language |
| `verify_circuit` | Verify circuit equivalence |
| `analyze_quantum_circuit` | Analyze circuit properties (depth, gates, etc.) |
| `run_benchmark` | Run specific benchmark problems |
| `list_benchmarks` | List available benchmarks |
| `execute_on_simulator` | Execute circuits on local simulator |
| `check_server_status` | Check status of all components |
| `get_curriculum_problem` | Get problems from adaptive curriculum |
| `generate_latex_table` | Generate publication-ready LaTeX tables |

## Installation

### Prerequisites

- Python 3.10 or higher
- Airlock server running on your local GPU (optional but recommended)

### Setup

1. **Clone or extract the server:**
   ```bash
   cd uniq-mcp-server
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Airlock credentials
   ```

4. **Test the server:**
   ```bash
   python3 server.py
   ```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AIRLOCK_URL` | Cloudflare tunnel URL for Airlock | Yes (for synthesis) |
| `AIRLOCK_API_KEY` | Airlock API key | Yes (for synthesis) |
| `AWS_ACCESS_KEY_ID` | AWS access key | No (for Braket) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | No (for Braket) |
| `GEMINI_API_KEY` | Google Gemini API key | No (for Teacher) |
| `OPENROUTER_API_KEY` | OpenRouter API key | No (for Teacher) |

### MCP Client Configuration

To use with Claude Desktop or other MCP clients, add to your MCP configuration:

```json
{
  "mcpServers": {
    "uniq-mcp": {
      "command": "python3",
      "args": ["/path/to/uniq-mcp-server/server.py"],
      "env": {
        "AIRLOCK_URL": "https://your-tunnel.trycloudflare.com",
        "AIRLOCK_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Usage Examples

### From Manus Conversations

Once configured, you can use UniQ-MCP directly in conversations:

**Circuit Synthesis:**
```
User: "Generate a 3-qubit GHZ state circuit"
Manus: [calls synthesize_circuit] → Returns verified Qiskit code
```

**Run Benchmarks:**
```
User: "Run the Bell state benchmark"
Manus: [calls run_benchmark("bell_state")] → Returns results
```

**Execute on Simulator:**
```
User: "Execute this circuit and show me the measurement results"
Manus: [calls execute_on_simulator] → Returns counts
```

**Generate LaTeX:**
```
User: "Generate a LaTeX table of the benchmark results"
Manus: [calls generate_latex_table] → Returns publication-ready LaTeX
```

### Programmatic Usage

```python
import asyncio
from server import synthesize_circuit, run_benchmark

async def main():
    # Synthesize a circuit
    result = await synthesize_circuit("Create a Bell state")
    print(result["code"])
    
    # Run a benchmark
    benchmark = await run_benchmark("bell_state")
    print(f"Success: {benchmark['success']}, Time: {benchmark['time_taken']}s")

asyncio.run(main())
```

## Available Benchmarks

| ID | Category | Difficulty | Description |
|----|----------|------------|-------------|
| `bell_state` | entanglement | 1 | Bell state |Φ+⟩ |
| `ghz_3` | entanglement | 2 | 3-qubit GHZ state |
| `ghz_5` | entanglement | 3 | 5-qubit GHZ state |
| `x_gate` | basic_gates | 1 | X gate application |
| `hadamard` | basic_gates | 1 | Hadamard gate |
| `cnot` | basic_gates | 1 | CNOT gate |
| `swap` | basic_gates | 2 | SWAP using CNOTs |
| `toffoli` | multi_qubit | 3 | Toffoli gate |
| `qft_2` | algorithms | 3 | 2-qubit QFT |
| `qft_3` | algorithms | 4 | 3-qubit QFT |
| `grover_2` | algorithms | 4 | 2-qubit Grover's |
| `teleportation` | protocols | 3 | Quantum teleportation |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UniQ-MCP Server                          │
├─────────────────────────────────────────────────────────────┤
│  MCP Interface Layer (FastMCP)                              │
│  └── @mcp.tool() decorated functions                        │
├─────────────────────────────────────────────────────────────┤
│  Core Services                                              │
│  ├── Circuit Synthesis (Airlock + LLM)                      │
│  ├── Verification (Qiskit Operator)                         │
│  ├── Benchmarks (12 standard problems)                      │
│  └── Simulation (Qiskit Statevector)                        │
├─────────────────────────────────────────────────────────────┤
│  Backends                                                   │
│  ├── Airlock (Local GPU - Mistral 7B)                       │
│  ├── Qiskit (Circuit manipulation)                          │
│  └── AWS Braket (Cloud/Hardware - optional)                 │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Server won't start

1. Check Python version: `python3 --version` (need 3.10+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check for import errors: `python3 -c "import mcp; import qiskit"`

### Airlock connection fails

1. Verify Airlock is running on your local machine
2. Check the tunnel URL is correct and accessible
3. Verify API key is correct

### Circuit synthesis fails

1. Check Airlock health: Use `check_server_status` tool
2. Try simpler circuits first (e.g., X gate)
3. Increase `max_attempts` parameter

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
