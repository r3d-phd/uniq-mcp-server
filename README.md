# UniQ-MCP v4 - Quantum Circuit Synthesis Server

**Simplified OpenRouter Edition** - No local GPU, tunnels, or HTTP server required!

## Quick Start

### 1. Get OpenRouter API Key
Visit [openrouter.ai/keys](https://openrouter.ai/keys) and create an API key.

### 2. Configure Environment
```bash
cd ~/Downloads/uniq-mcp-server
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Manus MCP
Add to your Manus MCP configuration:
```json
{
  "mcpServers": {
    "uniq-mcp": {
      "command": "python",
      "args": ["/path/to/uniq-mcp-server/server.py"],
      "env": {
        "OPENROUTER_API_KEY": "your_key_here"
      }
    }
  }
}
```

### 5. Use in Manus
```
Using UniQ-MCP, synthesize a Bell state circuit
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `check_server_status` | Check server status and configuration |
| `synthesize_circuit` | Generate quantum circuit from description |
| `synthesize_with_teacher` | Teacher-guided synthesis with stepping stones |
| `run_benchmark` | Run a curriculum benchmark |
| `list_benchmarks` | List all 26 benchmarks |
| `execute_on_simulator` | Run circuit on local Qiskit simulator |
| `execute_on_hardware` | Run on AWS Braket hardware (when available) |
| `get_available_hardware` | List available quantum hardware |
| `get_curriculum_problem` | Get next adaptive curriculum problem |
| `record_learning_attempt` | Record learning attempt for adaptation |
| `get_curriculum_statistics` | Get curriculum learning statistics |
| `list_available_models` | Show available OpenRouter models |
| `generate_latex_table` | Generate LaTeX results table |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UniQ-MCP v4                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Manus ◄──MCP STDIO──► UniQ-MCP Server                     │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                       │
│                    │   OpenRouter    │                       │
│                    │  ┌───────────┐  │                       │
│                    │  │ DeepSeek  │  │ ◄── Student (fast)   │
│                    │  │   Chat    │  │                       │
│                    │  └───────────┘  │                       │
│                    │  ┌───────────┐  │                       │
│                    │  │ DeepSeek  │  │ ◄── Teacher (reason) │
│                    │  │    R1     │  │                       │
│                    │  └───────────┘  │                       │
│                    └─────────────────┘                       │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                       │
│                    │  Qiskit/Braket  │ ◄── Execution        │
│                    └─────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Options

You can specify any OpenRouter model for synthesis:

| Model | Use Case | Cost |
|-------|----------|------|
| `deepseek/deepseek-chat` | Fast synthesis (default) | Low |
| `deepseek/deepseek-reasoner` | Teacher reasoning | Medium |
| `anthropic/claude-3.5-sonnet` | High quality code | Medium |
| `openai/gpt-4-turbo` | Maximum reliability | High |
| `meta-llama/llama-3.1-70b-instruct` | Good balance | Low |

Example:
```
Using UniQ-MCP, synthesize a GHZ state using claude-3.5-sonnet
```

---

## Benchmarks

26 problems across 5 categories:

| Category | Problems |
|----------|----------|
| **Basic Gates** | H, X, Y, Z, T gates |
| **Two-Qubit** | CNOT, CZ, SWAP, iSWAP |
| **Entanglement** | Bell states, GHZ (3-5 qubits), W states |
| **Algorithms** | QFT (2-4 qubits), Grover (2-3 qubits), Deutsch-Jozsa |
| **Advanced** | Toffoli, Fredkin, Teleportation, VQE ansatz |

---

## What Changed from v3?

| v3 (Airlock) | v4 (OpenRouter) |
|--------------|-----------------|
| Local RTX 2070 required | No local GPU needed |
| Cloudflare tunnel required | No tunnels needed |
| HTTP server wrapper | Direct MCP STDIO |
| Mistral 7B only | Multiple models available |
| Complex startup scripts | Simple API key setup |
| ~30s latency | ~5-15s latency |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | Your OpenRouter API key |
| `UNIQ_DEFAULT_MODEL` | No | Default model (deepseek/deepseek-chat) |
| `UNIQ_TEACHER_MODEL` | No | Teacher model (deepseek/deepseek-reasoner) |
| `AWS_ACCESS_KEY_ID` | No | For AWS Braket hardware |
| `AWS_SECRET_ACCESS_KEY` | No | For AWS Braket hardware |

---

## Usage Examples

### Basic Synthesis
```
Using UniQ-MCP, create a Bell state circuit
```

### With Model Selection
```
Using UniQ-MCP, synthesize a 3-qubit QFT using gpt-4-turbo
```

### Run Benchmark
```
Using UniQ-MCP, run the ghz_3 benchmark
```

### Execute on Simulator
```
Using UniQ-MCP, execute this Bell state on the simulator with 1000 shots
```

### Generate LaTeX Table
```
Using UniQ-MCP, run all entanglement benchmarks and generate a LaTeX table
```

---

## Troubleshooting

### "OPENROUTER_API_KEY not set"
Add your API key to `.env` or pass it via MCP config.

### Synthesis fails
- Check your OpenRouter balance
- Try a different model
- Simplify the circuit description

### Module import errors
```bash
pip install mcp qiskit httpx chromadb
```

---

## License

MIT License - For academic research use.

## Contributing

Contributions welcome! Submit issues and PRs on GitHub.
