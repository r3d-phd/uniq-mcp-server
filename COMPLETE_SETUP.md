# UniQ-MCP Complete Setup Guide

This document provides complete instructions for GitHub setup, Manus integration, and research usage.

## Table of Contents

1. [GitHub Repository Setup](#1-github-repository-setup)
2. [Manus MCP Integration](#2-manus-mcp-integration)
3. [Daily Workflow](#3-daily-workflow)
4. [Research Usage Examples](#4-research-usage-examples)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. GitHub Repository Setup

### Option A: Automated Setup (Recommended)

```bash
# Make the script executable
chmod +x ~/Downloads/uniq-mcp-server/setup_github.sh

# Run the setup script
~/Downloads/uniq-mcp-server/setup_github.sh
```

### Option B: Manual Setup

```bash
cd ~/Downloads/uniq-mcp-server

# Initialize git
git init
git branch -m main

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.venv/
.env
chroma_data/
*.log
.DS_Store
EOF

# Add and commit
git add -A
git commit -m "Initial commit: UniQ-MCP v13"

# Create private GitHub repo and push
gh repo create uniq-mcp-server --private --source=. --push
```

### Keeping GitHub Updated

After making changes:

```bash
cd ~/Downloads/uniq-mcp-server
git add -A
git commit -m "Description of changes"
git push
```

---

## 2. Manus MCP Integration

UniQ-MCP is already configured as an MCP server in Manus. To ensure it works:

### Step 1: Verify Configuration

The MCP configuration should be at `~/.manus/mcp_config.json` or similar:

```json
{
  "mcpServers": {
    "uniq-mcp": {
      "command": "/home/raad/Downloads/uniq-mcp-server/.venv/bin/python",
      "args": ["/home/raad/Downloads/uniq-mcp-server/server.py"],
      "env": {
        "AIRLOCK_URL": "${AIRLOCK_URL}",
        "AIRLOCK_API_KEY": "${AIRLOCK_API_KEY}"
      }
    }
  }
}
```

### Step 2: Update Environment Before Each Session

Before using UniQ-MCP in Manus, always update the Airlock configuration:

```bash
~/update-airlock-env.sh
```

### Step 3: Using UniQ-MCP in Manus

Once configured, you can use these tools in Manus conversations:

| Tool | Description | Example Prompt |
|------|-------------|----------------|
| `synthesize_circuit` | Generate circuits | "Create a Bell state circuit" |
| `verify_circuit` | Verify equivalence | "Verify these two circuits are equivalent" |
| `analyze_quantum_circuit` | Analyze properties | "Analyze this circuit's depth" |
| `execute_on_simulator` | Run simulation | "Execute with 1000 shots" |
| `run_benchmark` | Run benchmarks | "Run the QFT benchmark" |
| `list_benchmarks` | List benchmarks | "Show available benchmarks" |
| `get_curriculum_problem` | Get problems | "Give me a difficulty 0.5 problem" |
| `generate_latex_table` | Generate LaTeX | "Create a LaTeX table" |
| `check_server_status` | Check status | "Check UniQ-MCP status" |

### Example Manus Conversation

```
You: Using UniQ-MCP, create a quantum circuit that prepares a 3-qubit GHZ state

Manus: [Calls synthesize_circuit tool]

Result: 
from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)

You: Now run it on the simulator with 10000 shots

Manus: [Calls execute_on_simulator tool]

Result: {'000': 5012, '111': 4988}
```

---

## 3. Daily Workflow

### Morning Setup

```bash
# 1. Start Airlock (if not running)
~/start-airlock.sh

# 2. Wait for tunnel to establish (10-15 seconds)
sleep 15

# 3. Update UniQ-MCP configuration
~/update-airlock-env.sh

# 4. Verify everything works
cd ~/Downloads/uniq-mcp-server
source .venv/bin/activate
python test_server.py
```

### During Research

```bash
# If Airlock restarts or tunnel changes
~/update-airlock-env.sh

# To run the server directly
cd ~/Downloads/uniq-mcp-server
source .venv/bin/activate
python server.py
```

### End of Day

```bash
# Commit any changes to GitHub
cd ~/Downloads/uniq-mcp-server
git add -A
git commit -m "Daily progress: $(date +%Y-%m-%d)"
git push
```

---

## 4. Research Usage Examples

### Example 1: Generate Training Data for DRL

```python
# In Python, import the server functions
import sys
sys.path.append('/home/raad/Downloads/uniq-mcp-server')
from server import synthesize_circuit, analyze_quantum_circuit

# Generate circuits at increasing complexity
training_circuits = []
descriptions = [
    "Apply Hadamard gate to qubit 0",
    "Create Bell state with 2 qubits",
    "Prepare 3-qubit GHZ state",
    "2-qubit Quantum Fourier Transform",
]

for desc in descriptions:
    result = synthesize_circuit(desc)
    if result['success']:
        training_circuits.append({
            'description': desc,
            'qasm': result['qasm'],
            'analysis': analyze_quantum_circuit(result['qasm'])
        })
```

### Example 2: Benchmark Your DRL Agent

```python
from server import run_benchmark, list_benchmarks

# Get all available benchmarks
benchmarks = list_benchmarks()

# Run each benchmark and collect results
results = []
for category, circuits in benchmarks['benchmarks'].items():
    for circuit in circuits:
        result = run_benchmark(circuit['name'])
        results.append({
            'category': category,
            'circuit': circuit['name'],
            'success': result['success'],
            'time': result['synthesis_time']
        })

# Calculate success rate
success_rate = sum(1 for r in results if r['success']) / len(results)
print(f"Overall success rate: {success_rate:.1%}")
```

### Example 3: Generate LaTeX for Paper

```python
from server import generate_latex_table

# Your experimental results
data = [
    {'Circuit': 'X Gate', 'Success': '✓', 'Time (s)': '18.2', 'Fidelity': '1.000'},
    {'Circuit': 'Bell State', 'Success': '✓', 'Time (s)': '21.5', 'Fidelity': '0.998'},
    {'Circuit': 'GHZ-3', 'Success': '✓', 'Time (s)': '25.3', 'Fidelity': '0.995'},
    {'Circuit': 'QFT-2', 'Success': '✓', 'Time (s)': '28.7', 'Fidelity': '0.992'},
]

latex = generate_latex_table(
    data,
    caption="UniQ-MCP Circuit Synthesis Performance",
    label="tab:synthesis-performance"
)

# Save to file for your paper
with open('results_table.tex', 'w') as f:
    f.write(latex)
```

---

## 5. Troubleshooting

### Issue: Airlock Connection Failed

```bash
# Check if Airlock is running
curl -s https://YOUR-TUNNEL-URL.trycloudflare.com/health

# If not responding, restart Airlock
~/start-airlock.sh

# Update configuration
~/update-airlock-env.sh
```

### Issue: Synthesis Timeout

The RTX 2070 with Mistral 7B takes ~20-30 seconds per synthesis. If timing out:

1. Check GPU utilization: `nvidia-smi`
2. Ensure no other processes are using the GPU
3. Consider simpler circuit descriptions

### Issue: MCP Server Not Recognized in Manus

1. Verify the server path is correct in MCP config
2. Ensure the virtual environment path is correct
3. Try restarting Manus

### Issue: Test Failures

```bash
# Run tests with verbose output
cd ~/Downloads/uniq-mcp-server
source .venv/bin/activate
python -v test_server.py

# Check server logs
cat ~/.airlock/server.log
```

---

## File Structure

```
~/Downloads/uniq-mcp-server/
├── server.py           # Main MCP server (9 tools)
├── test_server.py      # Test suite
├── .env                # Airlock configuration (auto-updated)
├── .env.example        # Template for .env
├── requirements.txt    # Python dependencies
├── start_server.sh     # Server startup script
├── setup_github.sh     # GitHub setup script
├── README.md           # Project overview
├── API_DESIGN.md       # API documentation
├── MANUS_INTEGRATION.md # Manus setup guide
├── RESEARCH_GUIDE.md   # Research usage examples
├── COMPLETE_SETUP.md   # This file
└── .venv/              # Python virtual environment
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start Airlock | `~/start-airlock.sh` |
| Update config | `~/update-airlock-env.sh` |
| Run tests | `python test_server.py` |
| Start server | `python server.py` |
| Push to GitHub | `git add -A && git commit -m "msg" && git push` |

---

*UniQ-MCP v13 - Ready for quantum DRL research*
