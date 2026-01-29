# UniQ-MCP v3.0 - SOAR Framework Implementation

**Version:** 3.0 (Phase 3 Complete)  
**Date:** January 2026  
**Status:** ✅ All Tests Passing (17/17)

## Overview

UniQ-MCP v3.0 implements the full **SOAR (Self-Optimization via Asymmetric RL)** framework for quantum circuit synthesis. This version introduces a Teacher-Student architecture where a powerful reasoning model (DeepSeek-R1) generates "stepping stone" problems to guide a smaller Student model (Mistral 7B) through increasingly complex synthesis tasks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UniQ-MCP v3.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    Stepping Stones    ┌─────────────────────┐ │
│  │   Teacher   │ ─────────────────────▶│      Student        │ │
│  │ DeepSeek-R1 │                       │    Mistral 7B       │ │
│  │ (OpenRouter)│ ◀─────────────────────│    (Airlock)        │ │
│  └─────────────┘    Grounded Reward    └─────────────────────┘ │
│         │                                       │               │
│         ▼                                       ▼               │
│  ┌─────────────┐                       ┌─────────────────────┐ │
│  │  Curriculum │                       │   Circuit Synth     │ │
│  │   Manager   │                       │   & Verification    │ │
│  │ (RL Pacing) │                       │     (Qiskit)        │ │
│  └─────────────┘                       └─────────────────────┘ │
│         │                                       │               │
│         ▼                                       ▼               │
│  ┌─────────────┐                       ┌─────────────────────┐ │
│  │  Episodic   │                       │  Quantum Hardware   │ │
│  │   Memory    │                       │  IBM/IonQ/Simulator │ │
│  │ (ChromaDB)  │                       └─────────────────────┘ │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## New Modules

### 1. Teacher Module (`teacher.py`)

The Teacher generates stepping stone problems that bridge the gap between the Student's current capabilities and hard target problems.

**Key Features:**
- DeepSeek-R1 reasoning model via OpenRouter
- Stepping stone generation with difficulty estimation
- Curriculum generation (multiple stepping stones)
- Solution evaluation with feedback
- ChromaDB-based episodic memory

**Usage:**
```python
from teacher import TeacherClient

async with TeacherClient() as teacher:
    stepping_stone = await teacher.generate_stepping_stone(
        target_problem="Create a 5-qubit GHZ state with depth < 10",
        capability_level=0.3,
        failure_trace="Failed on 4-qubit version"
    )
```

### 2. Curriculum Module (`curriculum.py`)

RL-based adaptive curriculum pacing using Multi-Armed Bandit (UCB) for problem selection.

**Key Features:**
- UCB-based problem selection
- Performance tracking by category
- Adaptive difficulty adjustment
- Grounded reward computation for Teacher
- 26 curriculum problems across 6 categories

**Categories:**
| Category | Problems | Difficulty Range |
|----------|----------|------------------|
| basic_gates | 6 | 0.1 - 0.15 |
| two_qubit | 4 | 0.2 - 0.3 |
| entanglement | 6 | 0.3 - 0.55 |
| algorithms | 5 | 0.5 - 0.75 |
| advanced | 5 | 0.7 - 0.95 |

### 3. Multi-Agent Module (`multi_agent.py`)

Parallel synthesis across multiple inference backends with voting-based selection.

**Supported Backends:**
- **Airlock** (Local GPU - Mistral 7B) - Priority 1
- **OpenRouter** (Cloud - DeepSeek) - Priority 2
- **Ollama** (Local alternative) - Priority 3

**Features:**
- Parallel synthesis with configurable agents
- Fallback chain for reliability
- Code similarity voting
- Backend health monitoring

### 4. Quantum Hardware Module (`quantum_hardware.py`)

Integration with real quantum hardware providers.

**Providers:**
| Provider | Status | Notes |
|----------|--------|-------|
| Simulator | ✅ Ready | Local Qiskit simulation |
| IBM Quantum | 🔧 Configured | Requires IBM_QUANTUM_TOKEN |
| IonQ (Braket) | ⏳ Pending | AWS Braket activation pending |

**Features:**
- Automatic fallback to simulator
- Job timeout handling
- Qubit limit enforcement (free tier)
- Result caching

## MCP Tools (20 Total)

### Core Synthesis
| Tool | Description |
|------|-------------|
| `synthesize_circuit` | Generate circuit from description |
| `synthesize_with_stepping_stones` | Generate curriculum to target |
| `parallel_synthesize` | Multi-agent parallel synthesis |

### Verification & Analysis
| Tool | Description |
|------|-------------|
| `verify_circuit` | Compare two circuits |
| `analyze_quantum_circuit` | Get circuit metrics |

### Curriculum & Learning
| Tool | Description |
|------|-------------|
| `get_curriculum_problem` | Get adaptive problem |
| `record_learning_attempt` | Record attempt for pacing |
| `get_curriculum_statistics` | Get learning progress |

### Execution
| Tool | Description |
|------|-------------|
| `execute_on_simulator` | Run on local simulator |
| `execute_on_hardware` | Run on real hardware |
| `get_available_hardware` | Check hardware status |

### Teacher
| Tool | Description |
|------|-------------|
| `generate_stepping_stone` | Get Teacher stepping stone |
| `evaluate_solution` | Get Teacher feedback |

### Memory
| Tool | Description |
|------|-------------|
| `find_similar_circuits` | Search episodic memory |

### Benchmarks
| Tool | Description |
|------|-------------|
| `run_benchmark` | Run specific benchmark |
| `list_benchmarks` | List available benchmarks |

### Utilities
| Tool | Description |
|------|-------------|
| `check_server_status` | Full system status |
| `get_backend_status` | Inference backend status |
| `generate_latex_table` | Export results to LaTeX |

## Installation

### Prerequisites
```bash
# Python packages
pip install qiskit qiskit-aer chromadb httpx mcp

# Optional for IBM Quantum
pip install qiskit-ibm-runtime

# Optional for AWS Braket
pip install amazon-braket-sdk
```

### Environment Variables
```bash
# Required
export AIRLOCK_URL="https://your-tunnel.trycloudflare.com"
export AIRLOCK_API_KEY="your-airlock-key"

# For Teacher (recommended)
export OPENROUTER_API_KEY="your-openrouter-key"

# For IBM Quantum (optional)
export IBM_QUANTUM_TOKEN="your-ibm-token"
```

### Running the Server
```bash
# Use v3 server
python server_v3.py

# Or update symlink
ln -sf server_v3.py server.py
python server.py
```

## Testing

```bash
# Run Phase 3 test suite
python test_phase3.py

# Expected output:
# Total:   18
# Passed:  17 ✅
# Failed:  0 ❌
# Skipped: 1 ⏭️ (AWS Braket pending)
```

## Usage Examples

### Basic Synthesis
```
Using UniQ-MCP, synthesize a Bell state circuit
```

### With Teacher Intervention
```
Using UniQ-MCP with Teacher enabled, synthesize a 5-qubit GHZ state
```

### Curriculum Learning
```
Using UniQ-MCP, generate a curriculum of stepping stones for implementing QFT
```

### Hardware Execution
```
Using UniQ-MCP, execute this Bell state on IBM Quantum hardware
```

## Research Integration

### For Your PhD

This implementation directly supports your research on:

1. **DRL for Quantum Circuit Synthesis** - The Teacher-Student architecture demonstrates curriculum learning for circuit generation

2. **Scalability Analysis** - The curriculum system tracks performance across difficulty levels

3. **Hardware Validation** - Execute synthesized circuits on real quantum hardware (when AWS Braket activates)

### LaTeX Export
```python
# Generate publication-ready tables
results = [await run_benchmark(b) for b in ["bell_state", "ghz_3", "qft_2"]]
latex = await generate_latex_table(results, "Synthesis Benchmark Results")
```

## File Structure

```
uniq-mcp-server/
├── server_v3.py          # Main MCP server (Phase 3)
├── teacher.py            # Teacher module (DeepSeek-R1)
├── curriculum.py         # RL-based curriculum pacing
├── multi_agent.py        # Multi-agent parallel synthesis
├── quantum_hardware.py   # Hardware integration
├── test_phase3.py        # Phase 3 test suite
├── server.py             # Original v2 server (backup)
├── test_server.py        # Original test suite
├── PHASE3_README.md      # This file
├── PHASE3_REQUIREMENTS.md
├── requirements.txt
└── chroma_data/          # Episodic memory storage
```

## Next Steps

1. **AWS Braket Activation** - Once activated, IonQ hardware will auto-enable
2. **Phase 4 Planning** - Advanced features (noise-aware synthesis, VQE integration)
3. **Paper Integration** - Use benchmark results in IEEE TQE submission

## References

- SOAR Framework: Self-Optimization via Asymmetric RL
- DeepSeek-R1: Reasoning model for curriculum generation
- Qiskit 2.x: Quantum circuit framework
- ChromaDB: Vector database for episodic memory
