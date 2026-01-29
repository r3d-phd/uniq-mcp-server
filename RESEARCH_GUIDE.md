# UniQ-MCP Research Usage Guide

This guide demonstrates how to use UniQ-MCP for your quantum DRL research, with practical examples for circuit synthesis, benchmarking, and publication preparation.

## Quick Start

### Daily Workflow

```bash
# 1. Start Airlock (if not running)
~/start-airlock.sh

# 2. Update UniQ-MCP configuration
~/update-airlock-env.sh

# 3. Activate environment
cd ~/Downloads/uniq-mcp-server
source .venv/bin/activate

# 4. Run server or tests
python server.py      # For MCP integration
python test_server.py # For verification
```

## Research Use Cases

### 1. Generating Quantum Circuits for DRL Training

UniQ-MCP can generate target circuits for your DRL agent to learn:

```python
from server import synthesize_circuit

# Generate circuits at different complexity levels
circuits = [
    synthesize_circuit("Single qubit in superposition"),
    synthesize_circuit("Bell state with 2 qubits"),
    synthesize_circuit("3-qubit GHZ state"),
    synthesize_circuit("2-qubit QFT circuit"),
    synthesize_circuit("Grover's algorithm for 2-qubit search"),
]
```

### 2. Curriculum Learning for DRL

Use the curriculum system to progressively train your DRL agent:

```python
from server import get_curriculum_problem, run_benchmark

# Start with easy problems
for difficulty in [0.2, 0.4, 0.6, 0.8, 1.0]:
    problem = get_curriculum_problem(difficulty)
    print(f"Training on: {problem['name']} (difficulty: {difficulty})")
    
    # Your DRL agent trains on this circuit
    # ...
    
    # Verify learning
    result = run_benchmark(problem['name'])
    if result['success']:
        print(f"  ✓ Mastered {problem['name']}")
```

### 3. Circuit Verification for Research Validation

Verify that synthesized circuits match expected behavior:

```python
from server import verify_circuit, analyze_quantum_circuit

# Synthesize a circuit
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

# Verify against reference
verification = verify_circuit(qasm, qasm)
print(f"Equivalent: {verification['equivalent']}")
print(f"Method: {verification['method']}")

# Analyze properties
analysis = analyze_quantum_circuit(qasm)
print(f"Qubits: {analysis['num_qubits']}")
print(f"Depth: {analysis['depth']}")
print(f"Gate count: {analysis['gate_count']}")
```

### 4. Generating Publication-Ready Results

Create LaTeX tables for your papers:

```python
from server import generate_latex_table, run_benchmark

# Run benchmarks and collect results
results = []
benchmarks = ['x_gate', 'h_gate', 'bell_state', 'ghz_3', 'qft_2']

for name in benchmarks:
    result = run_benchmark(name)
    results.append({
        'Circuit': name,
        'Success': '✓' if result['success'] else '✗',
        'Time (s)': f"{result['synthesis_time']:.2f}",
        'Attempts': result['attempts']
    })

# Generate LaTeX table
latex = generate_latex_table(
    results,
    caption="UniQ-MCP Synthesis Performance",
    label="tab:synthesis-results"
)
print(latex)
```

## Integration with Your DRL Framework

### Example: Using UniQ-MCP as Target Oracle

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

class UniQMCPOracle:
    """Use UniQ-MCP to generate target circuits for DRL training."""
    
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
    
    def get_target_unitary(self, description: str) -> np.ndarray:
        """Get target unitary from natural language description."""
        # Call UniQ-MCP synthesis
        result = synthesize_circuit(description)
        
        if result['success']:
            # Parse QASM and get unitary
            qc = QuantumCircuit.from_qasm_str(result['qasm'])
            return Operator(qc).data
        else:
            raise ValueError(f"Synthesis failed: {result['error']}")
    
    def verify_agent_circuit(self, agent_qasm: str, target_qasm: str) -> bool:
        """Verify if agent's circuit matches target."""
        result = verify_circuit(agent_qasm, target_qasm)
        return result['equivalent']

# Usage in DRL training loop
oracle = UniQMCPOracle()

# Get target for current training episode
target_unitary = oracle.get_target_unitary("Bell state preparation")

# DRL agent generates a circuit
agent_circuit = drl_agent.generate_circuit(target_unitary)

# Verify correctness
is_correct = oracle.verify_agent_circuit(agent_circuit.qasm(), target_qasm)
reward = 1.0 if is_correct else -0.1
```

### Example: Curriculum-Based DRL Training

```python
class CurriculumDRLTrainer:
    """Train DRL agent using UniQ-MCP curriculum."""
    
    def __init__(self, agent):
        self.agent = agent
        self.current_difficulty = 0.2
        self.mastery_threshold = 0.9
    
    def train_episode(self):
        # Get problem at current difficulty
        problem = get_curriculum_problem(self.current_difficulty)
        
        # Train agent on this problem
        success_rate = self.agent.train_on_circuit(
            problem['qasm'],
            problem['description']
        )
        
        # Adaptive difficulty adjustment
        if success_rate > self.mastery_threshold:
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            print(f"Advancing to difficulty {self.current_difficulty}")
        elif success_rate < 0.5:
            self.current_difficulty = max(0.1, self.current_difficulty - 0.1)
            print(f"Reducing to difficulty {self.current_difficulty}")
        
        return success_rate
```

## Benchmark Categories

UniQ-MCP includes benchmarks organized by difficulty:

| Category | Circuits | Difficulty Range |
|----------|----------|------------------|
| Single Gate | X, Y, Z, H, S, T | 0.1 - 0.2 |
| Entanglement | Bell, GHZ-3, GHZ-4 | 0.3 - 0.5 |
| Algorithms | QFT-2, QFT-3 | 0.6 - 0.8 |
| Advanced | Grover-2 | 0.9 - 1.0 |

## Performance Metrics

Track these metrics for your research:

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class SynthesisMetrics:
    circuit_name: str
    success: bool
    synthesis_time: float
    attempts: int
    gate_count: int
    circuit_depth: int

def collect_metrics(benchmarks: List[str]) -> List[SynthesisMetrics]:
    metrics = []
    
    for name in benchmarks:
        start = time.time()
        result = run_benchmark(name)
        elapsed = time.time() - start
        
        analysis = analyze_quantum_circuit(result.get('qasm', ''))
        
        metrics.append(SynthesisMetrics(
            circuit_name=name,
            success=result['success'],
            synthesis_time=elapsed,
            attempts=result.get('attempts', 1),
            gate_count=analysis.get('gate_count', 0),
            circuit_depth=analysis.get('depth', 0)
        ))
    
    return metrics
```

## Episodic Memory for Learning

UniQ-MCP stores successful syntheses in ChromaDB for future reference:

```python
# The server automatically learns from successful syntheses
# Each successful synthesis is stored with:
# - Natural language description (embedding)
# - Generated QASM code
# - Circuit properties (qubits, depth, gates)

# Future syntheses can retrieve similar past successes
# to improve generation quality
```

## Tips for Research Use

### 1. Batch Processing

For large-scale experiments, process circuits in batches:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def batch_synthesize(descriptions: List[str], max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, synthesize_circuit, desc)
            for desc in descriptions
        ]
        return await asyncio.gather(*tasks)
```

### 2. Error Handling

Always handle synthesis failures gracefully:

```python
def safe_synthesize(description: str, max_retries=3):
    for attempt in range(max_retries):
        result = synthesize_circuit(description)
        if result['success']:
            return result
        time.sleep(1)  # Brief pause before retry
    
    # Log failure for analysis
    logging.warning(f"Failed to synthesize: {description}")
    return None
```

### 3. Result Caching

Cache results to avoid redundant API calls:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_synthesize(description: str):
    return synthesize_circuit(description)
```

## Next Steps

1. **Phase 3**: Implement Teacher-Student architecture with DeepSeek-R1
2. **Phase 4**: Add real quantum hardware integration (IBM Quantum, IonQ)
3. **Publication**: Use benchmark results for your journal paper

## Support

- **Server Issues**: Check `test_server.py` output
- **Airlock Issues**: Check `~/.airlock/server.log`
- **Research Questions**: Document in your research notes

---

*UniQ-MCP v13 - Supporting your quantum DRL research*
