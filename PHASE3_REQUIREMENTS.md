# Phase 3 Requirements - Full SOAR Implementation

## Source: DesigningUniQ-MCPv13withSOAR.pdf

## Core Architecture (from design doc)

### 1. SOAR Framework (Self-Optimization via Asymmetric RL)
- **Stepping Stone Hypothesis**: Teacher generates intermediate problems to guide Student
- **Grounded Reward**: Teacher rewarded by Student's measured improvement, not intrinsic quality
- **Memory-Based Adaptation**: Uses In-Context Learning + Episodic Memory instead of weight updates

### 2. Teacher-Student Architecture

#### Teacher Node: DeepSeek-R1-Distill-Llama-70B
- **Role**: "Prefrontal Cortex" - high-level planning, stepping stone generation, complex verification
- **Capabilities**: 70B parameters, AIME 2024 pass@1: 70.0%, MATH-500: 94.5%
- **Deployment**: High-memory node (A100 80GB or multiple A100s), not in critical path
- **For our setup**: Use cloud API (OpenRouter/DeepSeek API) or Aziz Supercomputer

#### Student Node: DeepSeek-R1-Distill-Llama-8B (or Mistral 7B)
- **Role**: "Motor Cortex" - real-time interactions, circuit drafting, executing stepping stones
- **Capabilities**: 8B model, AIME 2024 pass@1: 50.4%, MATH-500: 89.1%
- **Efficiency**: 10x faster and cheaper than 70B
- **For our setup**: Mistral 7B on RTX 2070 via Airlock (already working)

### 3. Three-Layer Architecture

#### Compute Layer (Brains)
- Teacher: DeepSeek-R1-70B for curriculum generation
- Student: Mistral 7B for execution (via Airlock)

#### Transport Layer (Nervous System)
- FastMCP for tool orchestration
- Docket for async task queue (Redis Streams)
- Background workers for "Dreaming" process

#### Domain Layer (Grounding Reality)
- MQT DDSIM: Decision diagram simulator for fast verification
- MQT QCEC: Quantum circuit equivalence checking (already integrated)
- ChromaDB: Episodic memory storage (already integrated)

## Phase 3 Implementation Tasks

### Task 1: Teacher Module
- [ ] Integrate DeepSeek-R1-70B via OpenRouter API
- [ ] Implement stepping stone generation algorithm
- [ ] Create curriculum difficulty progression
- [ ] Add Teacher-Student communication protocol

### Task 2: RL-Based Curriculum Pacing
- [ ] Implement adaptive difficulty adjustment
- [ ] Track Student performance metrics
- [ ] Create reward signal from Student improvement
- [ ] Implement curriculum progression logic

### Task 3: Multi-Agent Collaboration
- [ ] Parallel problem solving with multiple Students
- [ ] Result aggregation and voting
- [ ] Load balancing across inference endpoints

### Task 4: Real Quantum Hardware Integration
- [ ] IBM Quantum via MCP (ibm-quantum server)
- [ ] IonQ via MCP (ionq-quantum server)
- [ ] Fallback to simulator when hardware unavailable

### Task 5: Dreaming Layer (Background Consolidation)
- [ ] Async task queue with Docket/Redis
- [ ] Memory consolidation during idle time
- [ ] Pruning invalid solutions
- [ ] Reinforcing successful reasoning traces

## API Endpoints for Teacher (OpenRouter)

```python
# DeepSeek-R1 via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TEACHER_MODEL = "deepseek/deepseek-r1"  # or "deepseek/deepseek-r1-distill-llama-70b"

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Endpoint: https://openrouter.ai/api/v1/chat/completions
```

## Success Criteria

1. Teacher successfully generates stepping stones for complex circuits
2. Student shows measurable improvement on validation set
3. System can synthesize circuits beyond current capability (e.g., 5+ qubit circuits)
4. Real quantum hardware execution works for at least one provider
5. Memory consolidation improves synthesis quality over time


## SOAR Meta-RL Loop Algorithm (from Section 4)

### Phase 1: Generation (Teacher)
The Teacher (70B model) analyzes a "Hard Problem" from the MQT Bench backlog that the Student has failed to solve.

**Input**: Target problem description + Student failure trace
**Action**: Generate batch of synthetic Stepping Stones (simplified versions)
**Output**: Problem description + reference solution/unitary

### Phase 2: Attempt (Student)
The Student (8B model) attempts to solve the stepping stone.

**Action**: Generate candidate OpenQASM string
**Tool Use**: May call MQT DDSIM to debug circuit during generation

### Phase 3: Verification & Grounded Reward (Environment)
```python
from mqt import qcec
result = qcec.verify(student_circuit, teacher_reference)
success = (result.equivalence == 'equivalent')
```

**Reward**: Teacher receives positive reward ONLY IF Student's performance on Hard Problem improves after training on stepping stone.

## Prompt Engineering

### Teacher System Prompt
```
"You are the SOAR Teacher for a Quantum Logic Synthesis agent. Your goal is
NOT to solve the problem directly, but to generate a 'Stepping Stone' problem
that bridges the gap between the Student's current capabilities and the target
Hard Problem.

Target: {target_problem_description}
Student Failure: {failure_trace}

Generate a simplified version of this problem (e.g., fewer qubits, relaxed depth
constraints, or a specific sub-component). Provide the OpenQASM code for the
reference solution and the expected unitary."
```

### Student System Prompt
```
"You are the UniQ Student. You have access to a library of 'solved templates' in
your Episodic Memory. Use these templates to synthesize a circuit for the given
objective. Verify your intermediate steps using the mqt.ddsim tool. If you are
stuck, query the memory for 'similar stepping stones'."
```

## ChromaDB Schema for Reasoning Traces

Collection: `quantum_reasoning_traces`

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Unique identifier |
| embedding | Vector | Semantic embedding of problem + circuit |
| problem_desc | String | Natural language description |
| circuit_qasm | String | Successful OpenQASM code |
| difficulty | Float | Computed via MQT QCEC runtime |
| stepping_stone | Boolean | True if Teacher-generated |
| verified | Boolean | MQT QCEC verification result |
| parent_id | UUID | Link to original Hard Problem |

## Dreaming Layer (Section 5)

### Process A: Garbage Collection & Pruning
- Cluster vectors with cosine similarity > 0.95
- Keep single "best" trace (lowest circuit depth)
- Delete redundant entries

### Process B: Counterfactual Verification (Active Dreaming)
- Replay failed student attempts
- Mutate variables to find "near misses"
- Use MQT DDSIM to simulate alternatives
- Add synthetic successes to memory

### Process C: Rule Consolidation
- Generalize from specific instances
- Teacher synthesizes general Python functions (e.g., `generate_qft(n_qubits)`)
- Save as new FastMCP Tool

## Docket Integration Code Pattern
```python
from fastmcp import FastMCP
from docket import Docket, Worker
import chromadb

mcp = FastMCP("UniQ-v13-Student", dependencies=["pydocket"])

async def push_trace_to_dream(trace_data: dict):
    async with Docket(name="uniq_dream_queue", url="redis://localhost:6379") as docket:
        await docket.add("process_trace", trace_data)

@mcp.tool(task=True)  # FastMCP native background task
async def log_interaction(problem: str, circuit: str, success: bool):
    trace = {
        "problem": problem,
        "circuit": circuit,
        "success": success,
        "timestamp": "..."
    }
    await push_trace_to_dream(trace)
```
