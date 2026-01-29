# UniQ-MCP Server API Design

## Overview

The UniQ-MCP Server exposes quantum circuit synthesis capabilities as MCP tools that can be called directly from AI assistants like Manus.

## Tool Categories

### 1. Circuit Synthesis Tools

#### `synthesize_circuit`
Generate a quantum circuit from a natural language description.

**Parameters:**
- `description` (str): Natural language description of the desired circuit
- `num_qubits` (int, optional): Number of qubits (auto-detected if not specified)
- `use_stepping_stones` (bool, optional): Use Teacher stepping stones for complex problems

**Returns:**
- `code` (str): Generated Qiskit circuit code
- `success` (bool): Whether synthesis succeeded
- `attempts` (int): Number of attempts taken
- `verification_result` (str): Verification status

#### `synthesize_batch`
Generate multiple circuits in parallel.

**Parameters:**
- `descriptions` (list[str]): List of circuit descriptions
- `parallel` (bool, optional): Use multi-agent parallel synthesis

**Returns:**
- `results` (list): List of synthesis results

### 2. Verification Tools

#### `verify_circuit`
Verify that a circuit implements the expected functionality.

**Parameters:**
- `circuit_code` (str): Qiskit circuit code to verify
- `reference_code` (str): Reference circuit code for comparison
- `method` (str, optional): Verification method ("mqt", "unitary", "simulation")

**Returns:**
- `equivalent` (bool): Whether circuits are equivalent
- `method_used` (str): Verification method used
- `details` (str): Additional verification details

#### `analyze_circuit`
Analyze circuit properties and metrics.

**Parameters:**
- `circuit_code` (str): Qiskit circuit code to analyze

**Returns:**
- `depth` (int): Circuit depth
- `gate_count` (int): Total gate count
- `gates_by_type` (dict): Gate counts by type
- `num_qubits` (int): Number of qubits

### 3. Benchmark Tools

#### `run_benchmark`
Run a specific benchmark problem.

**Parameters:**
- `benchmark_id` (str): Benchmark problem ID
- `category` (str, optional): Filter by category

**Returns:**
- `success` (bool): Whether benchmark passed
- `time_taken` (float): Time in seconds
- `attempts` (int): Number of attempts
- `generated_code` (str): Generated circuit code

#### `run_benchmark_suite`
Run the full benchmark suite.

**Parameters:**
- `categories` (list[str], optional): Categories to run
- `export_format` (str, optional): "markdown", "latex", or "json"

**Returns:**
- `summary` (dict): Overall results summary
- `results` (list): Individual benchmark results
- `export` (str): Formatted export string

#### `list_benchmarks`
List available benchmark problems.

**Parameters:**
- `category` (str, optional): Filter by category

**Returns:**
- `benchmarks` (list): List of available benchmarks

### 4. Curriculum Tools

#### `get_next_problem`
Get the next problem from the adaptive curriculum.

**Parameters:**
- `difficulty` (float, optional): Target difficulty (0-1)
- `category` (str, optional): Problem category

**Returns:**
- `problem_id` (str): Problem identifier
- `description` (str): Problem description
- `difficulty` (float): Problem difficulty
- `hints` (list[str]): Optional hints

#### `record_attempt`
Record an attempt result for curriculum adaptation.

**Parameters:**
- `problem_id` (str): Problem identifier
- `success` (bool): Whether attempt succeeded
- `attempts` (int): Number of attempts taken
- `time_taken` (float): Time in seconds

**Returns:**
- `curriculum_updated` (bool): Whether curriculum was updated
- `next_difficulty` (float): Recommended next difficulty

#### `get_curriculum_stats`
Get curriculum learning statistics.

**Returns:**
- `total_problems` (int): Total problems attempted
- `success_rate` (float): Overall success rate
- `current_difficulty` (float): Current difficulty level
- `learning_curve` (list): Learning curve data

### 5. Hardware Execution Tools

#### `execute_on_simulator`
Execute a circuit on a local simulator.

**Parameters:**
- `circuit_code` (str): Qiskit circuit code
- `shots` (int, optional): Number of shots (default: 1000)
- `simulator` (str, optional): Simulator type ("aer", "braket_local")

**Returns:**
- `counts` (dict): Measurement counts
- `execution_time` (float): Execution time in seconds

#### `execute_on_braket`
Execute a circuit on AWS Braket (simulator or hardware).

**Parameters:**
- `circuit_code` (str): Qiskit circuit code
- `device` (str): Device name ("sv1", "ionq_forte", "iqm_garnet", etc.)
- `shots` (int, optional): Number of shots (default: 1000)
- `estimate_only` (bool, optional): Only estimate cost, don't execute

**Returns:**
- `counts` (dict): Measurement counts (if executed)
- `estimated_cost` (float): Estimated cost in USD
- `execution_time` (float): Execution time (if executed)
- `task_arn` (str): AWS task ARN (if executed)

#### `list_available_devices`
List available quantum devices.

**Returns:**
- `simulators` (list): Available simulators
- `hardware` (list): Available hardware devices
- `status` (dict): Device availability status

### 6. Evaluation Tools

#### `get_metrics`
Get evaluation metrics for recorded attempts.

**Parameters:**
- `metric_type` (str, optional): "accuracy", "efficiency", "scalability", "learning", "robustness"

**Returns:**
- `metrics` (list): Computed metrics
- `summary` (dict): Metrics summary

#### `generate_report`
Generate a comprehensive evaluation report.

**Parameters:**
- `title` (str): Report title
- `format` (str, optional): "markdown", "latex", or "json"

**Returns:**
- `report` (str): Formatted report
- `recommendations` (list): Improvement recommendations

### 7. Meta-Learning Tools

#### `get_domain_knowledge`
Get relevant knowledge for a problem domain.

**Parameters:**
- `domain` (str): Problem domain
- `task_description` (str): Task description

**Returns:**
- `patterns` (list): Relevant patterns
- `success_rates` (dict): Historical success rates
- `recommendations` (list): Approach recommendations

#### `update_knowledge`
Update meta-learning knowledge from experience.

**Parameters:**
- `domain` (str): Problem domain
- `pattern_name` (str): Pattern identifier
- `success` (bool): Whether pattern succeeded
- `context` (dict): Additional context

**Returns:**
- `knowledge_updated` (bool): Whether knowledge was updated

## Configuration

### Environment Variables

```bash
# Airlock (Local GPU)
AIRLOCK_URL=https://your-tunnel.trycloudflare.com
AIRLOCK_API_KEY=your-api-key

# AWS Braket (optional)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Cloud LLMs (optional, for Teacher)
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
```

## Usage Examples

### From Manus Conversation

```
User: "Generate a Bell state circuit"
Manus: [calls synthesize_circuit with description="Bell state circuit"]
→ Returns verified Qiskit code

User: "Run the benchmark suite and export to LaTeX"
Manus: [calls run_benchmark_suite with export_format="latex"]
→ Returns publication-ready LaTeX table

User: "Execute this circuit on IonQ"
Manus: [calls execute_on_braket with device="ionq_forte"]
→ Returns measurement results and cost
```

## Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UniQ-MCP Server                          │
├─────────────────────────────────────────────────────────────┤
│  MCP Interface Layer (FastMCP)                              │
│  ├── Tool Definitions (decorators)                          │
│  ├── Parameter Validation                                   │
│  └── Response Formatting                                    │
├─────────────────────────────────────────────────────────────┤
│  Core Services Layer                                        │
│  ├── SynthesisService (Student + Teacher + Verifier)        │
│  ├── BenchmarkService (Benchmark Suite)                     │
│  ├── CurriculumService (Adaptive Curriculum)                │
│  ├── HardwareService (Simulators + Braket)                  │
│  ├── EvaluationService (Metrics + Reports)                  │
│  └── MetaLearningService (Domain Transfer)                  │
├─────────────────────────────────────────────────────────────┤
│  Backend Layer                                              │
│  ├── Airlock Client (Local GPU)                             │
│  ├── Cloud LLM Clients (Gemini, OpenRouter)                 │
│  ├── MQT Verifier                                           │
│  ├── ChromaDB (Episodic Memory)                             │
│  └── AWS Braket SDK                                         │
└─────────────────────────────────────────────────────────────┘
```
