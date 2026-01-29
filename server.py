"""
UniQ-MCP Server - Quantum Circuit Synthesis MCP Server

This server exposes quantum circuit synthesis capabilities as MCP tools
that can be called directly from AI assistants like Manus.
"""

import logging
import os
import sys
import json
import time
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging to stderr (required for MCP STDIO servers)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("uniq-mcp")

# MCP imports
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("uniq-mcp", dependencies=["qiskit", "httpx", "chromadb"])

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Server configuration loaded from environment variables."""
    airlock_url: str = ""
    airlock_api_key: str = ""
    aws_region: str = "us-east-1"
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            airlock_url=os.getenv("AIRLOCK_URL", ""),
            airlock_api_key=os.getenv("AIRLOCK_API_KEY", ""),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )

config = Config.from_env()

# ============================================================================
# Airlock Client (Local GPU)
# ============================================================================

import httpx

async def airlock_generate(prompt: str, max_tokens: int = 500) -> str:
    """Generate text using Airlock (local GPU with Mistral 7B)."""
    if not config.airlock_url or not config.airlock_api_key:
        raise ValueError("Airlock not configured. Set AIRLOCK_URL and AIRLOCK_API_KEY.")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{config.airlock_url}/generate",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.airlock_api_key}"
            },
            json={"prompt": prompt, "max_tokens": max_tokens}
        )
        response.raise_for_status()
        data = response.json()
        # Mistral returns the text in 'response' field
        return data.get("response", data.get("generated_text", data.get("text", "")))

async def check_airlock_health() -> dict:
    """Check Airlock server health."""
    if not config.airlock_url:
        return {"status": "not_configured", "healthy": False}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{config.airlock_url}/health",
                headers={"Authorization": f"Bearer {config.airlock_api_key}"}
            )
            if response.status_code == 200:
                return {"status": "healthy", "healthy": True, "details": response.json()}
            return {"status": "unhealthy", "healthy": False, "code": response.status_code}
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}

# ============================================================================
# Circuit Synthesis
# ============================================================================

SYNTHESIS_PROMPT = """Generate Qiskit code for: {description}

Rules:
- Use variable name 'qc' for the circuit
- Qiskit 2.x only (no Aer, no execute, no transpile)
- Output ONLY code, no explanations

Example format:
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

Your code:"""

def extract_circuit_code(response: str) -> str:
    """Extract clean circuit code from LLM response."""
    import re
    
    # Handle JSON response format from Airlock
    if isinstance(response, dict):
        response = response.get('response', response.get('generated_text', response.get('text', str(response))))
    
    # Remove markdown code blocks - handle various formats
    # Format: ```python\ncode\n``` or ```\ncode\n```
    code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if code_block_match:
        response = code_block_match.group(1)
    elif "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]
    
    # Clean up the code
    lines = response.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        # Skip simulation-related lines
        if any(skip in line.lower() for skip in ['aer', 'execute', 'backend', 'simulator', 'transpile', 'run(', 'print(', 'draw(']):
            continue
        # Skip empty lines at start
        if not clean_lines and not line.strip():
            continue
        clean_lines.append(line)
    
    code = '\n'.join(clean_lines).strip()
    
    # Try to find any QuantumCircuit variable name and rename to 'qc'
    # Look for patterns like: circuit = QuantumCircuit, my_circuit = QuantumCircuit, etc.
    circuit_var_match = re.search(r'(\w+)\s*=\s*QuantumCircuit\s*\(', code)
    if circuit_var_match:
        var_name = circuit_var_match.group(1)
        if var_name != 'qc':
            # Replace all occurrences of the variable name with 'qc'
            code = re.sub(rf'\b{var_name}\b', 'qc', code)
    
    # Ensure import statement
    if 'from qiskit import QuantumCircuit' not in code and 'import qiskit' not in code:
        code = 'from qiskit import QuantumCircuit\n' + code
    
    # If no QuantumCircuit found at all, try to construct from description
    if 'QuantumCircuit' not in code and 'qc' not in code:
        # Check if there are gate operations mentioned
        if any(gate in code.lower() for gate in ['h(', 'x(', 'cx(', 'cz(', 'cnot']):
            code = 'from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\n' + code
    
    return code

async def synthesize_with_airlock(description: str) -> dict:
    """Synthesize a quantum circuit using Airlock."""
    prompt = SYNTHESIS_PROMPT.format(description=description)
    
    try:
        response = await airlock_generate(prompt, max_tokens=500)
        code = extract_circuit_code(response)
        
        # Validate the code
        try:
            exec_globals = {}
            exec(code, exec_globals)
            if 'qc' in exec_globals:
                return {
                    "success": True,
                    "code": code,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "code": code,
                    "error": "No circuit 'qc' found in generated code"
                }
        except Exception as e:
            return {
                "success": False,
                "code": code,
                "error": f"Code validation failed: {str(e)}"
            }
    except Exception as e:
        return {
            "success": False,
            "code": None,
            "error": f"Generation failed: {str(e)}"
        }

# ============================================================================
# Circuit Verification
# ============================================================================

def verify_circuits(code1: str, code2: str) -> dict:
    """Verify that two circuits are equivalent."""
    try:
        from qiskit.quantum_info import Operator
        
        # Execute both codes to get circuits
        exec_globals1 = {}
        exec_globals2 = {}
        exec(code1, exec_globals1)
        exec(code2, exec_globals2)
        
        qc1 = exec_globals1.get('qc')
        qc2 = exec_globals2.get('qc')
        
        if qc1 is None or qc2 is None:
            return {
                "equivalent": False,
                "method": "error",
                "details": "Could not extract circuits from code"
            }
        
        # Compare using unitary matrices
        op1 = Operator(qc1)
        op2 = Operator(qc2)
        
        equivalent = op1.equiv(op2)
        
        return {
            "equivalent": equivalent,
            "method": "unitary_comparison",
            "details": "Circuits are equivalent (up to global phase)" if equivalent else "Circuits are not equivalent"
        }
    except Exception as e:
        return {
            "equivalent": False,
            "method": "error",
            "details": f"Verification failed: {str(e)}"
        }

def analyze_circuit(code: str) -> dict:
    """Analyze circuit properties."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        qc = exec_globals.get('qc')
        
        if qc is None:
            return {"error": "No circuit 'qc' found"}
        
        # Count gates by type
        gate_counts = {}
        for instruction in qc.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        return {
            "num_qubits": qc.num_qubits,
            "depth": qc.depth(),
            "gate_count": sum(gate_counts.values()),
            "gates_by_type": gate_counts,
            "num_parameters": qc.num_parameters
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Benchmarks
# ============================================================================

BENCHMARK_PROBLEMS = [
    {"id": "bell_state", "category": "entanglement", "difficulty": 1, "description": "Create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2"},
    {"id": "ghz_3", "category": "entanglement", "difficulty": 2, "description": "Create a 3-qubit GHZ state"},
    {"id": "ghz_5", "category": "entanglement", "difficulty": 3, "description": "Create a 5-qubit GHZ state"},
    {"id": "x_gate", "category": "basic_gates", "difficulty": 1, "description": "Apply X gate to qubit 0"},
    {"id": "hadamard", "category": "basic_gates", "difficulty": 1, "description": "Apply Hadamard gate to create superposition"},
    {"id": "cnot", "category": "basic_gates", "difficulty": 1, "description": "Apply CNOT gate with qubit 0 as control and qubit 1 as target"},
    {"id": "swap", "category": "basic_gates", "difficulty": 2, "description": "Implement SWAP gate using only CNOT gates"},
    {"id": "toffoli", "category": "multi_qubit", "difficulty": 3, "description": "Implement Toffoli (CCX) gate"},
    {"id": "qft_2", "category": "algorithms", "difficulty": 3, "description": "Implement 2-qubit Quantum Fourier Transform"},
    {"id": "qft_3", "category": "algorithms", "difficulty": 4, "description": "Implement 3-qubit Quantum Fourier Transform"},
    {"id": "grover_2", "category": "algorithms", "difficulty": 4, "description": "Implement Grover's algorithm for 2 qubits"},
    {"id": "teleportation", "category": "protocols", "difficulty": 3, "description": "Implement quantum teleportation circuit"},
]

# Reference solutions for verification
REFERENCE_SOLUTIONS = {
    "bell_state": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)""",
    "ghz_3": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)""",
    "x_gate": """from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.x(0)""",
    "hadamard": """from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.h(0)""",
    "cnot": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.cx(0, 1)""",
}

# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def synthesize_circuit(
    description: str,
    num_qubits: int = 0,
    max_attempts: int = 3
) -> dict:
    """Generate a quantum circuit from a natural language description.
    
    Args:
        description: Natural language description of the desired circuit
        num_qubits: Number of qubits (auto-detected if 0)
        max_attempts: Maximum synthesis attempts
    
    Returns:
        Dictionary with code, success status, and verification result
    """
    logger.info(f"Synthesizing circuit: {description}")
    
    for attempt in range(max_attempts):
        result = await synthesize_with_airlock(description)
        
        if result["success"]:
            # Analyze the generated circuit
            analysis = analyze_circuit(result["code"])
            
            return {
                "success": True,
                "code": result["code"],
                "attempts": attempt + 1,
                "analysis": analysis,
                "verification_result": "Code validated successfully"
            }
    
    return {
        "success": False,
        "code": result.get("code"),
        "attempts": max_attempts,
        "error": result.get("error", "Max attempts reached"),
        "verification_result": "Failed"
    }

@mcp.tool()
async def verify_circuit(
    circuit_code: str,
    reference_code: str,
    method: str = "unitary"
) -> dict:
    """Verify that a circuit implements the expected functionality.
    
    Args:
        circuit_code: Qiskit circuit code to verify
        reference_code: Reference circuit code for comparison
        method: Verification method (unitary, simulation)
    
    Returns:
        Dictionary with equivalence result and details
    """
    logger.info("Verifying circuit equivalence")
    return verify_circuits(circuit_code, reference_code)

@mcp.tool()
async def analyze_quantum_circuit(circuit_code: str) -> dict:
    """Analyze circuit properties and metrics.
    
    Args:
        circuit_code: Qiskit circuit code to analyze
    
    Returns:
        Dictionary with circuit metrics (depth, gate count, etc.)
    """
    logger.info("Analyzing circuit")
    return analyze_circuit(circuit_code)

@mcp.tool()
async def run_benchmark(
    benchmark_id: str,
    max_attempts: int = 3
) -> dict:
    """Run a specific benchmark problem.
    
    Args:
        benchmark_id: Benchmark problem ID (e.g., 'bell_state', 'ghz_3')
        max_attempts: Maximum synthesis attempts
    
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running benchmark: {benchmark_id}")
    
    # Find the benchmark
    benchmark = None
    for b in BENCHMARK_PROBLEMS:
        if b["id"] == benchmark_id:
            benchmark = b
            break
    
    if not benchmark:
        return {"error": f"Benchmark '{benchmark_id}' not found"}
    
    start_time = time.time()
    
    # Synthesize the circuit
    result = await synthesize_circuit(
        description=benchmark["description"],
        max_attempts=max_attempts
    )
    
    elapsed = time.time() - start_time
    
    # Verify against reference if available
    verification = None
    if result["success"] and benchmark_id in REFERENCE_SOLUTIONS:
        verification = verify_circuits(result["code"], REFERENCE_SOLUTIONS[benchmark_id])
    
    return {
        "benchmark_id": benchmark_id,
        "category": benchmark["category"],
        "difficulty": benchmark["difficulty"],
        "success": result["success"],
        "time_taken": round(elapsed, 2),
        "attempts": result.get("attempts", 0),
        "generated_code": result.get("code"),
        "verification": verification,
        "analysis": result.get("analysis")
    }

@mcp.tool()
async def list_benchmarks(category: str = "") -> dict:
    """List available benchmark problems.
    
    Args:
        category: Filter by category (optional)
    
    Returns:
        List of available benchmarks
    """
    if category:
        benchmarks = [b for b in BENCHMARK_PROBLEMS if b["category"] == category]
    else:
        benchmarks = BENCHMARK_PROBLEMS
    
    categories = list(set(b["category"] for b in BENCHMARK_PROBLEMS))
    
    return {
        "benchmarks": benchmarks,
        "total": len(benchmarks),
        "categories": categories
    }

@mcp.tool()
async def execute_on_simulator(
    circuit_code: str,
    shots: int = 1000
) -> dict:
    """Execute a circuit on a local simulator.
    
    Args:
        circuit_code: Qiskit circuit code
        shots: Number of shots (default: 1000)
    
    Returns:
        Dictionary with measurement counts
    """
    logger.info(f"Executing circuit on simulator ({shots} shots)")
    
    try:
        from qiskit.quantum_info import Statevector
        import numpy as np
        
        exec_globals = {}
        exec(circuit_code, exec_globals)
        qc = exec_globals.get('qc')
        
        if qc is None:
            return {"error": "No circuit 'qc' found"}
        
        # Add measurements if not present
        if qc.num_clbits == 0:
            qc.measure_all()
        
        # Simulate using statevector
        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        probs = sv.probabilities_dict()
        
        # Sample from probabilities
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        samples = np.random.choice(len(outcomes), size=shots, p=probabilities)
        counts = {}
        for s in samples:
            outcome = outcomes[s]
            counts[outcome] = counts.get(outcome, 0) + 1
        
        return {
            "counts": counts,
            "shots": shots,
            "num_qubits": qc.num_qubits
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def check_server_status() -> dict:
    """Check the status of all UniQ-MCP server components.
    
    Returns:
        Dictionary with status of each component
    """
    status = {
        "server": "running",
        "components": {}
    }
    
    # Check Airlock
    airlock_status = await check_airlock_health()
    status["components"]["airlock"] = airlock_status
    
    # Check Qiskit
    try:
        import qiskit
        status["components"]["qiskit"] = {
            "status": "available",
            "version": qiskit.__version__
        }
    except ImportError:
        status["components"]["qiskit"] = {"status": "not_installed"}
    
    # Check ChromaDB
    try:
        import chromadb
        status["components"]["chromadb"] = {
            "status": "available",
            "version": chromadb.__version__
        }
    except ImportError:
        status["components"]["chromadb"] = {"status": "not_installed"}
    
    # Check AWS Braket
    try:
        import braket
        try:
            version = braket.__version__
        except AttributeError:
            version = "unknown"
        status["components"]["braket"] = {
            "status": "available",
            "version": version
        }
    except ImportError:
        status["components"]["braket"] = {"status": "not_installed"}
    
    return status

@mcp.tool()
async def get_curriculum_problem(
    difficulty: float = 0.5,
    category: str = ""
) -> dict:
    """Get a problem from the curriculum based on difficulty.
    
    Args:
        difficulty: Target difficulty (0.0 to 1.0)
        category: Problem category filter (optional)
    
    Returns:
        A problem matching the criteria
    """
    # Map difficulty to problem difficulty levels (1-5)
    target_level = int(difficulty * 4) + 1
    
    # Filter problems
    candidates = BENCHMARK_PROBLEMS
    if category:
        candidates = [b for b in candidates if b["category"] == category]
    
    # Find closest difficulty match
    best_match = None
    best_diff = float('inf')
    
    for problem in candidates:
        diff = abs(problem["difficulty"] - target_level)
        if diff < best_diff:
            best_diff = diff
            best_match = problem
    
    if best_match:
        return {
            "problem_id": best_match["id"],
            "description": best_match["description"],
            "difficulty": best_match["difficulty"] / 5.0,
            "category": best_match["category"],
            "hints": []
        }
    
    return {"error": "No matching problem found"}

@mcp.tool()
async def generate_latex_table(
    benchmark_results: list,
    title: str = "Benchmark Results"
) -> str:
    """Generate a LaTeX table from benchmark results.
    
    Args:
        benchmark_results: List of benchmark result dictionaries
        title: Table title
    
    Returns:
        LaTeX table code
    """
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{title}}}
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Benchmark}} & \\textbf{{Category}} & \\textbf{{Difficulty}} & \\textbf{{Success}} & \\textbf{{Time (s)}} \\\\
\\hline
"""
    
    for result in benchmark_results:
        success = "\\checkmark" if result.get("success") else "\\texttimes"
        latex += f"{result.get('benchmark_id', 'N/A')} & {result.get('category', 'N/A')} & {result.get('difficulty', 'N/A')} & {success} & {result.get('time_taken', 'N/A')} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}"""
    
    return latex

# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Run the UniQ-MCP server."""
    logger.info("Starting UniQ-MCP Server...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
