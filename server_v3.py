"""
UniQ-MCP Server v3 - Quantum Circuit Synthesis with SOAR Framework

This server exposes quantum circuit synthesis capabilities as MCP tools
with full SOAR (Self-Optimization via Asymmetric RL) integration.

Phase 3 Features:
- Teacher-Student architecture for stepping stone generation
- RL-based curriculum pacing with adaptive difficulty
- Multi-agent parallel synthesis
- Real quantum hardware integration (IBM Quantum, IonQ, AWS Braket)
- Episodic memory with ChromaDB
"""

import logging
import os
import sys
import json
import time
import asyncio
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging to stderr (required for MCP STDIO servers)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("uniq-mcp")

# MCP imports
from mcp.server.fastmcp import FastMCP

# Phase 3 module imports
from teacher import TeacherClient, TeacherConfig, SteppingStoneMemory, test_teacher_connection
from curriculum import CurriculumManager, CurriculumConfig, CURRICULUM_PROBLEMS, get_problems_by_difficulty
from multi_agent import MultiAgentCoordinator, MultiAgentConfig
from quantum_hardware import QuantumHardwareManager, HardwareProvider, execute_on_quantum_hardware

# Initialize FastMCP server
mcp = FastMCP("uniq-mcp-v3", dependencies=["qiskit", "httpx", "chromadb"])

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Server configuration loaded from environment variables."""
    airlock_url: str = ""
    airlock_api_key: str = ""
    openrouter_api_key: str = ""
    ibm_quantum_token: str = ""
    aws_region: str = "us-east-1"
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            airlock_url=os.getenv("AIRLOCK_URL", ""),
            airlock_api_key=os.getenv("AIRLOCK_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            ibm_quantum_token=os.getenv("IBM_QUANTUM_TOKEN", ""),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

config = Config.from_env()

# Initialize Phase 3 components
curriculum_manager = CurriculumManager()
stepping_stone_memory = SteppingStoneMemory()
multi_agent = MultiAgentCoordinator()
hardware_manager = QuantumHardwareManager()

# ============================================================================
# Airlock Client (Local GPU - Student Model)
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
# Circuit Synthesis (Student)
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
    
    if isinstance(response, dict):
        response = response.get('response', response.get('generated_text', response.get('text', str(response))))
    
    # Remove markdown code blocks
    code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if code_block_match:
        response = code_block_match.group(1)
    elif "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]
    
    # Clean up
    lines = response.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        if any(skip in line.lower() for skip in ['aer', 'execute', 'backend', 'simulator', 'transpile', 'run(', 'print(', 'draw(']):
            continue
        if not clean_lines and not line.strip():
            continue
        clean_lines.append(line)
    
    code = '\n'.join(clean_lines).strip()
    
    # Normalize variable name to 'qc'
    circuit_var_match = re.search(r'(\w+)\s*=\s*QuantumCircuit\s*\(', code)
    if circuit_var_match:
        var_name = circuit_var_match.group(1)
        if var_name != 'qc':
            code = re.sub(rf'\b{var_name}\b', 'qc', code)
    
    # Ensure import
    if 'from qiskit import QuantumCircuit' not in code and 'import qiskit' not in code:
        code = 'from qiskit import QuantumCircuit\n' + code
    
    return code

async def synthesize_with_student(description: str) -> dict:
    """Synthesize using the Student model (Airlock/Mistral 7B)."""
    prompt = SYNTHESIS_PROMPT.format(description=description)
    
    try:
        response = await airlock_generate(prompt, max_tokens=500)
        code = extract_circuit_code(response)
        
        # Validate
        try:
            exec_globals = {}
            exec(code, exec_globals)
            if 'qc' in exec_globals:
                return {"success": True, "code": code, "error": None}
            else:
                return {"success": False, "code": code, "error": "No circuit 'qc' found"}
        except Exception as e:
            return {"success": False, "code": code, "error": f"Validation failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "code": None, "error": f"Generation failed: {str(e)}"}

# ============================================================================
# Circuit Verification
# ============================================================================

def verify_circuits(code1: str, code2: str) -> dict:
    """Verify that two circuits are equivalent."""
    try:
        from qiskit.quantum_info import Operator
        
        exec_globals1 = {}
        exec_globals2 = {}
        exec(code1, exec_globals1)
        exec(code2, exec_globals2)
        
        qc1 = exec_globals1.get('qc')
        qc2 = exec_globals2.get('qc')
        
        if qc1 is None or qc2 is None:
            return {"equivalent": False, "method": "error", "details": "Could not extract circuits"}
        
        op1 = Operator(qc1)
        op2 = Operator(qc2)
        equivalent = op1.equiv(op2)
        
        return {
            "equivalent": equivalent,
            "method": "unitary_comparison",
            "details": "Equivalent (up to global phase)" if equivalent else "Not equivalent"
        }
    except Exception as e:
        return {"equivalent": False, "method": "error", "details": str(e)}

def analyze_circuit(code: str) -> dict:
    """Analyze circuit properties."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        qc = exec_globals.get('qc')
        
        if qc is None:
            return {"error": "No circuit 'qc' found"}
        
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
# Reference Solutions
# ============================================================================

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
    "h_gate": """from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.h(0)""",
    "cnot": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.cx(0, 1)""",
}

# ============================================================================
# MCP Tools - Core Synthesis
# ============================================================================

@mcp.tool()
async def synthesize_circuit(
    description: str,
    use_teacher: bool = False,
    max_attempts: int = 3
) -> dict:
    """Generate a quantum circuit from natural language description.
    
    Args:
        description: Natural language description of the desired circuit
        use_teacher: Whether to use Teacher for stepping stones if Student fails
        max_attempts: Maximum synthesis attempts
    
    Returns:
        Dictionary with code, success status, and analysis
    """
    logger.info(f"Synthesizing circuit: {description}")
    
    # Try with Student first
    for attempt in range(max_attempts):
        result = await synthesize_with_student(description)
        
        if result["success"]:
            analysis = analyze_circuit(result["code"])
            
            # Record success in curriculum
            curriculum_manager.record_attempt(
                problem_id=f"custom_{hash(description) % 10000}",
                description=description,
                difficulty=0.5,  # Estimate
                success=True,
                synthesis_time=0,
                attempts=attempt + 1,
                category="custom"
            )
            
            # Store in episodic memory
            stepping_stone_memory.store_stepping_stone(
                problem_desc=description,
                circuit_qasm=result["code"],
                difficulty=0.5,
                verified=True
            )
            
            return {
                "success": True,
                "code": result["code"],
                "attempts": attempt + 1,
                "analysis": analysis,
                "method": "student"
            }
    
    # If Student failed and Teacher is enabled
    if use_teacher and config.openrouter_api_key:
        logger.info("Student failed, requesting Teacher stepping stone...")
        
        async with TeacherClient() as teacher:
            stepping_stone = await teacher.generate_stepping_stone(
                target_problem=description,
                capability_level=curriculum_manager.state.capability_level,
                failure_trace=result.get("error", "")
            )
            
            if stepping_stone["success"]:
                return {
                    "success": False,
                    "original_error": result.get("error"),
                    "stepping_stone": stepping_stone["stepping_stone"],
                    "recommendation": "Try the stepping stone problem first",
                    "method": "teacher_intervention"
                }
    
    return {
        "success": False,
        "code": result.get("code"),
        "attempts": max_attempts,
        "error": result.get("error", "Max attempts reached"),
        "method": "student_failed"
    }

@mcp.tool()
async def synthesize_with_stepping_stones(
    target_description: str,
    num_steps: int = 3
) -> dict:
    """Generate a curriculum of stepping stones leading to the target circuit.
    
    This uses the Teacher to create a learning path from simple to complex.
    
    Args:
        target_description: Description of the target (hard) circuit
        num_steps: Number of stepping stones to generate
    
    Returns:
        Dictionary with curriculum and synthesis results
    """
    if not config.openrouter_api_key:
        return {"error": "Teacher not configured. Set OPENROUTER_API_KEY."}
    
    logger.info(f"Generating curriculum for: {target_description}")
    
    async with TeacherClient() as teacher:
        curriculum = await teacher.generate_curriculum(
            target_problem=target_description,
            num_steps=num_steps,
            start_difficulty=curriculum_manager.state.capability_level
        )
        
        if not curriculum["success"]:
            return {"error": "Failed to generate curriculum", "details": curriculum}
        
        # Try to synthesize each stepping stone
        results = []
        for step in curriculum["curriculum"]:
            ss = step.get("stepping_stone", {})
            desc = ss.get("description", "")
            
            if desc:
                synth_result = await synthesize_with_student(desc)
                results.append({
                    "step": step["step_number"],
                    "description": desc,
                    "difficulty": ss.get("estimated_difficulty", 0),
                    "synthesis_success": synth_result["success"],
                    "code": synth_result.get("code") if synth_result["success"] else None
                })
        
        return {
            "success": True,
            "target": target_description,
            "curriculum": results,
            "num_steps_completed": sum(1 for r in results if r["synthesis_success"])
        }

# ============================================================================
# MCP Tools - Verification & Analysis
# ============================================================================

@mcp.tool()
async def verify_circuit(circuit_code: str, reference_code: str) -> dict:
    """Verify that a circuit implements the expected functionality.
    
    Args:
        circuit_code: Qiskit circuit code to verify
        reference_code: Reference circuit code for comparison
    
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
        Dictionary with circuit metrics
    """
    logger.info("Analyzing circuit")
    return analyze_circuit(circuit_code)

# ============================================================================
# MCP Tools - Curriculum & Learning
# ============================================================================

@mcp.tool()
async def get_curriculum_problem(
    difficulty: float = 0.5,
    category: str = ""
) -> dict:
    """Get a problem from the adaptive curriculum.
    
    Args:
        difficulty: Target difficulty (0.0 to 1.0)
        category: Problem category filter (optional)
    
    Returns:
        A problem matching the criteria
    """
    problems = get_problems_by_difficulty(
        min_diff=max(0, difficulty - 0.15),
        max_diff=min(1, difficulty + 0.15),
        category=category
    )
    
    if not problems:
        problems = CURRICULUM_PROBLEMS
    
    selected = curriculum_manager.get_next_problem(problems, category)
    
    if selected:
        return {
            "problem_id": selected["id"],
            "description": selected["description"],
            "difficulty": selected["difficulty"],
            "category": selected["category"],
            "student_capability": curriculum_manager.state.capability_level
        }
    
    return {"error": "No matching problem found"}

@mcp.tool()
async def record_learning_attempt(
    problem_id: str,
    description: str,
    difficulty: float,
    success: bool,
    synthesis_time: float,
    category: str = ""
) -> dict:
    """Record a learning attempt for curriculum adaptation.
    
    Args:
        problem_id: Problem identifier
        description: Problem description
        difficulty: Problem difficulty (0-1)
        success: Whether synthesis succeeded
        synthesis_time: Time taken in seconds
        category: Problem category
    
    Returns:
        Updated curriculum state and recommendations
    """
    return curriculum_manager.record_attempt(
        problem_id=problem_id,
        description=description,
        difficulty=difficulty,
        success=success,
        synthesis_time=synthesis_time,
        attempts=1,
        category=category
    )

@mcp.tool()
async def get_curriculum_statistics() -> dict:
    """Get comprehensive curriculum learning statistics.
    
    Returns:
        Statistics about learning progress
    """
    return curriculum_manager.get_statistics()

# ============================================================================
# MCP Tools - Execution
# ============================================================================

@mcp.tool()
async def execute_on_simulator(circuit_code: str, shots: int = 1000) -> dict:
    """Execute a circuit on a local simulator.
    
    Args:
        circuit_code: Qiskit circuit code
        shots: Number of shots (default: 1000)
    
    Returns:
        Dictionary with measurement counts
    """
    logger.info(f"Executing circuit on simulator ({shots} shots)")
    return await execute_on_quantum_hardware(circuit_code, "simulator", "", shots)

@mcp.tool()
async def execute_on_hardware(
    circuit_code: str,
    provider: str = "simulator",
    backend: str = "",
    shots: int = 1000
) -> dict:
    """Execute a circuit on quantum hardware.
    
    Args:
        circuit_code: Qiskit circuit code
        provider: "simulator", "ibm_quantum", or "ionq"
        backend: Specific backend name (optional)
        shots: Number of shots
    
    Returns:
        Dictionary with execution results
    
    Note: IBM Quantum and IonQ require respective API tokens.
          AWS Braket activation is pending.
    """
    logger.info(f"Executing circuit on {provider} ({shots} shots)")
    return await execute_on_quantum_hardware(circuit_code, provider, backend, shots)

@mcp.tool()
async def get_available_hardware() -> dict:
    """Get status of available quantum hardware.
    
    Returns:
        Dictionary with hardware availability status
    """
    return await hardware_manager.get_available_hardware()

# ============================================================================
# MCP Tools - Benchmarks
# ============================================================================

@mcp.tool()
async def run_benchmark(benchmark_id: str, max_attempts: int = 3) -> dict:
    """Run a specific benchmark problem.
    
    Args:
        benchmark_id: Benchmark problem ID
        max_attempts: Maximum synthesis attempts
    
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running benchmark: {benchmark_id}")
    
    # Find benchmark
    benchmark = None
    for b in CURRICULUM_PROBLEMS:
        if b["id"] == benchmark_id:
            benchmark = b
            break
    
    if not benchmark:
        return {"error": f"Benchmark '{benchmark_id}' not found"}
    
    start_time = time.time()
    result = await synthesize_circuit(benchmark["description"], max_attempts=max_attempts)
    elapsed = time.time() - start_time
    
    # Verify against reference
    verification = None
    if result["success"] and benchmark_id in REFERENCE_SOLUTIONS:
        verification = verify_circuits(result["code"], REFERENCE_SOLUTIONS[benchmark_id])
    
    # Record in curriculum
    curriculum_manager.record_attempt(
        problem_id=benchmark_id,
        description=benchmark["description"],
        difficulty=benchmark["difficulty"],
        success=result["success"],
        synthesis_time=elapsed,
        attempts=result.get("attempts", max_attempts),
        category=benchmark["category"]
    )
    
    return {
        "benchmark_id": benchmark_id,
        "category": benchmark["category"],
        "difficulty": benchmark["difficulty"],
        "success": result["success"],
        "synthesis_time": round(elapsed, 2),
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
        benchmarks = [b for b in CURRICULUM_PROBLEMS if b["category"] == category]
    else:
        benchmarks = CURRICULUM_PROBLEMS
    
    # Group by category
    by_category = {}
    for b in benchmarks:
        cat = b["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "id": b["id"],
            "description": b["description"],
            "difficulty": b["difficulty"]
        })
    
    return {
        "benchmarks": by_category,
        "total": len(benchmarks),
        "categories": list(by_category.keys())
    }

# ============================================================================
# MCP Tools - Multi-Agent
# ============================================================================

@mcp.tool()
async def parallel_synthesize(
    description: str,
    num_agents: int = 2
) -> dict:
    """Synthesize using multiple agents in parallel.
    
    Args:
        description: Circuit description
        num_agents: Number of parallel synthesis attempts
    
    Returns:
        Results from all agents with best selection
    """
    logger.info(f"Parallel synthesis with {num_agents} agents")
    
    prompt = SYNTHESIS_PROMPT.format(description=description)
    result = await multi_agent.parallel_synthesize(prompt, num_attempts=num_agents)
    
    # Extract and validate best result
    if result["success"] and result.get("best_result"):
        code = extract_circuit_code(result["best_result"]["code"])
        try:
            exec_globals = {}
            exec(code, exec_globals)
            if 'qc' in exec_globals:
                result["best_result"]["validated_code"] = code
                result["best_result"]["analysis"] = analyze_circuit(code)
        except:
            pass
    
    return result

@mcp.tool()
async def get_backend_status() -> dict:
    """Get status of all inference backends.
    
    Returns:
        Dictionary with backend availability
    """
    return await multi_agent.get_backend_status()

# ============================================================================
# MCP Tools - Teacher
# ============================================================================

@mcp.tool()
async def generate_stepping_stone(
    target_problem: str,
    failure_trace: str = ""
) -> dict:
    """Generate a stepping stone problem using the Teacher.
    
    Args:
        target_problem: The hard target problem
        failure_trace: Previous failure information
    
    Returns:
        Stepping stone problem with reference solution
    """
    if not config.openrouter_api_key:
        return {"error": "Teacher not configured. Set OPENROUTER_API_KEY."}
    
    async with TeacherClient() as teacher:
        return await teacher.generate_stepping_stone(
            target_problem=target_problem,
            capability_level=curriculum_manager.state.capability_level,
            failure_trace=failure_trace
        )

@mcp.tool()
async def evaluate_solution(
    problem_description: str,
    student_code: str,
    reference_code: str
) -> dict:
    """Evaluate a student solution using the Teacher.
    
    Args:
        problem_description: The problem that was attempted
        student_code: The student's generated code
        reference_code: The reference solution
    
    Returns:
        Evaluation with feedback and hints
    """
    if not config.openrouter_api_key:
        return {"error": "Teacher not configured. Set OPENROUTER_API_KEY."}
    
    async with TeacherClient() as teacher:
        return await teacher.evaluate_student_solution(
            problem_description=problem_description,
            student_code=student_code,
            reference_code=reference_code
        )

# ============================================================================
# MCP Tools - Memory
# ============================================================================

@mcp.tool()
async def find_similar_circuits(
    query: str,
    n_results: int = 5
) -> dict:
    """Find similar circuits from episodic memory.
    
    Args:
        query: Problem description to search for
        n_results: Number of results to return
    
    Returns:
        List of similar problems with solutions
    """
    results = stepping_stone_memory.find_similar_problems(query, n_results)
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }

# ============================================================================
# MCP Tools - Status & Utilities
# ============================================================================

@mcp.tool()
async def check_server_status() -> dict:
    """Check the status of all UniQ-MCP server components.
    
    Returns:
        Dictionary with status of each component
    """
    status = {
        "server": "running",
        "version": "3.0 (SOAR)",
        "components": {}
    }
    
    # Check Airlock (Student)
    airlock_status = await check_airlock_health()
    status["components"]["airlock_student"] = airlock_status
    
    # Check Teacher
    if config.openrouter_api_key:
        teacher_status = await test_teacher_connection()
        status["components"]["teacher"] = teacher_status
    else:
        status["components"]["teacher"] = {"status": "not_configured"}
    
    # Check Qiskit
    try:
        import qiskit
        status["components"]["qiskit"] = {"status": "available", "version": qiskit.__version__}
    except ImportError:
        status["components"]["qiskit"] = {"status": "not_installed"}
    
    # Check ChromaDB
    try:
        import chromadb
        status["components"]["chromadb"] = {"status": "available", "version": chromadb.__version__}
    except ImportError:
        status["components"]["chromadb"] = {"status": "not_installed"}
    
    # Check hardware
    status["components"]["quantum_hardware"] = await hardware_manager.get_available_hardware()
    
    # Curriculum state
    status["curriculum"] = {
        "capability_level": curriculum_manager.state.capability_level,
        "total_attempts": curriculum_manager.state.total_attempts,
        "success_rate": curriculum_manager.state.success_rate
    }
    
    return status

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
        latex += f"{result.get('benchmark_id', 'N/A')} & {result.get('category', 'N/A')} & {result.get('difficulty', 'N/A')} & {success} & {result.get('synthesis_time', 'N/A')} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}"""
    
    return latex

# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Run the UniQ-MCP v3 server."""
    logger.info("Starting UniQ-MCP Server v3 (SOAR Framework)...")
    logger.info(f"Airlock: {'configured' if config.airlock_url else 'not configured'}")
    logger.info(f"Teacher: {'configured' if config.openrouter_api_key else 'not configured'}")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
