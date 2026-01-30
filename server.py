"""
UniQ-MCP Server v4 - Simplified Quantum Circuit Synthesis

This server exposes quantum circuit synthesis capabilities as MCP tools
using OpenRouter API directly - no local GPU or tunnels required!

Features:
- Direct OpenRouter integration (DeepSeek, Claude, GPT-4, etc.)
- Teacher-Student architecture for stepping stone generation
- RL-based curriculum pacing with adaptive difficulty
- Real quantum hardware integration (AWS Braket)
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
from teacher import TeacherClient, TeacherConfig, SteppingStoneMemory
from curriculum import CurriculumManager, CurriculumConfig, CURRICULUM_PROBLEMS, get_problems_by_difficulty
from quantum_hardware import QuantumHardwareManager, execute_on_quantum_hardware

# Initialize FastMCP server
mcp = FastMCP("uniq-mcp-v4", dependencies=["qiskit", "httpx", "chromadb"])

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Server configuration loaded from environment variables."""
    openrouter_api_key: str = ""
    default_model: str = "deepseek/deepseek-chat"  # Fast and good for code
    teacher_model: str = "deepseek/deepseek-reasoner"  # DeepSeek-R1 for reasoning
    aws_region: str = "us-east-1"
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            default_model=os.getenv("UNIQ_DEFAULT_MODEL", "deepseek/deepseek-chat"),
            teacher_model=os.getenv("UNIQ_TEACHER_MODEL", "deepseek/deepseek-reasoner"),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

config = Config.from_env()

# Initialize components
curriculum_manager = CurriculumManager()
stepping_stone_memory = SteppingStoneMemory()
hardware_manager = QuantumHardwareManager()

# ============================================================================
# OpenRouter Client (Cloud LLM - replaces Airlock)
# ============================================================================

import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

async def openrouter_generate(
    prompt: str, 
    max_tokens: int = 1000,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """Generate text using OpenRouter API."""
    if not config.openrouter_api_key:
        raise ValueError("OpenRouter not configured. Set OPENROUTER_API_KEY environment variable.")
    
    model = model or config.default_model
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://uniq-mcp.research",
                "X-Title": "UniQ-MCP Quantum Circuit Synthesis"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

async def check_openrouter_health() -> dict:
    """Check OpenRouter API availability."""
    if not config.openrouter_api_key:
        return {"status": "not_configured", "error": "OPENROUTER_API_KEY not set"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {config.openrouter_api_key}"}
            )
            if response.status_code == 200:
                return {"status": "healthy", "models_available": True}
            else:
                return {"status": "error", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ============================================================================
# Circuit Synthesis Prompts
# ============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are an expert quantum computing engineer specializing in Qiskit circuit synthesis.

Your task is to generate valid Qiskit Python code that creates the requested quantum circuit.

IMPORTANT RULES:
1. Always use 'from qiskit import QuantumCircuit' at the start
2. Create a QuantumCircuit object named 'qc'
3. Add classical registers if measurements are needed
4. Use standard Qiskit gates: h(), x(), y(), z(), cx(), cz(), swap(), t(), s(), rx(), ry(), rz(), etc.
5. Include measurements with qc.measure() if the circuit should be executable
6. Return ONLY the Python code, no explanations

Example output format:
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
```
"""

def create_synthesis_prompt(description: str, constraints: Optional[Dict] = None) -> str:
    """Create a prompt for circuit synthesis."""
    prompt = f"{SYNTHESIS_SYSTEM_PROMPT}\n\nCreate a quantum circuit that: {description}"
    
    if constraints:
        if constraints.get("max_qubits"):
            prompt += f"\n- Maximum qubits: {constraints['max_qubits']}"
        if constraints.get("max_depth"):
            prompt += f"\n- Maximum circuit depth: {constraints['max_depth']}"
        if constraints.get("allowed_gates"):
            prompt += f"\n- Allowed gates: {', '.join(constraints['allowed_gates'])}"
    
    prompt += "\n\nProvide ONLY the Python code:"
    return prompt

def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to extract code block
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()
    
    # If no code block, return cleaned response
    lines = response.strip().split('\n')
    code_lines = [l for l in lines if not l.startswith('#') or 'import' in l.lower()]
    return '\n'.join(code_lines)

# ============================================================================
# Circuit Verification
# ============================================================================

def verify_circuit_code(code: str) -> Dict[str, Any]:
    """Verify that circuit code is valid and extract properties."""
    try:
        # Create isolated namespace
        namespace = {}
        exec(code, namespace)
        
        # Find the circuit
        qc = namespace.get('qc')
        if qc is None:
            for var in namespace.values():
                if hasattr(var, 'num_qubits') and hasattr(var, 'depth'):
                    qc = var
                    break
        
        if qc is None:
            return {"valid": False, "error": "No QuantumCircuit found in code"}
        
        return {
            "valid": True,
            "num_qubits": qc.num_qubits,
            "num_clbits": qc.num_clbits,
            "depth": qc.depth(),
            "gate_count": len(qc.data),
            "gates": [instr.operation.name for instr in qc.data]
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def check_server_status() -> Dict[str, Any]:
    """Check UniQ-MCP server status and available components."""
    openrouter_status = await check_openrouter_health()
    
    return {
        "status": "online",
        "version": "4.0",
        "timestamp": datetime.now().isoformat(),
        "backend": "openrouter",
        "default_model": config.default_model,
        "teacher_model": config.teacher_model,
        "openrouter": openrouter_status,
        "modules": {
            "curriculum": True,
            "hardware": True,
            "teacher": True
        }
    }

@mcp.tool()
async def synthesize_circuit(
    description: str,
    max_qubits: Optional[int] = None,
    max_depth: Optional[int] = None,
    model: Optional[str] = None,
    verify: bool = True
) -> Dict[str, Any]:
    """
    Synthesize a quantum circuit from natural language description.
    
    Args:
        description: Natural language description of the desired circuit
        max_qubits: Optional maximum number of qubits
        max_depth: Optional maximum circuit depth
        model: Optional model override (default: deepseek/deepseek-chat)
        verify: Whether to verify the generated code
    
    Returns:
        Dictionary with synthesized code and verification results
    """
    start_time = time.time()
    
    constraints = {}
    if max_qubits:
        constraints["max_qubits"] = max_qubits
    if max_depth:
        constraints["max_depth"] = max_depth
    
    prompt = create_synthesis_prompt(description, constraints if constraints else None)
    
    try:
        response = await openrouter_generate(prompt, max_tokens=1000, model=model)
        code = extract_code_from_response(response)
        
        result = {
            "success": True,
            "code": code,
            "model": model or config.default_model,
            "synthesis_time": time.time() - start_time
        }
        
        if verify:
            verification = verify_circuit_code(code)
            result["verification"] = verification
            result["valid"] = verification.get("valid", False)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "synthesis_time": time.time() - start_time
        }

@mcp.tool()
async def synthesize_with_teacher(
    description: str,
    difficulty: float = 0.5,
    use_stepping_stones: bool = True
) -> Dict[str, Any]:
    """
    Synthesize a circuit using Teacher-guided approach with stepping stones.
    
    Args:
        description: Natural language description of the target circuit
        difficulty: Difficulty level (0.0 to 1.0)
        use_stepping_stones: Whether to generate intermediate problems
    
    Returns:
        Dictionary with synthesis results and learning path
    """
    start_time = time.time()
    results = {"steps": [], "final_code": None}
    
    try:
        if use_stepping_stones and difficulty > 0.3:
            # Generate stepping stones using Teacher model
            stepping_prompt = f"""As a quantum computing teacher, break down this problem into simpler steps:

Target: {description}
Difficulty: {difficulty}

Generate 2-3 stepping stone problems that build up to the target.
Format each as: "Step N: [simpler problem description]"
"""
            stepping_response = await openrouter_generate(
                stepping_prompt, 
                model=config.teacher_model,
                max_tokens=500
            )
            
            # Parse stepping stones
            steps = []
            for line in stepping_response.split('\n'):
                if line.strip().startswith('Step'):
                    steps.append(line.strip())
            
            # Synthesize each step
            for i, step in enumerate(steps[:3]):  # Max 3 steps
                step_result = await synthesize_circuit(step, verify=True)
                results["steps"].append({
                    "step": i + 1,
                    "description": step,
                    "result": step_result
                })
        
        # Final synthesis
        final_result = await synthesize_circuit(description, verify=True)
        results["final_code"] = final_result.get("code")
        results["final_verification"] = final_result.get("verification")
        results["success"] = final_result.get("valid", False)
        results["total_time"] = time.time() - start_time
        
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_time": time.time() - start_time
        }

@mcp.tool()
async def list_benchmarks() -> Dict[str, Any]:
    """List all available curriculum benchmark problems."""
    problems = []
    categories = set()
    
    for problem in CURRICULUM_PROBLEMS:
        problems.append({
            "id": problem.id,
            "description": problem.description,
            "difficulty": problem.difficulty,
            "category": problem.category
        })
        categories.add(problem.category)
    
    return {
        "total": len(problems),
        "problems": sorted(problems, key=lambda x: x["difficulty"]),
        "categories": list(categories)
    }

@mcp.tool()
async def run_benchmark(
    benchmark_id: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a specific benchmark problem.
    
    Args:
        benchmark_id: ID of the benchmark (e.g., 'bell_state', 'ghz_3')
        model: Optional model override
    
    Returns:
        Benchmark results including success, time, and verification
    """
    # Find the benchmark
    problem = None
    for p in CURRICULUM_PROBLEMS:
        if p.id == benchmark_id:
            problem = p
            break
    
    if not problem:
        return {"success": False, "error": f"Benchmark '{benchmark_id}' not found"}
    
    start_time = time.time()
    
    # Synthesize
    result = await synthesize_circuit(
        problem.description,
        model=model,
        verify=True
    )
    
    return {
        "benchmark_id": benchmark_id,
        "description": problem.description,
        "difficulty": problem.difficulty,
        "category": problem.category,
        "success": result.get("valid", False),
        "code": result.get("code"),
        "verification": result.get("verification"),
        "time": time.time() - start_time,
        "model": model or config.default_model
    }

@mcp.tool()
async def get_curriculum_problem(
    student_level: float = 0.5,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the next curriculum problem based on student level.
    
    Args:
        student_level: Current student proficiency (0.0 to 1.0)
        category: Optional category filter
    
    Returns:
        Next recommended problem
    """
    return curriculum_manager.get_next_problem(student_level, category)

@mcp.tool()
async def record_learning_attempt(
    problem_id: str,
    success: bool,
    time_taken: float,
    code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record a learning attempt for curriculum adaptation.
    
    Args:
        problem_id: ID of the attempted problem
        success: Whether the attempt was successful
        time_taken: Time taken in seconds
        code: Optional generated code
    
    Returns:
        Updated curriculum statistics
    """
    curriculum_manager.record_attempt(problem_id, success, time_taken)
    return curriculum_manager.get_statistics()

@mcp.tool()
async def get_curriculum_statistics() -> Dict[str, Any]:
    """Get curriculum learning statistics."""
    return curriculum_manager.get_statistics()

@mcp.tool()
async def execute_on_simulator(
    circuit_code: str,
    shots: int = 1000
) -> Dict[str, Any]:
    """
    Execute a circuit on the local Qiskit simulator.
    
    Args:
        circuit_code: Python code that creates a QuantumCircuit named 'qc'
        shots: Number of measurement shots
    
    Returns:
        Execution results with measurement counts
    """
    return await hardware_manager.run_on_simulator(circuit_code, shots)

@mcp.tool()
async def execute_on_hardware(
    circuit_code: str,
    provider: str = "local_simulator",
    device: Optional[str] = None,
    shots: int = 1000
) -> Dict[str, Any]:
    """
    Execute a circuit on quantum hardware or simulator.
    
    Args:
        circuit_code: Python code that creates a QuantumCircuit named 'qc'
        provider: Hardware provider ('local_simulator', 'ionq', 'rigetti', 'iqm')
        device: Specific device ID (optional)
        shots: Number of measurement shots
    
    Returns:
        Execution results
    """
    return await hardware_manager.run_on_hardware(
        circuit_code, 
        provider=provider,
        device=device,
        shots=shots
    )

@mcp.tool()
async def get_available_hardware() -> Dict[str, Any]:
    """Get available quantum hardware status."""
    return hardware_manager.get_hardware_status()

@mcp.tool()
async def list_available_models() -> Dict[str, Any]:
    """List available OpenRouter models for synthesis."""
    recommended = [
        {
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek Chat",
            "description": "Fast, good for code generation",
            "cost": "Low"
        },
        {
            "id": "deepseek/deepseek-reasoner",
            "name": "DeepSeek R1",
            "description": "Best for complex reasoning (Teacher)",
            "cost": "Medium"
        },
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "description": "Excellent code quality",
            "cost": "Medium"
        },
        {
            "id": "openai/gpt-4-turbo",
            "name": "GPT-4 Turbo",
            "description": "High quality, reliable",
            "cost": "High"
        },
        {
            "id": "meta-llama/llama-3.1-70b-instruct",
            "name": "Llama 3.1 70B",
            "description": "Good balance of quality and cost",
            "cost": "Low"
        }
    ]
    
    return {
        "default_model": config.default_model,
        "teacher_model": config.teacher_model,
        "recommended_models": recommended,
        "note": "You can use any OpenRouter model by passing the model ID"
    }

@mcp.tool()
async def generate_latex_table(
    benchmark_ids: List[str],
    include_code: bool = False
) -> Dict[str, Any]:
    """
    Run benchmarks and generate a LaTeX table of results.
    
    Args:
        benchmark_ids: List of benchmark IDs to run
        include_code: Whether to include code snippets
    
    Returns:
        LaTeX table and raw results
    """
    results = []
    
    for bid in benchmark_ids:
        result = await run_benchmark(bid)
        results.append(result)
    
    # Generate LaTeX
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{UniQ-MCP Benchmark Results}\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
    latex += "Benchmark & Difficulty & Success & Time (s) & Gates \\\\\n\\hline\n"
    
    for r in results:
        success = "\\checkmark" if r.get("success") else "\\texttimes"
        gates = r.get("verification", {}).get("gate_count", "N/A")
        latex += f"{r['benchmark_id']} & {r['difficulty']:.2f} & {success} & {r['time']:.2f} & {gates} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\end{table}"
    
    return {
        "latex": latex,
        "results": results,
        "summary": {
            "total": len(results),
            "successful": sum(1 for r in results if r.get("success")),
            "average_time": sum(r["time"] for r in results) / len(results) if results else 0
        }
    }

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    logger.info("Starting UniQ-MCP v4 (OpenRouter Edition)")
    logger.info(f"Default model: {config.default_model}")
    logger.info(f"Teacher model: {config.teacher_model}")
    
    if not config.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set - synthesis will fail")
    
    mcp.run()

if __name__ == "__main__":
    main()
