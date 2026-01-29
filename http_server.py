#!/usr/bin/env python3
"""
UniQ-MCP HTTP Server Wrapper

This server exposes UniQ-MCP tools via HTTP endpoints, allowing remote access
from Manus cloud through a Cloudflare tunnel.

Usage:
    python http_server.py [--port 8001]
    
Then expose via Cloudflare:
    cloudflared tunnel --url http://localhost:8001
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

# FastAPI for HTTP server
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("uniq-mcp-http")

# ============================================================================
# Request/Response Models
# ============================================================================

class ToolRequest(BaseModel):
    """Request model for tool invocation."""
    tool: str
    arguments: Dict[str, Any] = {}

class ToolResponse(BaseModel):
    """Response model for tool invocation."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

# ============================================================================
# UniQ-MCP Tool Registry
# ============================================================================

# Import UniQ-MCP modules
try:
    from curriculum import CurriculumManager, CURRICULUM_PROBLEMS, get_problems_by_difficulty
    from teacher import TeacherClient, SteppingStoneMemory, TeacherConfig
    from multi_agent import MultiAgentCoordinator, InferenceBackend
    from quantum_hardware import QuantumHardwareManager, execute_on_quantum_hardware
    MODULES_LOADED = True
except ImportError as e:
    logger.warning(f"Some modules not loaded: {e}")
    MODULES_LOADED = False

# Global instances
curriculum_manager = None
hardware_manager = None
teacher_client = None
multi_agent = None

def init_managers():
    """Initialize global manager instances."""
    global curriculum_manager, hardware_manager, teacher_client, multi_agent
    
    if not MODULES_LOADED:
        return
    
    curriculum_manager = CurriculumManager()
    hardware_manager = QuantumHardwareManager()
    # Teacher and multi-agent are initialized on demand

# ============================================================================
# Tool Implementations
# ============================================================================

async def tool_check_server_status() -> Dict:
    """Check server status and available components."""
    return {
        "status": "online",
        "version": "3.1",
        "timestamp": datetime.utcnow().isoformat(),
        "modules": {
            "curriculum": curriculum_manager is not None,
            "hardware": hardware_manager is not None,
            "teacher": MODULES_LOADED,
            "multi_agent": MODULES_LOADED
        },
        "airlock": {
            "url": os.getenv("AIRLOCK_URL", "not_configured"),
            "configured": bool(os.getenv("AIRLOCK_URL"))
        }
    }

async def tool_list_benchmarks() -> Dict:
    """List available curriculum benchmarks."""
    if not MODULES_LOADED:
        return {"error": "Modules not loaded"}
    
    problems = []
    for p in CURRICULUM_PROBLEMS:
        problems.append({
            "id": p["id"],
            "description": p["description"],
            "difficulty": p["difficulty"],
            "category": p["category"]
        })
    
    return {
        "total": len(problems),
        "problems": problems,
        "categories": list(set(p["category"] for p in CURRICULUM_PROBLEMS))
    }

async def tool_get_curriculum_problem(category: str = None, max_difficulty: float = 1.0) -> Dict:
    """Get next curriculum problem based on student state."""
    if not curriculum_manager:
        return {"error": "Curriculum manager not initialized"}
    
    # Filter problems
    if category:
        problems = [p for p in CURRICULUM_PROBLEMS if p["category"] == category]
    else:
        problems = CURRICULUM_PROBLEMS
    
    if max_difficulty < 1.0:
        problems = [p for p in problems if p["difficulty"] <= max_difficulty]
    
    problem = curriculum_manager.get_next_problem(problems)
    return problem if problem else {"error": "No suitable problem found"}

async def tool_record_learning_attempt(
    problem_id: str,
    description: str,
    difficulty: float,
    success: bool,
    synthesis_time: float,
    attempts: int,
    category: str
) -> Dict:
    """Record a learning attempt for curriculum adaptation."""
    if not curriculum_manager:
        return {"error": "Curriculum manager not initialized"}
    
    return curriculum_manager.record_attempt(
        problem_id=problem_id,
        description=description,
        difficulty=difficulty,
        success=success,
        synthesis_time=synthesis_time,
        attempts=attempts,
        category=category
    )

async def tool_get_curriculum_statistics() -> Dict:
    """Get curriculum learning statistics."""
    if not curriculum_manager:
        return {"error": "Curriculum manager not initialized"}
    
    return curriculum_manager.get_statistics()

async def tool_execute_on_simulator(circuit_code: str, shots: int = 1000) -> Dict:
    """Execute circuit on local simulator."""
    if not hardware_manager:
        return {"error": "Hardware manager not initialized"}
    
    return await hardware_manager.run_on_hardware(
        circuit_code,
        provider="local_simulator",
        shots=shots
    )

async def tool_execute_on_hardware(
    circuit_code: str,
    provider: str = "local_simulator",
    device: str = "",
    shots: int = 1000
) -> Dict:
    """Execute circuit on quantum hardware."""
    if not hardware_manager:
        return {"error": "Hardware manager not initialized"}
    
    return await hardware_manager.run_on_hardware(
        circuit_code,
        provider=provider,
        device=device,
        shots=shots,
        fallback_to_simulator=True
    )

async def tool_get_available_hardware() -> Dict:
    """Get available quantum hardware status."""
    if not hardware_manager:
        return {"error": "Hardware manager not initialized"}
    
    return await hardware_manager.get_available_hardware()

async def tool_list_all_devices() -> Dict:
    """List all available quantum devices."""
    if not hardware_manager:
        return {"error": "Hardware manager not initialized"}
    
    return {"devices": hardware_manager.list_all_devices()}

async def tool_synthesize_circuit(description: str, max_attempts: int = 3) -> Dict:
    """Synthesize a quantum circuit from description using Airlock."""
    import httpx
    
    airlock_url = os.getenv("AIRLOCK_URL")
    airlock_key = os.getenv("AIRLOCK_API_KEY", "")
    
    if not airlock_url:
        return {"error": "AIRLOCK_URL not configured"}
    
    prompt = f"""You are a quantum computing expert. Generate a Qiskit circuit for the following:

{description}

Requirements:
- Use Qiskit 1.x syntax
- Create a QuantumCircuit named 'qc'
- Include necessary imports
- Keep the circuit efficient

Respond with ONLY the Python code, no explanations."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{airlock_url}/generate",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {airlock_key}"
                },
                json={
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                code = result.get("response", result.get("text", ""))
                
                # Clean up code
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                return {
                    "success": True,
                    "code": code.strip(),
                    "description": description
                }
            else:
                return {"error": f"Airlock returned {response.status_code}"}
                
    except Exception as e:
        return {"error": str(e)}

async def tool_generate_stepping_stone(
    target_problem: str,
    capability_level: float = 0.5,
    failure_trace: str = ""
) -> Dict:
    """Generate a stepping stone problem using Teacher."""
    global teacher_client
    
    if teacher_client is None:
        try:
            teacher_client = TeacherClient()
        except Exception as e:
            return {"error": f"Failed to initialize Teacher: {e}"}
    
    try:
        async with teacher_client as tc:
            result = await tc.generate_stepping_stone(
                target_problem=target_problem,
                capability_level=capability_level,
                failure_trace=failure_trace
            )
            return result
    except Exception as e:
        return {"error": str(e)}

# Tool registry
TOOLS = {
    "check_server_status": tool_check_server_status,
    "list_benchmarks": tool_list_benchmarks,
    "get_curriculum_problem": tool_get_curriculum_problem,
    "record_learning_attempt": tool_record_learning_attempt,
    "get_curriculum_statistics": tool_get_curriculum_statistics,
    "execute_on_simulator": tool_execute_on_simulator,
    "execute_on_hardware": tool_execute_on_hardware,
    "get_available_hardware": tool_get_available_hardware,
    "list_all_devices": tool_list_all_devices,
    "synthesize_circuit": tool_synthesize_circuit,
    "generate_stepping_stone": tool_generate_stepping_stone,
}

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="UniQ-MCP HTTP Server",
    description="HTTP interface for UniQ-MCP quantum circuit synthesis tools",
    version="3.1"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize managers on startup."""
    logger.info("Starting UniQ-MCP HTTP Server...")
    init_managers()
    logger.info("Managers initialized")

@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "name": "UniQ-MCP HTTP Server",
        "version": "3.1",
        "status": "online",
        "endpoints": {
            "/health": "Health check",
            "/tools": "List available tools",
            "/tools/{tool_name}": "Get tool info",
            "/invoke": "Invoke a tool (POST)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "modules_loaded": MODULES_LOADED
    }

@app.get("/tools")
async def list_tools():
    """List all available tools."""
    tools_info = []
    for name, func in TOOLS.items():
        tools_info.append({
            "name": name,
            "description": func.__doc__ or "No description"
        })
    return {"tools": tools_info, "count": len(tools_info)}

@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get information about a specific tool."""
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    func = TOOLS[tool_name]
    return {
        "name": tool_name,
        "description": func.__doc__ or "No description"
    }

@app.post("/invoke")
async def invoke_tool(request: ToolRequest):
    """Invoke a tool with arguments."""
    start_time = datetime.utcnow()
    
    if request.tool not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool}' not found")
    
    try:
        func = TOOLS[request.tool]
        result = await func(**request.arguments)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ToolResponse(
            success=True,
            result=result,
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Tool invocation error: {e}")
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        return ToolResponse(
            success=False,
            error=str(e),
            execution_time=execution_time
        )

# MCP-compatible endpoints
@app.post("/mcp/tools/list")
async def mcp_list_tools():
    """MCP-compatible tool listing."""
    tools = []
    for name, func in TOOLS.items():
        tools.append({
            "name": name,
            "description": func.__doc__ or "No description",
            "inputSchema": {"type": "object", "properties": {}}
        })
    return {"tools": tools}

@app.post("/mcp/tools/call")
async def mcp_call_tool(request: Request):
    """MCP-compatible tool invocation."""
    body = await request.json()
    tool_name = body.get("name")
    arguments = body.get("arguments", {})
    
    if tool_name not in TOOLS:
        return {"error": f"Tool '{tool_name}' not found"}
    
    try:
        func = TOOLS[tool_name]
        result = await func(**arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="UniQ-MCP HTTP Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              UniQ-MCP HTTP Server v3.1                       ║
╠══════════════════════════════════════════════════════════════╣
║  Starting on http://{args.host}:{args.port}                          ║
║                                                              ║
║  To expose via Cloudflare tunnel:                            ║
║  $ cloudflared tunnel --url http://localhost:{args.port}             ║
║                                                              ║
║  Then update Manus MCP config with the tunnel URL            ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
