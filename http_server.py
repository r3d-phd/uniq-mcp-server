#!/usr/bin/env python3
"""
UniQ-MCP HTTP Server with MCP Streamable HTTP Transport

This server implements the Model Context Protocol (MCP) Streamable HTTP transport,
allowing it to work as a remote MCP server accessible from Manus.
"""

import os
import sys
import json
import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, AsyncGenerator
from datetime import datetime

# FastAPI for HTTP server
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("uniq-mcp-http")

# ============================================================================
# MCP Protocol Constants
# ============================================================================

MCP_VERSION = "2025-03-26"
SERVER_NAME = "uniq-mcp"
SERVER_VERSION = "4.0"

# ============================================================================
# UniQ-MCP Tool Registry
# ============================================================================

# Import UniQ-MCP modules
try:
    from curriculum import CurriculumManager, CURRICULUM_PROBLEMS, get_problems_by_difficulty
    from teacher import TeacherClient, SteppingStoneMemory, TeacherConfig
    from multi_agent import MultiAgentCoordinator, BackendType
    from quantum_hardware import QuantumHardwareManager, execute_on_quantum_hardware
    MODULES_LOADED = True
except ImportError as e:
    logger.warning(f"Some modules not loaded: {e}")
    MODULES_LOADED = False

# Global instances
curriculum_manager = None
hardware_manager = None
teacher_client = None
sessions = {}

def init_managers():
    """Initialize global manager instances."""
    global curriculum_manager, hardware_manager
    
    if not MODULES_LOADED:
        return
    
    curriculum_manager = CurriculumManager()
    hardware_manager = QuantumHardwareManager()

# ============================================================================
# Tool Implementations
# ============================================================================

async def tool_check_server_status() -> Dict:
    """Check server status and available components."""
    return {
        "status": "online",
        "version": SERVER_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "backend": "openrouter",
        "default_model": os.getenv("UNIQ_DEFAULT_MODEL", "deepseek/deepseek-chat"),
        "teacher_model": os.getenv("UNIQ_TEACHER_MODEL", "deepseek/deepseek-reasoner"),
        "openrouter": {
            "status": "healthy" if os.getenv("OPENROUTER_API_KEY") else "not_configured",
            "models_available": bool(os.getenv("OPENROUTER_API_KEY"))
        },
        "modules": {
            "curriculum": curriculum_manager is not None,
            "hardware": hardware_manager is not None,
            "teacher": MODULES_LOADED
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

async def tool_synthesize_circuit(description: str, model: str = None, max_attempts: int = 3) -> Dict:
    """Synthesize a quantum circuit from description using OpenRouter."""
    import httpx
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not configured"}
    
    model = model or os.getenv("UNIQ_DEFAULT_MODEL", "deepseek/deepseek-chat")
    
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
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                code = result["choices"][0]["message"]["content"]
                
                # Clean up code
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                return {
                    "success": True,
                    "code": code.strip(),
                    "description": description,
                    "model": model
                }
            else:
                return {"error": f"OpenRouter returned {response.status_code}: {response.text}"}
                
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

# Tool registry with schemas
TOOLS = {
    "check_server_status": {
        "func": tool_check_server_status,
        "description": "Check server status and available components",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    "list_benchmarks": {
        "func": tool_list_benchmarks,
        "description": "List available curriculum benchmarks for quantum circuit synthesis",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    "get_curriculum_problem": {
        "func": tool_get_curriculum_problem,
        "description": "Get next curriculum problem based on student state",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Problem category filter"},
                "max_difficulty": {"type": "number", "description": "Maximum difficulty (0-1)"}
            },
            "required": []
        }
    },
    "record_learning_attempt": {
        "func": tool_record_learning_attempt,
        "description": "Record a learning attempt for curriculum adaptation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem_id": {"type": "string"},
                "description": {"type": "string"},
                "difficulty": {"type": "number"},
                "success": {"type": "boolean"},
                "synthesis_time": {"type": "number"},
                "attempts": {"type": "integer"},
                "category": {"type": "string"}
            },
            "required": ["problem_id", "description", "difficulty", "success", "synthesis_time", "attempts", "category"]
        }
    },
    "get_curriculum_statistics": {
        "func": tool_get_curriculum_statistics,
        "description": "Get curriculum learning statistics",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    "execute_on_simulator": {
        "func": tool_execute_on_simulator,
        "description": "Execute quantum circuit on local Qiskit simulator",
        "inputSchema": {
            "type": "object",
            "properties": {
                "circuit_code": {"type": "string", "description": "Python code defining the quantum circuit"},
                "shots": {"type": "integer", "description": "Number of measurement shots", "default": 1000}
            },
            "required": ["circuit_code"]
        }
    },
    "execute_on_hardware": {
        "func": tool_execute_on_hardware,
        "description": "Execute quantum circuit on real quantum hardware via AWS Braket",
        "inputSchema": {
            "type": "object",
            "properties": {
                "circuit_code": {"type": "string"},
                "provider": {"type": "string", "default": "local_simulator"},
                "device": {"type": "string", "default": ""},
                "shots": {"type": "integer", "default": 1000}
            },
            "required": ["circuit_code"]
        }
    },
    "get_available_hardware": {
        "func": tool_get_available_hardware,
        "description": "Get available quantum hardware status",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    "list_all_devices": {
        "func": tool_list_all_devices,
        "description": "List all available quantum devices",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    "synthesize_circuit": {
        "func": tool_synthesize_circuit,
        "description": "Synthesize a quantum circuit from natural language description using AI",
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Natural language description of the circuit"},
                "model": {"type": "string", "description": "AI model to use (default: deepseek/deepseek-chat)"},
                "max_attempts": {"type": "integer", "default": 3}
            },
            "required": ["description"]
        }
    },
    "generate_stepping_stone": {
        "func": tool_generate_stepping_stone,
        "description": "Generate a stepping stone problem using the Teacher module",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_problem": {"type": "string"},
                "capability_level": {"type": "number", "default": 0.5},
                "failure_trace": {"type": "string", "default": ""}
            },
            "required": ["target_problem"]
        }
    }
}

# ============================================================================
# MCP JSON-RPC Handlers
# ============================================================================

def create_jsonrpc_response(id: Any, result: Any) -> Dict:
    """Create a JSON-RPC response."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }

def create_jsonrpc_error(id: Any, code: int, message: str, data: Any = None) -> Dict:
    """Create a JSON-RPC error response."""
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": error
    }

async def handle_initialize(params: Dict) -> Dict:
    """Handle MCP initialize request."""
    return {
        "protocolVersion": MCP_VERSION,
        "capabilities": {
            "tools": {"listChanged": False}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    }

async def handle_tools_list(params: Dict) -> Dict:
    """Handle tools/list request."""
    tools = []
    for name, tool_info in TOOLS.items():
        tools.append({
            "name": name,
            "description": tool_info["description"],
            "inputSchema": tool_info["inputSchema"]
        })
    return {"tools": tools}

async def handle_tools_call(params: Dict) -> Dict:
    """Handle tools/call request."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    func = TOOLS[tool_name]["func"]
    result = await func(**arguments)
    
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, default=str, indent=2)
            }
        ]
    }

async def process_jsonrpc_request(request: Dict) -> Dict:
    """Process a single JSON-RPC request."""
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")
    
    try:
        if method == "initialize":
            result = await handle_initialize(params)
        elif method == "initialized":
            # Notification, no response needed
            return None
        elif method == "tools/list":
            result = await handle_tools_list(params)
        elif method == "tools/call":
            result = await handle_tools_call(params)
        elif method == "ping":
            result = {}
        else:
            return create_jsonrpc_error(req_id, -32601, f"Method not found: {method}")
        
        if req_id is not None:
            return create_jsonrpc_response(req_id, result)
        return None
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        if req_id is not None:
            return create_jsonrpc_error(req_id, -32603, str(e))
        return None

# ============================================================================
# FastAPI Application with MCP Streamable HTTP Transport
# ============================================================================

app = FastAPI(
    title="UniQ-MCP Server",
    description="MCP server for quantum circuit synthesis with Streamable HTTP transport",
    version=SERVER_VERSION
)

# CORS middleware
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
    logger.info("Starting UniQ-MCP Server...")
    init_managers()
    logger.info("Managers initialized")

# ============================================================================
# MCP Endpoint (Streamable HTTP Transport)
# ============================================================================

@app.get("/mcp")
async def mcp_get(request: Request):
    """
    MCP GET endpoint for SSE stream.
    Used by clients to listen for server-initiated messages.
    """
    accept = request.headers.get("accept", "")
    
    if "text/event-stream" not in accept:
        return Response(status_code=405)
    
    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        # Keep connection alive with periodic pings
        while True:
            await asyncio.sleep(30)
            yield f"event: ping\ndata: {{}}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/mcp")
async def mcp_post(request: Request):
    """
    MCP POST endpoint for JSON-RPC messages.
    Implements the Streamable HTTP transport.
    """
    accept = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    
    try:
        body = await request.json()
    except:
        return JSONResponse(
            status_code=400,
            content=create_jsonrpc_error(None, -32700, "Parse error")
        )
    
    # Handle batch requests
    if isinstance(body, list):
        responses = []
        for req in body:
            resp = await process_jsonrpc_request(req)
            if resp:
                responses.append(resp)
        
        if not responses:
            return Response(status_code=202)
        
        return JSONResponse(content=responses if len(responses) > 1 else responses[0])
    
    # Handle single request
    response = await process_jsonrpc_request(body)
    
    if response is None:
        return Response(status_code=202)
    
    # Check if client accepts SSE
    if "text/event-stream" in accept and "application/json" in accept:
        # Return as SSE stream for compatibility
        async def single_event() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps(response)}\n\n"
        
        return StreamingResponse(
            single_event(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    return JSONResponse(content=response)

# ============================================================================
# Legacy HTTP Endpoints (for backwards compatibility)
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "name": "UniQ-MCP Server",
        "version": SERVER_VERSION,
        "protocol": "MCP Streamable HTTP",
        "mcp_version": MCP_VERSION,
        "status": "online",
        "endpoints": {
            "/mcp": "MCP Streamable HTTP endpoint (POST/GET)",
            "/health": "Health check",
            "/tools": "List available tools (legacy)",
            "/invoke": "Invoke a tool (legacy POST)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "modules_loaded": MODULES_LOADED,
        "mcp_version": MCP_VERSION
    }

@app.get("/tools")
async def list_tools():
    """List all available tools (legacy endpoint)."""
    tools_info = []
    for name, tool_info in TOOLS.items():
        tools_info.append({
            "name": name,
            "description": tool_info["description"]
        })
    return {"tools": tools_info, "count": len(tools_info)}

class LegacyToolRequest(BaseModel):
    """Request model for legacy tool invocation."""
    tool: str
    arguments: Dict[str, Any] = {}

@app.post("/invoke")
async def invoke_tool(request: LegacyToolRequest):
    """Invoke a tool with arguments (legacy endpoint)."""
    start_time = datetime.utcnow()
    
    if request.tool not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool}' not found")
    
    try:
        func = TOOLS[request.tool]["func"]
        result = await func(**request.arguments)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "success": True,
            "result": result,
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Tool invocation error: {e}")
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        return {
            "success": False,
            "error": str(e),
            "execution_time": execution_time
        }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              UniQ-MCP Server v{SERVER_VERSION}                          ║
║         MCP Streamable HTTP Transport                        ║
╠══════════════════════════════════════════════════════════════╣
║  MCP Endpoint: http://{host}:{port}/mcp                        ║
║  Health Check: http://{host}:{port}/health                     ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
