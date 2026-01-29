"""
UniQ-MCP Multi-Agent Module - Parallel Problem Solving

This module implements multi-agent collaboration for quantum circuit synthesis.
Multiple Student instances can work on the same problem in parallel, with
results aggregated through voting or best-selection mechanisms.

Features:
- Parallel synthesis with multiple inference endpoints
- Result aggregation and voting
- Load balancing across backends
- Fallback chain for reliability
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import httpx

logger = logging.getLogger("uniq-mcp.multi_agent")

# ============================================================================
# Configuration
# ============================================================================

class BackendType(Enum):
    """Types of inference backends."""
    AIRLOCK = "airlock"  # Local GPU (Mistral 7B)
    OPENROUTER = "openrouter"  # Cloud API
    OLLAMA = "ollama"  # Local Ollama
    AZIZ = "aziz"  # Supercomputer

@dataclass
class BackendConfig:
    """Configuration for an inference backend."""
    name: str
    backend_type: BackendType
    url: str
    api_key: str = ""
    model: str = ""
    priority: int = 1  # Lower = higher priority
    max_concurrent: int = 1
    timeout: float = 120.0
    enabled: bool = True
    
    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.url)

@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system."""
    backends: List[BackendConfig] = field(default_factory=list)
    voting_threshold: float = 0.5  # Minimum agreement for consensus
    max_parallel: int = 3  # Maximum parallel synthesis attempts
    timeout: float = 180.0  # Overall timeout
    
    @classmethod
    def from_env(cls) -> "MultiAgentConfig":
        """Create config from environment variables."""
        backends = []
        
        # Airlock (local GPU)
        if os.getenv("AIRLOCK_URL"):
            backends.append(BackendConfig(
                name="airlock",
                backend_type=BackendType.AIRLOCK,
                url=os.getenv("AIRLOCK_URL", ""),
                api_key=os.getenv("AIRLOCK_API_KEY", ""),
                model="mistral-7b",
                priority=1,
                max_concurrent=1
            ))
        
        # OpenRouter (cloud)
        if os.getenv("OPENROUTER_API_KEY"):
            backends.append(BackendConfig(
                name="openrouter-deepseek",
                backend_type=BackendType.OPENROUTER,
                url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                model="deepseek/deepseek-chat",
                priority=2,
                max_concurrent=3
            ))
        
        # Ollama (local alternative)
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        backends.append(BackendConfig(
            name="ollama",
            backend_type=BackendType.OLLAMA,
            url=ollama_url,
            model="codellama",
            priority=3,
            max_concurrent=1,
            enabled=False  # Disabled by default
        ))
        
        return cls(backends=backends)

# ============================================================================
# Backend Clients
# ============================================================================

class BaseBackendClient:
    """Base class for backend clients."""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from prompt. Override in subclasses."""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        raise NotImplementedError

class AirlockClient(BaseBackendClient):
    """Client for Airlock (local GPU)."""
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = await self._client.post(
            f"{self.config.url}/generate",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            },
            json={"prompt": prompt, "max_tokens": max_tokens}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", data.get("generated_text", ""))
    
    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self.config.url}/health",
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            return response.status_code == 200
        except:
            return False

class OpenRouterClient(BaseBackendClient):
    """Client for OpenRouter API."""
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = await self._client.post(
            f"{self.config.url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://uniq-mcp.research",
                "X-Title": "UniQ-MCP"
            },
            json={
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self.config.url}/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            return response.status_code == 200
        except:
            return False

class OllamaClient(BaseBackendClient):
    """Client for local Ollama."""
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = await self._client.post(
            f"{self.config.url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    
    async def health_check(self) -> bool:
        try:
            response = await self._client.get(f"{self.config.url}/api/tags")
            return response.status_code == 200
        except:
            return False

def get_client(config: BackendConfig) -> BaseBackendClient:
    """Factory function to get appropriate client."""
    clients = {
        BackendType.AIRLOCK: AirlockClient,
        BackendType.OPENROUTER: OpenRouterClient,
        BackendType.OLLAMA: OllamaClient,
    }
    client_class = clients.get(config.backend_type, BaseBackendClient)
    return client_class(config)

# ============================================================================
# Multi-Agent Coordinator
# ============================================================================

@dataclass
class SynthesisResult:
    """Result from a single synthesis attempt."""
    backend: str
    success: bool
    code: Optional[str]
    error: Optional[str]
    time_taken: float
    verified: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "success": self.success,
            "code": self.code,
            "error": self.error,
            "time_taken": self.time_taken,
            "verified": self.verified
        }

class MultiAgentCoordinator:
    """Coordinates multiple agents for parallel synthesis."""
    
    def __init__(self, config: Optional[MultiAgentConfig] = None):
        self.config = config or MultiAgentConfig.from_env()
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Create semaphores for each backend
        for backend in self.config.backends:
            self._semaphores[backend.name] = asyncio.Semaphore(backend.max_concurrent)
    
    async def _synthesize_with_backend(
        self,
        backend: BackendConfig,
        prompt: str,
        max_tokens: int = 500
    ) -> SynthesisResult:
        """Synthesize using a specific backend."""
        import time
        start_time = time.time()
        
        if not backend.is_available:
            return SynthesisResult(
                backend=backend.name,
                success=False,
                code=None,
                error="Backend not available",
                time_taken=0
            )
        
        async with self._semaphores[backend.name]:
            try:
                client = get_client(backend)
                async with client:
                    # Health check
                    if not await client.health_check():
                        return SynthesisResult(
                            backend=backend.name,
                            success=False,
                            code=None,
                            error="Backend health check failed",
                            time_taken=time.time() - start_time
                        )
                    
                    # Generate
                    response = await client.generate(prompt, max_tokens)
                    
                    return SynthesisResult(
                        backend=backend.name,
                        success=True,
                        code=response,
                        error=None,
                        time_taken=time.time() - start_time
                    )
                    
            except asyncio.TimeoutError:
                return SynthesisResult(
                    backend=backend.name,
                    success=False,
                    code=None,
                    error="Timeout",
                    time_taken=time.time() - start_time
                )
            except Exception as e:
                return SynthesisResult(
                    backend=backend.name,
                    success=False,
                    code=None,
                    error=str(e),
                    time_taken=time.time() - start_time
                )
    
    async def parallel_synthesize(
        self,
        prompt: str,
        max_tokens: int = 500,
        num_attempts: int = 0
    ) -> Dict[str, Any]:
        """Run synthesis in parallel across multiple backends.
        
        Args:
            prompt: The synthesis prompt
            max_tokens: Maximum tokens to generate
            num_attempts: Number of parallel attempts (0 = use all available)
        
        Returns:
            Dictionary with all results and best selection
        """
        # Get available backends sorted by priority
        available = [b for b in self.config.backends if b.is_available]
        available.sort(key=lambda x: x.priority)
        
        if not available:
            return {
                "success": False,
                "error": "No backends available",
                "results": []
            }
        
        # Limit number of attempts
        if num_attempts <= 0:
            num_attempts = min(len(available), self.config.max_parallel)
        
        backends_to_use = available[:num_attempts]
        
        # Run in parallel
        tasks = [
            self._synthesize_with_backend(backend, prompt, max_tokens)
            for backend in backends_to_use
        ]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Overall timeout exceeded",
                "results": []
            }
        
        # Process results
        successful = []
        all_results = []
        
        for result in results:
            if isinstance(result, Exception):
                all_results.append(SynthesisResult(
                    backend="unknown",
                    success=False,
                    code=None,
                    error=str(result),
                    time_taken=0
                ))
            else:
                all_results.append(result)
                if result.success:
                    successful.append(result)
        
        # Select best result
        best = None
        if successful:
            # Prefer fastest successful result
            successful.sort(key=lambda x: x.time_taken)
            best = successful[0]
        
        return {
            "success": len(successful) > 0,
            "best_result": best.to_dict() if best else None,
            "num_successful": len(successful),
            "num_attempted": len(all_results),
            "results": [r.to_dict() for r in all_results]
        }
    
    async def synthesize_with_fallback(
        self,
        prompt: str,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Try backends in priority order until one succeeds.
        
        This is useful when you want reliability over speed.
        """
        available = [b for b in self.config.backends if b.is_available]
        available.sort(key=lambda x: x.priority)
        
        for backend in available:
            result = await self._synthesize_with_backend(backend, prompt, max_tokens)
            if result.success:
                return {
                    "success": True,
                    "result": result.to_dict(),
                    "backend_used": backend.name
                }
        
        return {
            "success": False,
            "error": "All backends failed",
            "backends_tried": [b.name for b in available]
        }
    
    async def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends."""
        status = {}
        
        for backend in self.config.backends:
            if not backend.is_available:
                status[backend.name] = {
                    "available": False,
                    "reason": "Not configured or disabled"
                }
                continue
            
            try:
                client = get_client(backend)
                async with client:
                    healthy = await client.health_check()
                    status[backend.name] = {
                        "available": True,
                        "healthy": healthy,
                        "type": backend.backend_type.value,
                        "model": backend.model,
                        "priority": backend.priority
                    }
            except Exception as e:
                status[backend.name] = {
                    "available": True,
                    "healthy": False,
                    "error": str(e)
                }
        
        return status

# ============================================================================
# Voting and Consensus
# ============================================================================

def compute_code_similarity(code1: str, code2: str) -> float:
    """Compute similarity between two code snippets.
    
    Uses a simple token-based Jaccard similarity.
    """
    import re
    
    def tokenize(code: str) -> set:
        # Extract meaningful tokens
        tokens = re.findall(r'\b\w+\b', code.lower())
        return set(tokens)
    
    tokens1 = tokenize(code1)
    tokens2 = tokenize(code2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0

def select_by_voting(results: List[SynthesisResult], threshold: float = 0.5) -> Optional[str]:
    """Select best code by voting on similarity.
    
    Args:
        results: List of synthesis results
        threshold: Minimum similarity to count as agreement
    
    Returns:
        Code with most agreement, or None if no consensus
    """
    successful = [r for r in results if r.success and r.code]
    
    if not successful:
        return None
    
    if len(successful) == 1:
        return successful[0].code
    
    # Count votes for each code
    votes = []
    for i, result in enumerate(successful):
        vote_count = 1  # Self-vote
        for j, other in enumerate(successful):
            if i != j:
                sim = compute_code_similarity(result.code, other.code)
                if sim >= threshold:
                    vote_count += 1
        votes.append((vote_count, result.code))
    
    # Return code with most votes
    votes.sort(key=lambda x: x[0], reverse=True)
    return votes[0][1]

# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Testing Multi-Agent Module...")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator()
        
        # Check backend status
        status = await coordinator.get_backend_status()
        print(f"Backend status: {json.dumps(status, indent=2)}")
        
        # Test synthesis (if backends available)
        prompt = """Generate Qiskit code for: Create a Bell state

Rules:
- Use variable name 'qc' for the circuit
- Qiskit 2.x only
- Output ONLY code

Your code:"""
        
        result = await coordinator.synthesize_with_fallback(prompt)
        print(f"\nSynthesis result: {json.dumps(result, indent=2)}")
    
    asyncio.run(main())
