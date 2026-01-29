"""
UniQ-MCP Teacher Module - SOAR Stepping Stone Generation

This module implements the Teacher component of the SOAR (Self-Optimization via
Asymmetric RL) framework. The Teacher generates "stepping stone" problems that
bridge the gap between the Student's current capabilities and hard target problems.

Architecture:
- Teacher: DeepSeek-R1 (70B) via OpenRouter API for curriculum generation
- Student: Mistral 7B via Airlock for execution (in server.py)
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import httpx

logger = logging.getLogger("uniq-mcp.teacher")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TeacherConfig:
    """Configuration for the Teacher module."""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    teacher_model: str = "deepseek/deepseek-r1"  # DeepSeek-R1 for reasoning
    fallback_model: str = "deepseek/deepseek-chat"  # Cheaper fallback
    max_tokens: int = 2000
    temperature: float = 0.7
    
    @classmethod
    def from_env(cls) -> "TeacherConfig":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )

# ============================================================================
# Teacher System Prompts
# ============================================================================

TEACHER_SYSTEM_PROMPT = """You are the SOAR Teacher for a Quantum Logic Synthesis agent. Your goal is NOT to solve the problem directly, but to generate a 'Stepping Stone' problem that bridges the gap between the Student's current capabilities and the target Hard Problem.

Your responsibilities:
1. Analyze the target problem and identify its core challenges
2. Generate a simplified version that teaches the key concepts
3. Provide a reference solution in OpenQASM/Qiskit format
4. Estimate the difficulty level (0.0 to 1.0)

Rules for Stepping Stones:
- Reduce qubit count (e.g., 5 qubits → 2-3 qubits)
- Relax depth constraints
- Focus on a specific sub-component
- Maintain the essential learning objective

Output Format (JSON):
{
    "stepping_stone": {
        "description": "Natural language description of the simplified problem",
        "num_qubits": <int>,
        "estimated_difficulty": <float 0-1>,
        "learning_objective": "What concept this teaches",
        "reference_code": "Qiskit code for the solution"
    },
    "reasoning": "Why this stepping stone helps bridge to the target"
}"""

STEPPING_STONE_PROMPT = """Target Problem: {target_problem}

Student's Current Capability Level: {capability_level}

Student's Previous Failure (if any): {failure_trace}

Generate a stepping stone problem that will help the Student progress toward solving the target problem. The stepping stone should be achievable at the Student's current level while teaching concepts needed for the target."""

# ============================================================================
# Teacher Client
# ============================================================================

class TeacherClient:
    """Client for interacting with the Teacher model via OpenRouter."""
    
    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=180.0)  # 3 min timeout for reasoning
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def _call_openrouter(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Make a call to OpenRouter API."""
        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        model = model or self.config.teacher_model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://uniq-mcp.research",  # Required by OpenRouter
            "X-Title": "UniQ-MCP Teacher"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = await self._client.post(
            f"{self.config.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def generate_stepping_stone(
        self,
        target_problem: str,
        capability_level: float = 0.5,
        failure_trace: str = ""
    ) -> Dict[str, Any]:
        """Generate a stepping stone problem for the given target.
        
        Args:
            target_problem: Description of the hard target problem
            capability_level: Student's current capability (0.0 to 1.0)
            failure_trace: Previous failure information (if any)
        
        Returns:
            Dictionary containing the stepping stone problem and metadata
        """
        user_prompt = STEPPING_STONE_PROMPT.format(
            target_problem=target_problem,
            capability_level=capability_level,
            failure_trace=failure_trace or "No previous failures recorded"
        )
        
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._call_openrouter(messages)
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            result = json.loads(json_str.strip())
            result["generated_at"] = datetime.utcnow().isoformat()
            result["target_problem"] = target_problem
            result["capability_level"] = capability_level
            
            return {
                "success": True,
                "stepping_stone": result.get("stepping_stone", {}),
                "reasoning": result.get("reasoning", ""),
                "metadata": {
                    "model": self.config.teacher_model,
                    "generated_at": result["generated_at"]
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Teacher response as JSON: {e}")
            return {
                "success": False,
                "error": f"JSON parse error: {str(e)}",
                "raw_response": response if 'response' in dir() else None
            }
        except Exception as e:
            logger.error(f"Teacher generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_curriculum(
        self,
        target_problem: str,
        num_steps: int = 3,
        start_difficulty: float = 0.2
    ) -> Dict[str, Any]:
        """Generate a complete curriculum of stepping stones.
        
        Args:
            target_problem: The final target problem
            num_steps: Number of stepping stones to generate
            start_difficulty: Starting difficulty level
        
        Returns:
            Dictionary containing the curriculum
        """
        curriculum = []
        current_difficulty = start_difficulty
        difficulty_increment = (1.0 - start_difficulty) / (num_steps + 1)
        
        for i in range(num_steps):
            step = await self.generate_stepping_stone(
                target_problem=target_problem,
                capability_level=current_difficulty,
                failure_trace=""
            )
            
            if step["success"]:
                step["step_number"] = i + 1
                step["target_difficulty"] = current_difficulty
                curriculum.append(step)
                current_difficulty += difficulty_increment
            else:
                logger.warning(f"Failed to generate step {i+1}: {step.get('error')}")
        
        return {
            "success": len(curriculum) > 0,
            "curriculum": curriculum,
            "target_problem": target_problem,
            "num_steps_requested": num_steps,
            "num_steps_generated": len(curriculum)
        }
    
    async def evaluate_student_solution(
        self,
        problem_description: str,
        student_code: str,
        reference_code: str
    ) -> Dict[str, Any]:
        """Evaluate a student's solution and provide feedback.
        
        Args:
            problem_description: The problem that was attempted
            student_code: The student's generated code
            reference_code: The reference solution
        
        Returns:
            Dictionary with evaluation results and feedback
        """
        eval_prompt = f"""Evaluate this quantum circuit solution:

Problem: {problem_description}

Student's Code:
```python
{student_code}
```

Reference Solution:
```python
{reference_code}
```

Provide:
1. Correctness assessment (correct/incorrect/partial)
2. Specific feedback on what's wrong (if anything)
3. Hints for improvement (without giving the answer)
4. Estimated mastery level (0.0 to 1.0)

Output as JSON:
{{
    "correctness": "correct|incorrect|partial",
    "feedback": "Specific feedback",
    "hints": ["hint1", "hint2"],
    "mastery_level": <float>,
    "reasoning": "Why this assessment"
}}"""

        messages = [
            {"role": "system", "content": "You are a quantum computing instructor evaluating student work."},
            {"role": "user", "content": eval_prompt}
        ]
        
        try:
            response = await self._call_openrouter(
                messages,
                model=self.config.fallback_model,  # Use cheaper model for evaluation
                max_tokens=1000
            )
            
            # Parse JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            result = json.loads(json_str.strip())
            return {
                "success": True,
                "evaluation": result
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# Stepping Stone Database (ChromaDB Integration)
# ============================================================================

class SteppingStoneMemory:
    """Manages stepping stone storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
    
    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB."""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name="quantum_reasoning_traces",
                metadata={"description": "UniQ-MCP stepping stones and solutions"}
            )
    
    def store_stepping_stone(
        self,
        problem_desc: str,
        circuit_qasm: str,
        difficulty: float,
        verified: bool,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store a stepping stone in the database.
        
        Returns:
            The ID of the stored document
        """
        self._ensure_initialized()
        import uuid
        
        doc_id = str(uuid.uuid4())
        
        # Combine problem and circuit for embedding
        document = f"Problem: {problem_desc}\nSolution: {circuit_qasm}"
        
        meta = {
            "problem_desc": problem_desc,
            "difficulty": difficulty,
            "stepping_stone": True,
            "verified": verified,
            "timestamp": datetime.utcnow().isoformat()
        }
        if parent_id:
            meta["parent_id"] = parent_id
        if metadata:
            meta.update(metadata)
        
        self._collection.add(
            documents=[document],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        return doc_id
    
    def find_similar_problems(
        self,
        query: str,
        n_results: int = 5,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0
    ) -> List[Dict]:
        """Find similar problems from memory.
        
        Args:
            query: Problem description to search for
            n_results: Number of results to return
            min_difficulty: Minimum difficulty filter
            max_difficulty: Maximum difficulty filter
        
        Returns:
            List of similar problems with their solutions
        """
        self._ensure_initialized()
        
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "$and": [
                    {"difficulty": {"$gte": min_difficulty}},
                    {"difficulty": {"$lte": max_difficulty}}
                ]
            }
        )
        
        problems = []
        for i, doc_id in enumerate(results["ids"][0]):
            problems.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None
            })
        
        return problems
    
    def get_curriculum_path(
        self,
        target_difficulty: float,
        current_difficulty: float = 0.1
    ) -> List[Dict]:
        """Get a curriculum path from current to target difficulty.
        
        Args:
            target_difficulty: Target difficulty level
            current_difficulty: Current capability level
        
        Returns:
            Ordered list of problems forming a curriculum
        """
        self._ensure_initialized()
        
        # Get all verified stepping stones
        results = self._collection.query(
            query_texts=["quantum circuit"],  # Generic query
            n_results=50,
            where={
                "$and": [
                    {"verified": True},
                    {"difficulty": {"$gte": current_difficulty}},
                    {"difficulty": {"$lte": target_difficulty}}
                ]
            }
        )
        
        # Sort by difficulty
        problems = []
        for i, doc_id in enumerate(results["ids"][0]):
            problems.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        problems.sort(key=lambda x: x["metadata"].get("difficulty", 0))
        return problems
    
    def prune_duplicates(self, similarity_threshold: float = 0.95):
        """Remove duplicate or very similar entries.
        
        This is part of the "Dreaming" layer garbage collection.
        """
        self._ensure_initialized()
        
        # Get all documents
        all_docs = self._collection.get()
        
        if not all_docs["ids"]:
            return 0
        
        # Find clusters of similar documents
        to_delete = set()
        
        for i, doc_id in enumerate(all_docs["ids"]):
            if doc_id in to_delete:
                continue
            
            # Query for similar documents
            similar = self._collection.query(
                query_texts=[all_docs["documents"][i]],
                n_results=10
            )
            
            # Mark duplicates (keep the first one)
            for j, sim_id in enumerate(similar["ids"][0][1:], 1):  # Skip self
                if similar["distances"] and similar["distances"][0][j] < (1 - similarity_threshold):
                    to_delete.add(sim_id)
        
        # Delete duplicates
        if to_delete:
            self._collection.delete(ids=list(to_delete))
        
        return len(to_delete)

# ============================================================================
# Utility Functions
# ============================================================================

async def test_teacher_connection() -> Dict[str, Any]:
    """Test the connection to the Teacher model."""
    config = TeacherConfig.from_env()
    
    if not config.openrouter_api_key:
        return {
            "success": False,
            "error": "OPENROUTER_API_KEY not configured"
        }
    
    async with TeacherClient(config) as teacher:
        try:
            # Simple test query
            messages = [
                {"role": "user", "content": "Say 'Teacher ready' if you can read this."}
            ]
            response = await teacher._call_openrouter(
                messages,
                model=config.fallback_model,
                max_tokens=50
            )
            
            return {
                "success": True,
                "model": config.teacher_model,
                "fallback_model": config.fallback_model,
                "response": response[:100]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Testing Teacher Module...")
        
        # Test connection
        result = await test_teacher_connection()
        print(f"Connection test: {result}")
        
        if result["success"]:
            # Test stepping stone generation
            async with TeacherClient() as teacher:
                stepping_stone = await teacher.generate_stepping_stone(
                    target_problem="Create a 5-qubit GHZ state with depth less than 10",
                    capability_level=0.3,
                    failure_trace="Student failed to handle more than 3 qubits"
                )
                print(f"\nStepping stone: {json.dumps(stepping_stone, indent=2)}")
    
    asyncio.run(main())
