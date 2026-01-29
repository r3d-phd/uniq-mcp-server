"""
UniQ-MCP Curriculum Module - RL-Based Adaptive Pacing

This module implements the curriculum learning system with reinforcement learning
based pacing. It tracks student performance and adaptively adjusts difficulty.

Key Features:
- Multi-Armed Bandit for problem selection
- Performance tracking and metrics
- Adaptive difficulty adjustment
- Grounded reward computation
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import random
import math

logger = logging.getLogger("uniq-mcp.curriculum")

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ProblemAttempt:
    """Record of a single problem attempt."""
    problem_id: str
    description: str
    difficulty: float
    success: bool
    synthesis_time: float
    attempts: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    verification_method: str = ""
    error_message: str = ""

@dataclass
class StudentState:
    """Current state of the student model."""
    capability_level: float = 0.3  # Estimated capability (0-1)
    total_attempts: int = 0
    successful_attempts: int = 0
    current_streak: int = 0
    max_streak: int = 0
    difficulty_history: List[float] = field(default_factory=list)
    performance_by_category: Dict[str, Dict] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StudentState":
        return cls(**data)

@dataclass
class CurriculumConfig:
    """Configuration for the curriculum system."""
    # Difficulty adjustment parameters
    success_threshold: float = 0.7  # Success rate to increase difficulty
    failure_threshold: float = 0.3  # Success rate to decrease difficulty
    difficulty_increment: float = 0.1
    difficulty_decrement: float = 0.05
    
    # Exploration parameters (UCB)
    exploration_constant: float = 1.4  # UCB exploration parameter
    
    # Window sizes
    performance_window: int = 10  # Number of recent attempts to consider
    streak_bonus: int = 3  # Streak length to trigger difficulty increase
    
    # Boundaries
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0

# ============================================================================
# Multi-Armed Bandit for Problem Selection
# ============================================================================

class ProblemBandit:
    """UCB-based bandit for selecting problems at appropriate difficulty."""
    
    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self.problem_stats: Dict[str, Dict] = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "total_reward": 0.0,
            "difficulty": 0.5
        })
        self.total_pulls = 0
    
    def update(self, problem_id: str, success: bool, difficulty: float):
        """Update statistics for a problem after an attempt."""
        stats = self.problem_stats[problem_id]
        stats["attempts"] += 1
        stats["difficulty"] = difficulty
        self.total_pulls += 1
        
        if success:
            stats["successes"] += 1
            # Reward is higher for harder problems
            reward = difficulty * 1.0
        else:
            # Small negative reward for failure
            reward = -0.1
        
        stats["total_reward"] += reward
    
    def get_ucb_score(self, problem_id: str) -> float:
        """Calculate UCB score for a problem."""
        stats = self.problem_stats[problem_id]
        
        if stats["attempts"] == 0:
            return float('inf')  # Unexplored problems get priority
        
        # Average reward
        avg_reward = stats["total_reward"] / stats["attempts"]
        
        # Exploration bonus
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(self.total_pulls + 1) / stats["attempts"]
        )
        
        return avg_reward + exploration
    
    def select_problem(
        self,
        available_problems: List[Dict],
        target_difficulty: float,
        tolerance: float = 0.2
    ) -> Optional[Dict]:
        """Select the best problem using UCB with difficulty filtering."""
        # Filter problems within difficulty range
        candidates = [
            p for p in available_problems
            if abs(p.get("difficulty", 0.5) - target_difficulty) <= tolerance
        ]
        
        if not candidates:
            # Fallback to closest difficulty
            candidates = sorted(
                available_problems,
                key=lambda p: abs(p.get("difficulty", 0.5) - target_difficulty)
            )[:3]
        
        if not candidates:
            return None
        
        # Calculate UCB scores
        scored = []
        for problem in candidates:
            pid = problem.get("id", problem.get("problem_id", str(hash(str(problem)))))
            score = self.get_ucb_score(pid)
            scored.append((score, problem))
        
        # Select highest scoring problem
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

# ============================================================================
# Curriculum Manager
# ============================================================================

class CurriculumManager:
    """Manages the adaptive curriculum for the student."""
    
    def __init__(
        self,
        config: CurriculumConfig = None,
        state_file: str = "./curriculum_state.json"
    ):
        self.config = config or CurriculumConfig()
        self.state_file = state_file
        self.state = self._load_state()
        self.bandit = ProblemBandit(self.config)
        self.attempt_history: List[ProblemAttempt] = []
    
    def _load_state(self) -> StudentState:
        """Load state from file if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return StudentState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return StudentState()
    
    def _save_state(self):
        """Save current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def record_attempt(
        self,
        problem_id: str,
        description: str,
        difficulty: float,
        success: bool,
        synthesis_time: float,
        attempts: int,
        category: str = "",
        error_message: str = ""
    ) -> Dict[str, Any]:
        """Record a problem attempt and update state.
        
        Returns:
            Dictionary with updated state and recommendations
        """
        # Create attempt record
        attempt = ProblemAttempt(
            problem_id=problem_id,
            description=description,
            difficulty=difficulty,
            success=success,
            synthesis_time=synthesis_time,
            attempts=attempts,
            error_message=error_message
        )
        self.attempt_history.append(attempt)
        
        # Update state
        self.state.total_attempts += 1
        if success:
            self.state.successful_attempts += 1
            self.state.current_streak += 1
            self.state.max_streak = max(self.state.max_streak, self.state.current_streak)
        else:
            self.state.current_streak = 0
        
        self.state.difficulty_history.append(difficulty)
        
        # Update category performance
        if category:
            if category not in self.state.performance_by_category:
                self.state.performance_by_category[category] = {
                    "attempts": 0,
                    "successes": 0
                }
            self.state.performance_by_category[category]["attempts"] += 1
            if success:
                self.state.performance_by_category[category]["successes"] += 1
        
        # Update bandit
        self.bandit.update(problem_id, success, difficulty)
        
        # Adjust difficulty
        new_difficulty = self._adjust_difficulty(success, difficulty)
        self.state.capability_level = new_difficulty
        self.state.last_updated = datetime.utcnow().isoformat()
        
        # Save state
        self._save_state()
        
        return {
            "recorded": True,
            "new_capability_level": new_difficulty,
            "success_rate": self.state.success_rate,
            "current_streak": self.state.current_streak,
            "recommendation": self._get_recommendation()
        }
    
    def _adjust_difficulty(self, success: bool, current_difficulty: float) -> float:
        """Adjust difficulty based on recent performance."""
        # Get recent performance
        recent = self.attempt_history[-self.config.performance_window:]
        if not recent:
            return current_difficulty
        
        recent_success_rate = sum(1 for a in recent if a.success) / len(recent)
        
        # Adjust based on performance
        if recent_success_rate >= self.config.success_threshold:
            # Doing well, increase difficulty
            new_diff = current_difficulty + self.config.difficulty_increment
        elif recent_success_rate <= self.config.failure_threshold:
            # Struggling, decrease difficulty
            new_diff = current_difficulty - self.config.difficulty_decrement
        else:
            # Maintain current difficulty
            new_diff = current_difficulty
        
        # Streak bonus
        if self.state.current_streak >= self.config.streak_bonus:
            new_diff += self.config.difficulty_increment * 0.5
        
        # Clamp to bounds
        return max(self.config.min_difficulty, min(self.config.max_difficulty, new_diff))
    
    def _get_recommendation(self) -> str:
        """Get a recommendation based on current state."""
        if self.state.current_streak >= 5:
            return "Excellent streak! Consider attempting harder problems."
        elif self.state.success_rate < 0.3 and self.state.total_attempts > 5:
            return "Consider reviewing fundamentals or reducing difficulty."
        elif self.state.success_rate > 0.8:
            return "High success rate. Ready for more challenging problems."
        else:
            return "Good progress. Continue at current difficulty."
    
    def get_next_problem(
        self,
        available_problems: List[Dict],
        category: str = ""
    ) -> Optional[Dict]:
        """Get the next problem based on curriculum.
        
        Args:
            available_problems: List of available problems
            category: Optional category filter
        
        Returns:
            Selected problem or None
        """
        # Filter by category if specified
        if category:
            problems = [p for p in available_problems if p.get("category") == category]
        else:
            problems = available_problems
        
        if not problems:
            return None
        
        # Use bandit to select
        return self.bandit.select_problem(
            problems,
            target_difficulty=self.state.capability_level
        )
    
    def get_weak_categories(self) -> List[Tuple[str, float]]:
        """Identify categories where the student is struggling.
        
        Returns:
            List of (category, success_rate) tuples, sorted by success rate
        """
        weak = []
        for category, stats in self.state.performance_by_category.items():
            if stats["attempts"] >= 3:  # Minimum attempts for meaningful data
                rate = stats["successes"] / stats["attempts"]
                if rate < 0.5:
                    weak.append((category, rate))
        
        return sorted(weak, key=lambda x: x[1])
    
    def compute_grounded_reward(
        self,
        hard_problem_id: str,
        before_performance: float,
        after_performance: float
    ) -> float:
        """Compute the grounded reward for the Teacher.
        
        The Teacher is rewarded based on Student improvement on hard problems,
        not on the quality of stepping stones themselves.
        
        Args:
            hard_problem_id: ID of the hard problem
            before_performance: Student's performance before stepping stone
            after_performance: Student's performance after stepping stone
        
        Returns:
            Reward value (positive if improved, negative if worse)
        """
        improvement = after_performance - before_performance
        
        # Scale reward
        if improvement > 0:
            # Positive reward for improvement
            reward = improvement * 2.0  # Amplify positive signal
        else:
            # Smaller penalty for no improvement
            reward = improvement * 0.5
        
        return reward
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the curriculum progress."""
        return {
            "student_state": {
                "capability_level": self.state.capability_level,
                "total_attempts": self.state.total_attempts,
                "successful_attempts": self.state.successful_attempts,
                "success_rate": self.state.success_rate,
                "current_streak": self.state.current_streak,
                "max_streak": self.state.max_streak
            },
            "performance_by_category": self.state.performance_by_category,
            "weak_categories": self.get_weak_categories(),
            "recent_difficulties": self.state.difficulty_history[-10:],
            "recommendation": self._get_recommendation(),
            "bandit_stats": {
                "total_pulls": self.bandit.total_pulls,
                "problems_explored": len(self.bandit.problem_stats)
            }
        }
    
    def reset(self):
        """Reset the curriculum state."""
        self.state = StudentState()
        self.attempt_history = []
        self.bandit = ProblemBandit(self.config)
        self._save_state()

# ============================================================================
# Curriculum Problems Database
# ============================================================================

# Extended problem set with difficulty ratings
CURRICULUM_PROBLEMS = [
    # Basic Gates (difficulty 0.1-0.2)
    {"id": "x_gate", "category": "basic_gates", "difficulty": 0.1, 
     "description": "Apply X gate to qubit 0"},
    {"id": "y_gate", "category": "basic_gates", "difficulty": 0.1,
     "description": "Apply Y gate to qubit 0"},
    {"id": "z_gate", "category": "basic_gates", "difficulty": 0.1,
     "description": "Apply Z gate to qubit 0"},
    {"id": "h_gate", "category": "basic_gates", "difficulty": 0.15,
     "description": "Apply Hadamard gate to create superposition"},
    {"id": "s_gate", "category": "basic_gates", "difficulty": 0.15,
     "description": "Apply S gate (phase gate) to qubit 0"},
    {"id": "t_gate", "category": "basic_gates", "difficulty": 0.15,
     "description": "Apply T gate to qubit 0"},
    
    # Two-Qubit Gates (difficulty 0.2-0.3)
    {"id": "cnot", "category": "two_qubit", "difficulty": 0.2,
     "description": "Apply CNOT with qubit 0 as control, qubit 1 as target"},
    {"id": "cz", "category": "two_qubit", "difficulty": 0.2,
     "description": "Apply CZ gate between qubit 0 and qubit 1"},
    {"id": "swap", "category": "two_qubit", "difficulty": 0.25,
     "description": "Swap the states of qubit 0 and qubit 1"},
    {"id": "iswap", "category": "two_qubit", "difficulty": 0.3,
     "description": "Apply iSWAP gate between qubit 0 and qubit 1"},
    
    # Entanglement (difficulty 0.3-0.5)
    {"id": "bell_state", "category": "entanglement", "difficulty": 0.3,
     "description": "Create Bell state (|00⟩ + |11⟩)/√2"},
    {"id": "bell_phi_minus", "category": "entanglement", "difficulty": 0.35,
     "description": "Create Bell state (|00⟩ - |11⟩)/√2"},
    {"id": "bell_psi_plus", "category": "entanglement", "difficulty": 0.35,
     "description": "Create Bell state (|01⟩ + |10⟩)/√2"},
    {"id": "ghz_3", "category": "entanglement", "difficulty": 0.4,
     "description": "Create 3-qubit GHZ state"},
    {"id": "ghz_4", "category": "entanglement", "difficulty": 0.5,
     "description": "Create 4-qubit GHZ state"},
    {"id": "w_state_3", "category": "entanglement", "difficulty": 0.55,
     "description": "Create 3-qubit W state"},
    
    # Algorithms (difficulty 0.5-0.8)
    {"id": "qft_2", "category": "algorithms", "difficulty": 0.5,
     "description": "Implement 2-qubit Quantum Fourier Transform"},
    {"id": "qft_3", "category": "algorithms", "difficulty": 0.65,
     "description": "Implement 3-qubit Quantum Fourier Transform"},
    {"id": "grover_2", "category": "algorithms", "difficulty": 0.6,
     "description": "Implement Grover's search for 2 qubits"},
    {"id": "grover_3", "category": "algorithms", "difficulty": 0.75,
     "description": "Implement Grover's search for 3 qubits"},
    {"id": "deutsch_jozsa", "category": "algorithms", "difficulty": 0.55,
     "description": "Implement Deutsch-Jozsa algorithm for 2 qubits"},
    
    # Advanced (difficulty 0.7-1.0)
    {"id": "toffoli", "category": "advanced", "difficulty": 0.7,
     "description": "Implement Toffoli (CCX) gate"},
    {"id": "fredkin", "category": "advanced", "difficulty": 0.75,
     "description": "Implement Fredkin (CSWAP) gate"},
    {"id": "teleportation", "category": "advanced", "difficulty": 0.8,
     "description": "Implement quantum teleportation circuit"},
    {"id": "qft_4", "category": "advanced", "difficulty": 0.85,
     "description": "Implement 4-qubit Quantum Fourier Transform"},
    {"id": "ghz_5", "category": "advanced", "difficulty": 0.9,
     "description": "Create 5-qubit GHZ state"},
    {"id": "vqe_ansatz", "category": "advanced", "difficulty": 0.95,
     "description": "Create a 2-qubit VQE ansatz with parameterized gates"},
]

def get_problems_by_difficulty(
    min_diff: float = 0.0,
    max_diff: float = 1.0,
    category: str = ""
) -> List[Dict]:
    """Get problems within a difficulty range."""
    problems = [
        p for p in CURRICULUM_PROBLEMS
        if min_diff <= p["difficulty"] <= max_diff
    ]
    if category:
        problems = [p for p in problems if p["category"] == category]
    return problems

def get_categories() -> List[str]:
    """Get all unique categories."""
    return list(set(p["category"] for p in CURRICULUM_PROBLEMS))

# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing Curriculum Module...")
    
    # Create manager
    manager = CurriculumManager()
    
    # Simulate some attempts
    test_problems = [
        ("x_gate", 0.1, True),
        ("h_gate", 0.15, True),
        ("cnot", 0.2, True),
        ("bell_state", 0.3, False),
        ("bell_state", 0.3, True),
        ("ghz_3", 0.4, False),
    ]
    
    for pid, diff, success in test_problems:
        result = manager.record_attempt(
            problem_id=pid,
            description=f"Test problem {pid}",
            difficulty=diff,
            success=success,
            synthesis_time=15.0,
            attempts=1,
            category="test"
        )
        print(f"Attempt {pid}: success={success}, new_level={result['new_capability_level']:.2f}")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
    
    # Get next problem
    next_prob = manager.get_next_problem(CURRICULUM_PROBLEMS)
    print(f"\nNext recommended problem: {next_prob}")
