#!/usr/bin/env python3
"""
UniQ-MCP Phase 3 Test Suite

Tests for SOAR framework components:
- Teacher module (stepping stone generation)
- Curriculum module (RL-based pacing)
- Multi-agent module (parallel synthesis)
- Quantum hardware module (execution)
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Test results tracking
results = {
    "timestamp": datetime.utcnow().isoformat(),
    "tests": [],
    "passed": 0,
    "failed": 0,
    "skipped": 0
}

def log_test(name: str, passed: bool, details: str = "", skipped: bool = False):
    """Log a test result."""
    status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
    emoji = "⏭️" if skipped else ("✅" if passed else "❌")
    print(f"{emoji} {name}: {status}")
    if details:
        print(f"   {details}")
    
    results["tests"].append({
        "name": name,
        "status": status,
        "details": details
    })
    
    if skipped:
        results["skipped"] += 1
    elif passed:
        results["passed"] += 1
    else:
        results["failed"] += 1

# ============================================================================
# Curriculum Module Tests
# ============================================================================

def test_curriculum_manager():
    """Test CurriculumManager functionality."""
    print("\n📚 Testing Curriculum Module...")
    
    try:
        from curriculum import CurriculumManager, CURRICULUM_PROBLEMS, get_problems_by_difficulty
        
        # Test initialization
        manager = CurriculumManager(state_file="./test_curriculum_state.json")
        log_test("CurriculumManager initialization", True)
        
        # Test problem filtering
        easy_problems = get_problems_by_difficulty(0.0, 0.3)
        hard_problems = get_problems_by_difficulty(0.7, 1.0)
        log_test("Problem difficulty filtering", 
                 len(easy_problems) > 0 and len(hard_problems) > 0,
                 f"Easy: {len(easy_problems)}, Hard: {len(hard_problems)}")
        
        # Test recording attempts
        result = manager.record_attempt(
            problem_id="test_bell",
            description="Create Bell state",
            difficulty=0.3,
            success=True,
            synthesis_time=15.0,
            attempts=1,
            category="entanglement"
        )
        log_test("Record learning attempt", 
                 result.get("recorded") == True,
                 f"New capability: {result.get('new_capability_level', 0):.2f}")
        
        # Test statistics
        stats = manager.get_statistics()
        log_test("Get curriculum statistics",
                 "student_state" in stats and "recommendation" in stats)
        
        # Test problem selection
        next_prob = manager.get_next_problem(CURRICULUM_PROBLEMS)
        log_test("Get next curriculum problem",
                 next_prob is not None,
                 f"Selected: {next_prob.get('id', 'None') if next_prob else 'None'}")
        
        # Cleanup
        import os
        if os.path.exists("./test_curriculum_state.json"):
            os.remove("./test_curriculum_state.json")
        
    except Exception as e:
        log_test("Curriculum module import", False, str(e))

# ============================================================================
# Multi-Agent Module Tests
# ============================================================================

def test_multi_agent():
    """Test MultiAgentCoordinator functionality."""
    print("\n🤖 Testing Multi-Agent Module...")
    
    try:
        from multi_agent import (
            MultiAgentCoordinator, MultiAgentConfig, BackendConfig, BackendType,
            compute_code_similarity, select_by_voting, SynthesisResult
        )
        
        # Test initialization
        coordinator = MultiAgentCoordinator()
        log_test("MultiAgentCoordinator initialization", True)
        
        # Test code similarity
        code1 = "qc.h(0)\nqc.cx(0, 1)"
        code2 = "qc.h(0)\nqc.cx(0, 1)"
        code3 = "qc.x(0)\nqc.y(1)"
        
        sim_same = compute_code_similarity(code1, code2)
        sim_diff = compute_code_similarity(code1, code3)
        log_test("Code similarity computation",
                 sim_same > sim_diff,
                 f"Same: {sim_same:.2f}, Different: {sim_diff:.2f}")
        
        # Test voting
        results = [
            SynthesisResult("a", True, "qc.h(0)\nqc.cx(0,1)", None, 1.0),
            SynthesisResult("b", True, "qc.h(0)\nqc.cx(0,1)", None, 2.0),
            SynthesisResult("c", True, "qc.x(0)", None, 1.5),
        ]
        best = select_by_voting(results)
        log_test("Voting selection",
                 best is not None and "h(0)" in best,
                 f"Selected code with majority vote")
        
    except Exception as e:
        log_test("Multi-agent module import", False, str(e))

# ============================================================================
# Quantum Hardware Module Tests
# ============================================================================

async def test_quantum_hardware():
    """Test QuantumHardwareManager functionality."""
    print("\n⚛️ Testing Quantum Hardware Module...")
    
    try:
        from quantum_hardware import (
            QuantumHardwareManager, HardwareProvider, SimulatorClient, HardwareConfig
        )
        
        # Test initialization
        manager = QuantumHardwareManager()
        log_test("QuantumHardwareManager initialization", True)
        
        # Test simulator
        test_code = """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
        result = await manager.run_on_hardware(
            test_code,
            HardwareProvider.SIMULATOR,
            shots=100
        )
        log_test("Simulator execution",
                 result.get("success") == True,
                 f"Counts: {result.get('counts', {})}")
        
        # Test hardware status
        status = await manager.get_available_hardware()
        log_test("Hardware availability check",
                 "simulator" in status,
                 f"Simulator: {status.get('simulator', {}).get('status')}")
        
        # Note about pending AWS Braket
        log_test("AWS Braket (IonQ)", False, 
                 "Pending activation - will auto-enable when ready", skipped=True)
        
    except Exception as e:
        log_test("Quantum hardware module import", False, str(e))

# ============================================================================
# Teacher Module Tests
# ============================================================================

async def test_teacher():
    """Test TeacherClient functionality."""
    print("\n👨‍🏫 Testing Teacher Module...")
    
    try:
        from teacher import (
            TeacherClient, TeacherConfig, SteppingStoneMemory, test_teacher_connection
        )
        
        # Test initialization
        config = TeacherConfig.from_env()
        log_test("TeacherConfig initialization", True)
        
        # Test memory
        memory = SteppingStoneMemory(persist_directory="./test_chroma_data")
        doc_id = memory.store_stepping_stone(
            problem_desc="Test Bell state",
            circuit_qasm="qc.h(0); qc.cx(0,1)",
            difficulty=0.3,
            verified=True
        )
        log_test("SteppingStoneMemory store", 
                 doc_id is not None,
                 f"Stored with ID: {doc_id[:8]}...")
        
        # Test retrieval
        similar = memory.find_similar_problems("Bell state", n_results=1)
        log_test("SteppingStoneMemory retrieval",
                 len(similar) > 0,
                 f"Found {len(similar)} similar problems")
        
        # Test Teacher connection (if configured)
        if config.openrouter_api_key:
            connection = await test_teacher_connection()
            log_test("Teacher API connection",
                     connection.get("success") == True,
                     f"Model: {connection.get('model', 'N/A')}")
        else:
            log_test("Teacher API connection", False,
                     "OPENROUTER_API_KEY not set", skipped=True)
        
        # Cleanup
        import shutil
        if os.path.exists("./test_chroma_data"):
            shutil.rmtree("./test_chroma_data")
        
    except Exception as e:
        log_test("Teacher module import", False, str(e))

# ============================================================================
# Integration Tests
# ============================================================================

async def test_integration():
    """Test integration between modules."""
    print("\n🔗 Testing Module Integration...")
    
    try:
        from curriculum import CurriculumManager, CURRICULUM_PROBLEMS
        from teacher import SteppingStoneMemory
        
        # Test curriculum + memory integration
        manager = CurriculumManager(state_file="./test_integration_state.json")
        memory = SteppingStoneMemory(persist_directory="./test_integration_chroma")
        
        # Simulate learning loop
        for problem in CURRICULUM_PROBLEMS[:3]:
            # Record attempt
            manager.record_attempt(
                problem_id=problem["id"],
                description=problem["description"],
                difficulty=problem["difficulty"],
                success=True,
                synthesis_time=10.0,
                attempts=1,
                category=problem["category"]
            )
            
            # Store in memory
            memory.store_stepping_stone(
                problem_desc=problem["description"],
                circuit_qasm=f"# Solution for {problem['id']}",
                difficulty=problem["difficulty"],
                verified=True
            )
        
        # Check curriculum adapted
        stats = manager.get_statistics()
        log_test("Curriculum adaptation",
                 stats["student_state"]["total_attempts"] == 3,
                 f"Capability: {stats['student_state']['capability_level']:.2f}")
        
        # Check memory populated
        results = memory.find_similar_problems("quantum circuit", n_results=5)
        log_test("Memory population",
                 len(results) >= 3,
                 f"Stored {len(results)} problems")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_integration_state.json"):
            os.remove("./test_integration_state.json")
        if os.path.exists("./test_integration_chroma"):
            shutil.rmtree("./test_integration_chroma")
        
    except Exception as e:
        log_test("Integration test", False, str(e))

# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all tests."""
    print("=" * 60)
    print("UniQ-MCP Phase 3 Test Suite")
    print("=" * 60)
    
    # Run tests
    test_curriculum_manager()
    test_multi_agent()
    await test_quantum_hardware()
    await test_teacher()
    await test_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"Total:   {total}")
    print(f"Passed:  {results['passed']} ✅")
    print(f"Failed:  {results['failed']} ❌")
    print(f"Skipped: {results['skipped']} ⏭️")
    print("=" * 60)
    
    # Save results
    with open("test_phase3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to test_phase3_results.json")
    
    # Return exit code
    return 0 if results["failed"] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
