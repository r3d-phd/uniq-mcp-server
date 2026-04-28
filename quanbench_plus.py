"""
QuanBench+ Integration for UniQ-MCP
=====================================
Implements the 42 aligned benchmark tasks from:
  "QuanBench+: A Unified Multi-Framework Benchmark for LLM-Based Quantum Code Generation"
  Slim et al., arXiv:2604.08570, ICLR 2026 Workshop

Three additions to UniQ:
  1. QUANBENCH_TASKS — 42 benchmark problems with canonical Qiskit solutions
  2. kl_divergence_accept() — KL-divergence acceptance metric (τ = 0.05)
  3. feedback_repair_loop() — Up to 5 repair attempts using runtime error traces

Task breakdown (from Appendix D):
  - Quantum Algorithms: 31 tasks
  - State Preparation:   6 tasks
  - Gate Decomposition:  5 tasks
  Total: 42 tasks
"""

import math
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger("uniq-mcp.quanbench_plus")

# ============================================================================
# KL-Divergence Acceptance (Section 3.1 of QuanBench+)
# ============================================================================

KL_THRESHOLD = 0.05  # τ = 0.05 (calibrated at 99.7th percentile of null distribution)
KL_SMOOTHING = 1e-10  # ε for additive smoothing to avoid log(0)


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute KL divergence D_KL(P || Q) between two measurement distributions.

    Args:
        p: Reference (canonical) distribution as {bitstring: probability}
        q: Generated distribution as {bitstring: probability}

    Returns:
        KL divergence value (lower is better; accepted if < KL_THRESHOLD)
    """
    # Collect all states
    all_states = set(p.keys()) | set(q.keys())

    # Apply additive smoothing and renormalize
    p_smooth = {s: p.get(s, 0.0) + KL_SMOOTHING for s in all_states}
    q_smooth = {s: q.get(s, 0.0) + KL_SMOOTHING for s in all_states}

    p_sum = sum(p_smooth.values())
    q_sum = sum(q_smooth.values())

    kl = 0.0
    for s in all_states:
        p_s = p_smooth[s] / p_sum
        q_s = q_smooth[s] / q_sum
        kl += p_s * math.log(p_s / q_s)

    return kl


def kl_divergence_accept(
    counts_generated: Dict[str, int],
    counts_canonical: Dict[str, int],
    shots: int = 1000
) -> Dict[str, Any]:
    """Check if a generated circuit's measurement distribution matches the canonical one.

    Uses the QuanBench+ KL-divergence acceptance criterion (τ = 0.05).

    Args:
        counts_generated: Raw measurement counts from the generated circuit
        counts_canonical:  Raw measurement counts from the canonical circuit
        shots: Number of shots (used for normalization)

    Returns:
        {
          "accepted": bool,
          "kl_divergence": float,
          "threshold": float,
          "method": "kl_divergence"
        }
    """
    total_gen = sum(counts_generated.values()) or shots
    total_can = sum(counts_canonical.values()) or shots

    p = {k: v / total_can for k, v in counts_canonical.items()}
    q = {k: v / total_gen for k, v in counts_generated.items()}

    kl = kl_divergence(p, q)
    accepted = kl < KL_THRESHOLD

    return {
        "accepted": accepted,
        "kl_divergence": round(kl, 6),
        "threshold": KL_THRESHOLD,
        "method": "kl_divergence",
        "interpretation": (
            f"ACCEPTED (KL={kl:.4f} < τ={KL_THRESHOLD})"
            if accepted
            else f"REJECTED (KL={kl:.4f} ≥ τ={KL_THRESHOLD})"
        )
    }


# ============================================================================
# Feedback-Based Repair Loop (Section 5.3 of QuanBench+)
# ============================================================================

REPAIR_PROMPT = """The following quantum circuit code failed.

Original task: {description}

Failed code:
```python
{failed_code}
```

Error / wrong output:
{error_message}

Please provide a corrected version. Output ONLY the corrected Python code, no explanations.
Your corrected code:"""


async def feedback_repair_loop(
    description: str,
    initial_code: str,
    initial_error: str,
    generate_fn,
    execute_fn,
    canonical_counts: Optional[Dict[str, int]] = None,
    max_repairs: int = 5,
    shots: int = 1000
) -> Dict[str, Any]:
    """Repair a failed circuit using iterative LLM feedback.

    Implements the QuanBench+ feedback loop (Section 5.3):
    - Triggered on runtime exceptions OR wrong-answer outputs
    - Up to max_repairs (default 5) repair attempts
    - Uses KL-divergence to check distributional correctness

    Args:
        description:       Original task description
        initial_code:      The code that failed on the first attempt
        initial_error:     The error message or wrong-answer description
        generate_fn:       Async function(prompt: str) -> str  (LLM call)
        execute_fn:        Async function(code: str) -> dict   (simulator execution)
        canonical_counts:  Reference measurement counts for KL acceptance (optional)
        max_repairs:       Maximum repair attempts (default 5, per QuanBench+)
        shots:             Number of shots for execution

    Returns:
        {
          "success": bool,
          "code": str,
          "repair_attempts": int,
          "pass_at_1_fb": bool,
          "kl_result": dict or None,
          "error": str or None
        }
    """
    current_code = initial_code
    current_error = initial_error

    for repair_attempt in range(1, max_repairs + 1):
        logger.info(f"Feedback repair attempt {repair_attempt}/{max_repairs}")

        # Build repair prompt
        repair_prompt = REPAIR_PROMPT.format(
            description=description,
            failed_code=current_code or "(no code generated)",
            error_message=current_error
        )

        try:
            # Ask LLM to repair
            response = await generate_fn(repair_prompt)

            # Extract code
            from server import extract_circuit_code
            repaired_code = extract_circuit_code(response)

            # Validate syntax
            try:
                exec_globals = {}
                exec(repaired_code, exec_globals)
                if "qc" not in exec_globals:
                    current_code = repaired_code
                    current_error = "No circuit 'qc' found in repaired code"
                    continue
            except Exception as syntax_err:
                current_code = repaired_code
                current_error = f"SyntaxError: {str(syntax_err)}"
                continue

            # Execute on simulator
            exec_result = await execute_fn(repaired_code, shots=shots)

            if not exec_result.get("success", False):
                current_code = repaired_code
                current_error = exec_result.get("error", "Execution failed")
                continue

            # Check distributional correctness if canonical counts provided
            kl_result = None
            if canonical_counts:
                gen_counts = exec_result.get("counts", {})
                kl_result = kl_divergence_accept(gen_counts, canonical_counts, shots)
                if not kl_result["accepted"]:
                    current_code = repaired_code
                    current_error = (
                        f"Wrong output distribution: {kl_result['interpretation']}"
                    )
                    continue

            # Success
            logger.info(f"Repair succeeded on attempt {repair_attempt}")
            return {
                "success": True,
                "code": repaired_code,
                "repair_attempts": repair_attempt,
                "pass_at_1_fb": True,
                "kl_result": kl_result,
                "error": None
            }

        except Exception as e:
            current_code = current_code
            current_error = f"Repair generation failed: {str(e)}"
            continue

    # All repair attempts exhausted
    return {
        "success": False,
        "code": current_code,
        "repair_attempts": max_repairs,
        "pass_at_1_fb": False,
        "kl_result": None,
        "error": current_error
    }


# ============================================================================
# QuanBench+ Task Definitions (42 tasks, Qiskit framework)
# ============================================================================
# Task format mirrors CURRICULUM_PROBLEMS in curriculum.py, with additions:
#   - "quanbench_id": original task number (1-44, minus tasks 5 and 38 removed)
#   - "category_qb": QuanBench+ category (quantum_algorithms / state_preparation / decomposition)
#   - "canonical_solution": reference Qiskit code
#   - "is_probabilistic": whether KL-divergence acceptance is needed
#   - "expected_distribution": canonical measurement distribution (for probabilistic tasks)

QUANBENCH_TASKS = [

    # =========================================================================
    # GATE DECOMPOSITION (5 tasks)
    # =========================================================================
    {
        "id": "qb_decomp_01", "quanbench_id": 1,
        "category": "quanbench_decomposition", "difficulty": 0.3,
        "category_qb": "decomposition",
        "description": "Decompose a SWAP gate into three CNOT gates",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)"""
    },
    {
        "id": "qb_decomp_02", "quanbench_id": 2,
        "category": "quanbench_decomposition", "difficulty": 0.35,
        "category_qb": "decomposition",
        "description": "Decompose a Toffoli (CCX) gate using single-qubit and CNOT gates",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(2)
qc.cx(1, 2)
qc.tdg(2)
qc.cx(0, 2)
qc.t(2)
qc.cx(1, 2)
qc.tdg(2)
qc.cx(0, 2)
qc.t(1)
qc.t(2)
qc.h(2)
qc.cx(0, 1)
qc.t(0)
qc.tdg(1)
qc.cx(0, 1)"""
    },
    {
        "id": "qb_decomp_03", "quanbench_id": 3,
        "category": "quanbench_decomposition", "difficulty": 0.3,
        "category_qb": "decomposition",
        "description": "Decompose a CZ gate using CNOT and Hadamard gates",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(1)
qc.cx(0, 1)
qc.h(1)"""
    },
    {
        "id": "qb_decomp_04", "quanbench_id": 4,
        "category": "quanbench_decomposition", "difficulty": 0.4,
        "category_qb": "decomposition",
        "description": "Decompose a CSWAP (Fredkin) gate using Toffoli and CNOT gates",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.cx(2, 1)
qc.ccx(0, 1, 2)
qc.cx(2, 1)"""
    },
    {
        "id": "qb_decomp_05", "quanbench_id": 6,
        "category": "quanbench_decomposition", "difficulty": 0.35,
        "category_qb": "decomposition",
        "description": "Decompose an iSWAP gate using CNOT, S, and H gates",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.s(0)
qc.s(1)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 0)
qc.h(1)"""
    },

    # =========================================================================
    # STATE PREPARATION (6 tasks)
    # =========================================================================
    {
        "id": "qb_state_01", "quanbench_id": 7,
        "category": "quanbench_state_preparation", "difficulty": 0.3,
        "category_qb": "state_preparation",
        "description": "Prepare the Bell state (|00⟩ + |11⟩)/√2",
        "is_probabilistic": True,
        "expected_distribution": {"00": 0.5, "11": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)"""
    },
    {
        "id": "qb_state_02", "quanbench_id": 8,
        "category": "quanbench_state_preparation", "difficulty": 0.4,
        "category_qb": "state_preparation",
        "description": "Prepare the 3-qubit GHZ state (|000⟩ + |111⟩)/√2",
        "is_probabilistic": True,
        "expected_distribution": {"000": 0.5, "111": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)"""
    },
    {
        "id": "qb_state_03", "quanbench_id": 9,
        "category": "quanbench_state_preparation", "difficulty": 0.55,
        "category_qb": "state_preparation",
        "description": "Prepare the 3-qubit W state (|001⟩ + |010⟩ + |100⟩)/√3",
        "is_probabilistic": True,
        "expected_distribution": {"001": 0.333, "010": 0.333, "100": 0.333},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(3)
qc.ry(2 * np.arccos(np.sqrt(2/3)), 0)
qc.ch(0, 1)
qc.cx(1, 2)
qc.cx(0, 1)
qc.x(0)"""
    },
    {
        "id": "qb_state_04", "quanbench_id": 10,
        "category": "quanbench_state_preparation", "difficulty": 0.35,
        "category_qb": "state_preparation",
        "description": "Prepare the uniform superposition state |+⟩^⊗3 on 3 qubits",
        "is_probabilistic": True,
        "expected_distribution": {
            "000": 0.125, "001": 0.125, "010": 0.125, "011": 0.125,
            "100": 0.125, "101": 0.125, "110": 0.125, "111": 0.125
        },
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)"""
    },
    {
        "id": "qb_state_05", "quanbench_id": 11,
        "category": "quanbench_state_preparation", "difficulty": 0.45,
        "category_qb": "state_preparation",
        "description": "Prepare the 4-qubit GHZ state (|0000⟩ + |1111⟩)/√2",
        "is_probabilistic": True,
        "expected_distribution": {"0000": 0.5, "1111": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.cx(0, 3)"""
    },
    {
        "id": "qb_state_06", "quanbench_id": 12,
        "category": "quanbench_state_preparation", "difficulty": 0.5,
        "category_qb": "state_preparation",
        "description": "Prepare the 2-qubit state (|00⟩ + |01⟩ + |10⟩)/√3 (non-uniform superposition)",
        "is_probabilistic": True,
        "expected_distribution": {"00": 0.333, "01": 0.333, "10": 0.333},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(2)
qc.ry(2 * np.arccos(np.sqrt(2/3)), 0)
qc.ch(0, 1)
qc.x(0)
qc.cx(0, 1)
qc.x(0)"""
    },

    # =========================================================================
    # QUANTUM ALGORITHMS (31 tasks)
    # =========================================================================
    {
        "id": "qb_algo_01", "quanbench_id": 13,
        "category": "quanbench_algorithms", "difficulty": 0.5,
        "category_qb": "quantum_algorithms",
        "description": "Implement the 2-qubit Quantum Fourier Transform (QFT)",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(2)
qc.h(0)
qc.cp(np.pi/2, 1, 0)
qc.h(1)
qc.swap(0, 1)"""
    },
    {
        "id": "qb_algo_02", "quanbench_id": 14,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement the 3-qubit Quantum Fourier Transform (QFT)",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(3)
qc.h(0)
qc.cp(np.pi/2, 1, 0)
qc.cp(np.pi/4, 2, 0)
qc.h(1)
qc.cp(np.pi/2, 2, 1)
qc.h(2)
qc.swap(0, 2)"""
    },
    {
        "id": "qb_algo_03", "quanbench_id": 15,
        "category": "quanbench_algorithms", "difficulty": 0.6,
        "category_qb": "quantum_algorithms",
        "description": "Implement Grover's search algorithm for 2 qubits targeting state |11⟩",
        "is_probabilistic": True,
        "expected_distribution": {"11": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h([0, 1])
# Oracle for |11>
qc.cz(0, 1)
# Diffusion operator
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])"""
    },
    {
        "id": "qb_algo_04", "quanbench_id": 16,
        "category": "quanbench_algorithms", "difficulty": 0.55,
        "category_qb": "quantum_algorithms",
        "description": "Implement the Deutsch-Jozsa algorithm for a constant function (f(x)=0)",
        "is_probabilistic": True,
        "expected_distribution": {"00": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.x(1)
qc.h([0, 1])
# Constant oracle: do nothing
qc.h(0)"""
    },
    {
        "id": "qb_algo_05", "quanbench_id": 17,
        "category": "quanbench_algorithms", "difficulty": 0.55,
        "category_qb": "quantum_algorithms",
        "description": "Implement the Deutsch-Jozsa algorithm for a balanced function (CNOT oracle)",
        "is_probabilistic": True,
        "expected_distribution": {"10": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.x(1)
qc.h([0, 1])
# Balanced oracle: CNOT
qc.cx(0, 1)
qc.h(0)"""
    },
    {
        "id": "qb_algo_06", "quanbench_id": 18,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement quantum phase estimation for a single-qubit phase gate with phase π/4",
        "is_probabilistic": True,
        "expected_distribution": {"10": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(2)
qc.x(1)
qc.h(0)
qc.cp(np.pi/4, 0, 1)
qc.h(0)"""
    },
    {
        "id": "qb_algo_07", "quanbench_id": 19,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement the Bernstein-Vazirani algorithm for secret string '101'",
        "is_probabilistic": True,
        "expected_distribution": {"101": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.x(3)
qc.h([0, 1, 2, 3])
# Oracle for secret='101'
qc.cx(0, 3)
qc.cx(2, 3)
qc.h([0, 1, 2])"""
    },
    {
        "id": "qb_algo_08", "quanbench_id": 20,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement Simon's algorithm oracle for period s='11'",
        "is_probabilistic": True,
        "expected_distribution": {"00": 0.5, "11": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.h([0, 1])
# Simon oracle for s=11: f(x) = f(x XOR 11)
qc.cx(0, 2)
qc.cx(1, 3)
qc.cx(0, 3)
qc.cx(1, 2)
qc.h([0, 1])"""
    },
    {
        "id": "qb_algo_09", "quanbench_id": 21,
        "category": "quanbench_algorithms", "difficulty": 0.8,
        "category_qb": "quantum_algorithms",
        "description": "Implement quantum teleportation circuit (3 qubits: sender qubit, Bell pair)",
        "is_probabilistic": True,
        "expected_distribution": {"000": 0.25, "001": 0.25, "010": 0.25, "011": 0.25},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
# Prepare Bell pair between qubits 1 and 2
qc.h(1)
qc.cx(1, 2)
# Bell measurement on qubits 0 and 1
qc.cx(0, 1)
qc.h(0)"""
    },
    {
        "id": "qb_algo_10", "quanbench_id": 22,
        "category": "quanbench_algorithms", "difficulty": 0.75,
        "category_qb": "quantum_algorithms",
        "description": "Implement the 3-qubit Grover's search targeting state |101⟩",
        "is_probabilistic": True,
        "expected_distribution": {"101": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h([0, 1, 2])
# Oracle for |101>
qc.x(1)
qc.ccx(0, 1, 2)
qc.x(1)
# Diffusion
qc.h([0, 1, 2])
qc.x([0, 1, 2])
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x([0, 1, 2])
qc.h([0, 1, 2])"""
    },
    {
        "id": "qb_algo_11", "quanbench_id": 23,
        "category": "quanbench_algorithms", "difficulty": 0.6,
        "category_qb": "quantum_algorithms",
        "description": "Implement quantum superdense coding to send classical bits '10'",
        "is_probabilistic": True,
        "expected_distribution": {"10": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
# Create Bell pair
qc.h(0)
qc.cx(0, 1)
# Encode '10' (Z gate on qubit 0)
qc.z(0)
# Decode
qc.cx(0, 1)
qc.h(0)"""
    },
    {
        "id": "qb_algo_12", "quanbench_id": 24,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement the inverse QFT on 2 qubits",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(2)
qc.swap(0, 1)
qc.h(1)
qc.cp(-np.pi/2, 1, 0)
qc.h(0)"""
    },
    {
        "id": "qb_algo_13", "quanbench_id": 25,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement quantum amplitude amplification for a 2-qubit system with oracle marking |01⟩",
        "is_probabilistic": True,
        "expected_distribution": {"01": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h([0, 1])
# Oracle for |01>
qc.x(0)
qc.cz(0, 1)
qc.x(0)
# Diffusion
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])"""
    },
    {
        "id": "qb_algo_14", "quanbench_id": 26,
        "category": "quanbench_algorithms", "difficulty": 0.75,
        "category_qb": "quantum_algorithms",
        "description": "Implement a 2-qubit VQE ansatz with parameterized Ry rotations and CNOT entanglement",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
theta0 = Parameter('θ0')
theta1 = Parameter('θ1')
qc = QuantumCircuit(2)
qc.ry(theta0, 0)
qc.ry(theta1, 1)
qc.cx(0, 1)"""
    },
    {
        "id": "qb_algo_15", "quanbench_id": 27,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum random walk on a 3-qubit line graph (one step)",
        "is_probabilistic": True,
        "expected_distribution": {"000": 0.25, "001": 0.25, "010": 0.25, "011": 0.25},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)"""
    },
    {
        "id": "qb_algo_16", "quanbench_id": 28,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement the QAOA circuit for MaxCut on a 2-node graph with one edge (1 layer)",
        "is_probabilistic": True,
        "expected_distribution": {"01": 0.5, "10": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
gamma = Parameter('γ')
beta = Parameter('β')
qc = QuantumCircuit(2)
qc.h([0, 1])
# Cost layer
qc.cx(0, 1)
qc.rz(2 * gamma, 1)
qc.cx(0, 1)
# Mixer layer
qc.rx(2 * beta, 0)
qc.rx(2 * beta, 1)"""
    },
    {
        "id": "qb_algo_17", "quanbench_id": 29,
        "category": "quanbench_algorithms", "difficulty": 0.6,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum coin flip (Hadamard on |0⟩ then measure)",
        "is_probabilistic": True,
        "expected_distribution": {"0": 0.5, "1": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.h(0)"""
    },
    {
        "id": "qb_algo_18", "quanbench_id": 30,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum phase kickback circuit for eigenstate |1⟩ of Z gate",
        "is_probabilistic": True,
        "expected_distribution": {"10": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.x(1)
qc.h(0)
qc.cz(0, 1)
qc.h(0)"""
    },
    {
        "id": "qb_algo_19", "quanbench_id": 31,
        "category": "quanbench_algorithms", "difficulty": 0.75,
        "category_qb": "quantum_algorithms",
        "description": "Implement the 4-qubit Quantum Fourier Transform",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(4)
for j in range(4):
    qc.h(j)
    for k in range(j+1, 4):
        qc.cp(np.pi / 2**(k-j), k, j)
qc.swap(0, 3)
qc.swap(1, 2)"""
    },
    {
        "id": "qb_algo_20", "quanbench_id": 32,
        "category": "quanbench_algorithms", "difficulty": 0.8,
        "category_qb": "quantum_algorithms",
        "description": "Implement Shor's algorithm period-finding circuit for N=15, a=7 (simplified)",
        "is_probabilistic": True,
        "expected_distribution": {"0000": 0.25, "0100": 0.25, "1000": 0.25, "1100": 0.25},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.h([0, 1, 2, 3])
# Simplified modular exponentiation placeholder
qc.cx(0, 1)
qc.cx(1, 2)
# Inverse QFT
import numpy as np
qc.swap(0, 3)
qc.swap(1, 2)
qc.h(3)
qc.cp(-np.pi/2, 3, 2)
qc.h(2)
qc.cp(-np.pi/4, 3, 1)
qc.cp(-np.pi/2, 2, 1)
qc.h(1)
qc.cp(-np.pi/8, 3, 0)
qc.cp(-np.pi/4, 2, 0)
qc.cp(-np.pi/2, 1, 0)
qc.h(0)"""
    },
    {
        "id": "qb_algo_21", "quanbench_id": 33,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement quantum error detection using a 3-qubit bit-flip code (encoding only)",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.cx(0, 1)
qc.cx(0, 2)"""
    },
    {
        "id": "qb_algo_22", "quanbench_id": 34,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement the HHL algorithm for a 1-qubit linear system (simplified)",
        "is_probabilistic": True,
        "expected_distribution": {"1": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(1)
qc.ry(np.pi/2, 0)"""
    },
    {
        "id": "qb_algo_23", "quanbench_id": 35,
        "category": "quanbench_algorithms", "difficulty": 0.75,
        "category_qb": "quantum_algorithms",
        "description": "Implement quantum counting (estimate number of solutions in 2-qubit Grover)",
        "is_probabilistic": True,
        "expected_distribution": {"10": 0.5, "01": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(3)
qc.h([0, 1])
qc.x(2)
qc.h(2)
# Controlled-Grover oracle
qc.ccx(0, 2, 1)
qc.h([0, 1])"""
    },
    {
        "id": "qb_algo_24", "quanbench_id": 36,
        "category": "quanbench_algorithms", "difficulty": 0.6,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum half-adder circuit (2 input qubits, 2 output qubits)",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.cx(0, 2)
qc.cx(1, 2)
qc.ccx(0, 1, 3)"""
    },
    {
        "id": "qb_algo_25", "quanbench_id": 37,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum full-adder circuit (3 inputs: a, b, carry-in)",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.ccx(0, 1, 3)
qc.cx(0, 1)
qc.ccx(1, 2, 3)
qc.cx(1, 2)
qc.cx(0, 1)"""
    },
    {
        "id": "qb_algo_26", "quanbench_id": 39,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum Fourier transform on a computational basis state |3⟩ (2 qubits)",
        "is_probabilistic": True,
        "expected_distribution": {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(2)
qc.x([0, 1])
qc.h(0)
qc.cp(np.pi/2, 1, 0)
qc.h(1)
qc.swap(0, 1)"""
    },
    {
        "id": "qb_algo_27", "quanbench_id": 40,
        "category": "quanbench_algorithms", "difficulty": 0.75,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum phase estimation for eigenphase 1/4 (2 counting qubits)",
        "is_probabilistic": True,
        "expected_distribution": {"010": 1.0},
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(3)
qc.x(2)
qc.h([0, 1])
qc.cp(2 * np.pi / 4, 0, 2)
qc.cp(2 * 2 * np.pi / 4, 1, 2)
# Inverse QFT on counting register
qc.swap(0, 1)
qc.h(1)
qc.cp(-np.pi/2, 1, 0)
qc.h(0)"""
    },
    {
        "id": "qb_algo_28", "quanbench_id": 41,
        "category": "quanbench_algorithms", "difficulty": 0.65,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum ripple-carry adder for two 1-bit numbers",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)
qc.cx(0, 1)"""
    },
    {
        "id": "qb_algo_29", "quanbench_id": 42,
        "category": "quanbench_algorithms", "difficulty": 0.8,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum approximate optimization algorithm (QAOA) for 3-node MaxCut",
        "is_probabilistic": True,
        "expected_distribution": {"010": 0.25, "101": 0.25, "001": 0.25, "110": 0.25},
        "canonical_solution": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
gamma = Parameter('γ')
beta = Parameter('β')
qc = QuantumCircuit(3)
qc.h([0, 1, 2])
# Cost layer for edges (0,1), (1,2)
for edge in [(0,1), (1,2)]:
    qc.cx(edge[0], edge[1])
    qc.rz(2*gamma, edge[1])
    qc.cx(edge[0], edge[1])
# Mixer layer
for i in range(3):
    qc.rx(2*beta, i)"""
    },
    {
        "id": "qb_algo_30", "quanbench_id": 43,
        "category": "quanbench_algorithms", "difficulty": 0.7,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum swap test to compare two single-qubit states",
        "is_probabilistic": True,
        "expected_distribution": {"000": 0.5, "100": 0.5},
        "canonical_solution": """from qiskit import QuantumCircuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cswap(0, 1, 2)
qc.h(0)"""
    },
    {
        "id": "qb_algo_31", "quanbench_id": 44,
        "category": "quanbench_algorithms", "difficulty": 0.85,
        "category_qb": "quantum_algorithms",
        "description": "Implement the quantum singular value transformation (QSVT) for a 1-qubit signal",
        "is_probabilistic": False,
        "canonical_solution": """from qiskit import QuantumCircuit
import numpy as np
qc = QuantumCircuit(2)
qc.h(0)
qc.rz(np.pi/4, 0)
qc.cx(0, 1)
qc.rz(-np.pi/4, 1)
qc.cx(0, 1)
qc.rz(np.pi/4, 0)
qc.h(0)"""
    },
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_quanbench_tasks(
    category: str = "",
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0
) -> List[Dict]:
    """Get QuanBench+ tasks filtered by category and difficulty.

    Args:
        category: Filter by category_qb: 'quantum_algorithms', 'state_preparation', 'decomposition'
        min_difficulty: Minimum difficulty (0.0-1.0)
        max_difficulty: Maximum difficulty (0.0-1.0)

    Returns:
        List of matching task dictionaries
    """
    tasks = [
        t for t in QUANBENCH_TASKS
        if min_difficulty <= t["difficulty"] <= max_difficulty
    ]
    if category:
        tasks = [t for t in tasks if t["category_qb"] == category]
    return tasks


def get_quanbench_task_by_id(task_id: str) -> Optional[Dict]:
    """Get a specific QuanBench+ task by its ID."""
    for task in QUANBENCH_TASKS:
        if task["id"] == task_id or str(task.get("quanbench_id")) == str(task_id):
            return task
    return None


def get_quanbench_statistics() -> Dict[str, Any]:
    """Get summary statistics for the QuanBench+ task set."""
    by_category = {}
    for task in QUANBENCH_TASKS:
        cat = task["category_qb"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "probabilistic": 0, "difficulties": []}
        by_category[cat]["count"] += 1
        if task["is_probabilistic"]:
            by_category[cat]["probabilistic"] += 1
        by_category[cat]["difficulties"].append(task["difficulty"])

    for cat, stats in by_category.items():
        diffs = stats["difficulties"]
        stats["mean_difficulty"] = round(sum(diffs) / len(diffs), 3)
        stats["min_difficulty"] = min(diffs)
        stats["max_difficulty"] = max(diffs)
        del stats["difficulties"]

    return {
        "total_tasks": len(QUANBENCH_TASKS),
        "by_category": by_category,
        "kl_threshold": KL_THRESHOLD,
        "max_repair_attempts": 5,
        "source": "QuanBench+ (arXiv:2604.08570)",
        "frameworks_covered": ["Qiskit"],
        "note": "Canonical solutions are Qiskit-only; PennyLane and Cirq variants pending"
    }
