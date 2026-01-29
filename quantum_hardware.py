"""
UniQ-MCP Quantum Hardware Module - Real Quantum Device Integration

This module provides integration with real quantum hardware providers:
- IBM Quantum (via Qiskit Runtime)
- IonQ (via Amazon Braket or direct API)
- Simulator fallback for testing

The module handles:
- Device selection and availability checking
- Job submission and result retrieval
- Automatic fallback to simulators
- Result caching and error handling
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import subprocess

logger = logging.getLogger("uniq-mcp.quantum_hardware")

# ============================================================================
# Configuration
# ============================================================================

class HardwareProvider(Enum):
    """Supported quantum hardware providers."""
    IBM_QUANTUM = "ibm_quantum"
    IONQ = "ionq"
    AWS_BRAKET = "aws_braket"
    SIMULATOR = "simulator"

@dataclass
class HardwareConfig:
    """Configuration for quantum hardware access."""
    ibm_token: str = ""
    ibm_channel: str = "ibm_quantum"  # or "ibm_cloud"
    ionq_api_key: str = ""
    aws_region: str = "us-east-1"
    default_shots: int = 1000
    max_qubits: int = 5  # Limit for free tier
    timeout: float = 300.0  # 5 minutes for quantum jobs
    
    @classmethod
    def from_env(cls) -> "HardwareConfig":
        return cls(
            ibm_token=os.getenv("IBM_QUANTUM_TOKEN", ""),
            ibm_channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
            ionq_api_key=os.getenv("IONQ_API_KEY", ""),
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )

# ============================================================================
# IBM Quantum Integration
# ============================================================================

class IBMQuantumClient:
    """Client for IBM Quantum hardware."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self._service = None
        self._available_backends = []
    
    def _ensure_initialized(self):
        """Lazy initialization of IBM Quantum service."""
        if self._service is not None:
            return
        
        if not self.config.ibm_token:
            raise ValueError("IBM_QUANTUM_TOKEN not configured")
        
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            # Try to use saved credentials first
            try:
                self._service = QiskitRuntimeService()
            except:
                # Save and use new credentials
                QiskitRuntimeService.save_account(
                    channel=self.config.ibm_channel,
                    token=self.config.ibm_token,
                    overwrite=True
                )
                self._service = QiskitRuntimeService()
            
            # Get available backends
            self._available_backends = self._service.backends()
            
        except ImportError:
            raise ImportError("qiskit-ibm-runtime not installed. Run: pip install qiskit-ibm-runtime")
    
    def get_available_backends(self) -> List[Dict]:
        """Get list of available IBM Quantum backends."""
        self._ensure_initialized()
        
        backends = []
        for backend in self._available_backends:
            try:
                config = backend.configuration()
                status = backend.status()
                backends.append({
                    "name": backend.name,
                    "num_qubits": config.n_qubits,
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "simulator": config.simulator
                })
            except:
                continue
        
        return backends
    
    def get_least_busy_backend(self, min_qubits: int = 2) -> Optional[str]:
        """Get the least busy backend with sufficient qubits."""
        self._ensure_initialized()
        
        from qiskit_ibm_runtime import least_busy
        
        try:
            suitable = [
                b for b in self._available_backends
                if b.configuration().n_qubits >= min_qubits
                and not b.configuration().simulator
                and b.status().operational
            ]
            
            if suitable:
                backend = least_busy(suitable)
                return backend.name
        except:
            pass
        
        return None
    
    async def run_circuit(
        self,
        circuit_code: str,
        backend_name: Optional[str] = None,
        shots: int = 0
    ) -> Dict[str, Any]:
        """Run a circuit on IBM Quantum hardware.
        
        Args:
            circuit_code: Qiskit circuit code
            backend_name: Specific backend (or auto-select)
            shots: Number of shots (0 = use default)
        
        Returns:
            Dictionary with results or error
        """
        self._ensure_initialized()
        
        shots = shots or self.config.default_shots
        
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            
            # Execute code to get circuit
            exec_globals = {}
            exec(circuit_code, exec_globals)
            qc = exec_globals.get('qc')
            
            if qc is None:
                return {"error": "No circuit 'qc' found in code"}
            
            # Check qubit limit
            if qc.num_qubits > self.config.max_qubits:
                return {"error": f"Circuit has {qc.num_qubits} qubits, max allowed is {self.config.max_qubits}"}
            
            # Add measurements if needed
            if qc.num_clbits == 0:
                qc.measure_all()
            
            # Select backend
            if backend_name:
                backend = self._service.backend(backend_name)
            else:
                backend_name = self.get_least_busy_backend(qc.num_qubits)
                if not backend_name:
                    return {"error": "No suitable backend available"}
                backend = self._service.backend(backend_name)
            
            # Run using Sampler
            sampler = Sampler(backend)
            job = sampler.run([qc], shots=shots)
            
            # Wait for result (with timeout)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, job.result),
                timeout=self.config.timeout
            )
            
            # Extract counts
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
            
            return {
                "success": True,
                "counts": counts,
                "shots": shots,
                "backend": backend_name,
                "job_id": job.job_id()
            }
            
        except asyncio.TimeoutError:
            return {"error": "Job timed out", "timeout": self.config.timeout}
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# IonQ Integration (via Braket)
# ============================================================================

class IonQClient:
    """Client for IonQ quantum hardware via Amazon Braket."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
    
    async def run_circuit(
        self,
        circuit_code: str,
        device: str = "ionq_harmony",
        shots: int = 0
    ) -> Dict[str, Any]:
        """Run a circuit on IonQ hardware.
        
        Args:
            circuit_code: Qiskit circuit code
            device: IonQ device (ionq_harmony, ionq_aria)
            shots: Number of shots
        
        Returns:
            Dictionary with results or error
        """
        shots = shots or self.config.default_shots
        
        try:
            from braket.aws import AwsDevice
            from braket.circuits import Circuit as BraketCircuit
            
            # Execute code to get Qiskit circuit
            exec_globals = {}
            exec(circuit_code, exec_globals)
            qc = exec_globals.get('qc')
            
            if qc is None:
                return {"error": "No circuit 'qc' found in code"}
            
            # Convert Qiskit to Braket circuit
            braket_circuit = self._qiskit_to_braket(qc)
            
            # Get IonQ device
            device_arn = f"arn:aws:braket:::device/qpu/ionq/{device}"
            ionq_device = AwsDevice(device_arn)
            
            # Run
            task = ionq_device.run(braket_circuit, shots=shots)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, task.result),
                timeout=self.config.timeout
            )
            
            # Get counts
            counts = result.measurement_counts
            
            return {
                "success": True,
                "counts": dict(counts),
                "shots": shots,
                "device": device,
                "task_id": task.id
            }
            
        except ImportError:
            return {"error": "amazon-braket-sdk not installed"}
        except asyncio.TimeoutError:
            return {"error": "Job timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def _qiskit_to_braket(self, qc) -> Any:
        """Convert Qiskit circuit to Braket circuit."""
        from braket.circuits import Circuit as BraketCircuit
        
        braket = BraketCircuit()
        
        # Map common gates
        gate_map = {
            'h': lambda q: braket.h(q[0]),
            'x': lambda q: braket.x(q[0]),
            'y': lambda q: braket.y(q[0]),
            'z': lambda q: braket.z(q[0]),
            's': lambda q: braket.s(q[0]),
            't': lambda q: braket.t(q[0]),
            'cx': lambda q: braket.cnot(q[0], q[1]),
            'cz': lambda q: braket.cz(q[0], q[1]),
            'swap': lambda q: braket.swap(q[0], q[1]),
        }
        
        for instruction in qc.data:
            gate_name = instruction.operation.name.lower()
            qubits = [qc.find_bit(q).index for q in instruction.qubits]
            
            if gate_name in gate_map:
                gate_map[gate_name](qubits)
            elif gate_name == 'measure':
                continue  # Braket handles measurement differently
        
        return braket

# ============================================================================
# Simulator Fallback
# ============================================================================

class SimulatorClient:
    """Local simulator for testing and fallback."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
    
    async def run_circuit(
        self,
        circuit_code: str,
        shots: int = 0
    ) -> Dict[str, Any]:
        """Run circuit on local simulator.
        
        Args:
            circuit_code: Qiskit circuit code
            shots: Number of shots
        
        Returns:
            Dictionary with results
        """
        shots = shots or self.config.default_shots
        
        try:
            from qiskit.quantum_info import Statevector
            import numpy as np
            
            # Execute code
            exec_globals = {}
            exec(circuit_code, exec_globals)
            qc = exec_globals.get('qc')
            
            if qc is None:
                return {"error": "No circuit 'qc' found"}
            
            # Add measurements if needed
            if qc.num_clbits == 0:
                qc.measure_all()
            
            # Simulate
            sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
            probs = sv.probabilities_dict()
            
            # Sample
            outcomes = list(probs.keys())
            probabilities = list(probs.values())
            samples = np.random.choice(len(outcomes), size=shots, p=probabilities)
            
            counts = {}
            for s in samples:
                outcome = outcomes[s]
                counts[outcome] = counts.get(outcome, 0) + 1
            
            return {
                "success": True,
                "counts": counts,
                "shots": shots,
                "backend": "local_simulator",
                "statevector_available": True
            }
            
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# Unified Hardware Manager
# ============================================================================

class QuantumHardwareManager:
    """Unified manager for quantum hardware access."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or HardwareConfig.from_env()
        self._ibm_client: Optional[IBMQuantumClient] = None
        self._ionq_client: Optional[IonQClient] = None
        self._simulator = SimulatorClient(self.config)
    
    def _get_ibm_client(self) -> IBMQuantumClient:
        if self._ibm_client is None:
            self._ibm_client = IBMQuantumClient(self.config)
        return self._ibm_client
    
    def _get_ionq_client(self) -> IonQClient:
        if self._ionq_client is None:
            self._ionq_client = IonQClient(self.config)
        return self._ionq_client
    
    async def run_on_hardware(
        self,
        circuit_code: str,
        provider: HardwareProvider = HardwareProvider.SIMULATOR,
        backend: Optional[str] = None,
        shots: int = 0,
        fallback_to_simulator: bool = True
    ) -> Dict[str, Any]:
        """Run circuit on specified hardware with optional fallback.
        
        Args:
            circuit_code: Qiskit circuit code
            provider: Hardware provider to use
            backend: Specific backend name (provider-specific)
            shots: Number of shots
            fallback_to_simulator: Whether to fall back to simulator on failure
        
        Returns:
            Dictionary with results
        """
        result = None
        
        if provider == HardwareProvider.IBM_QUANTUM:
            try:
                client = self._get_ibm_client()
                result = await client.run_circuit(circuit_code, backend, shots)
            except Exception as e:
                result = {"error": f"IBM Quantum failed: {str(e)}"}
        
        elif provider == HardwareProvider.IONQ:
            try:
                client = self._get_ionq_client()
                result = await client.run_circuit(circuit_code, backend or "ionq_harmony", shots)
            except Exception as e:
                result = {"error": f"IonQ failed: {str(e)}"}
        
        elif provider == HardwareProvider.SIMULATOR:
            result = await self._simulator.run_circuit(circuit_code, shots)
        
        # Fallback to simulator if needed
        if fallback_to_simulator and result and result.get("error"):
            logger.warning(f"Hardware failed, falling back to simulator: {result['error']}")
            sim_result = await self._simulator.run_circuit(circuit_code, shots)
            sim_result["fallback"] = True
            sim_result["original_error"] = result["error"]
            return sim_result
        
        return result or {"error": "Unknown provider"}
    
    async def get_available_hardware(self) -> Dict[str, Any]:
        """Get status of all available hardware."""
        status = {
            "simulator": {"available": True, "status": "ready"},
            "ibm_quantum": {"available": False, "status": "not_configured"},
            "ionq": {"available": False, "status": "not_configured"}
        }
        
        # Check IBM Quantum
        if self.config.ibm_token:
            try:
                client = self._get_ibm_client()
                backends = client.get_available_backends()
                status["ibm_quantum"] = {
                    "available": True,
                    "status": "connected",
                    "backends": backends[:5]  # Limit to 5
                }
            except Exception as e:
                status["ibm_quantum"] = {
                    "available": False,
                    "status": "error",
                    "error": str(e)
                }
        
        # Check IonQ (via Braket)
        try:
            import braket
            status["ionq"] = {
                "available": True,
                "status": "sdk_available",
                "note": "Requires AWS credentials"
            }
        except ImportError:
            status["ionq"]["status"] = "sdk_not_installed"
        
        return status
    
    async def run_with_best_available(
        self,
        circuit_code: str,
        shots: int = 0,
        prefer_real_hardware: bool = False
    ) -> Dict[str, Any]:
        """Run on best available hardware.
        
        Args:
            circuit_code: Qiskit circuit code
            shots: Number of shots
            prefer_real_hardware: Try real hardware first
        
        Returns:
            Dictionary with results
        """
        if prefer_real_hardware:
            # Try IBM Quantum first
            if self.config.ibm_token:
                result = await self.run_on_hardware(
                    circuit_code,
                    HardwareProvider.IBM_QUANTUM,
                    shots=shots,
                    fallback_to_simulator=True
                )
                if result.get("success"):
                    return result
        
        # Fall back to simulator
        return await self._simulator.run_circuit(circuit_code, shots)

# ============================================================================
# MCP Tool Integration Helper
# ============================================================================

async def execute_on_quantum_hardware(
    circuit_code: str,
    provider: str = "simulator",
    backend: str = "",
    shots: int = 1000
) -> Dict[str, Any]:
    """Helper function for MCP tool integration.
    
    Args:
        circuit_code: Qiskit circuit code
        provider: "simulator", "ibm_quantum", or "ionq"
        backend: Specific backend name
        shots: Number of shots
    
    Returns:
        Execution results
    """
    manager = QuantumHardwareManager()
    
    provider_map = {
        "simulator": HardwareProvider.SIMULATOR,
        "ibm_quantum": HardwareProvider.IBM_QUANTUM,
        "ionq": HardwareProvider.IONQ
    }
    
    hw_provider = provider_map.get(provider.lower(), HardwareProvider.SIMULATOR)
    
    return await manager.run_on_hardware(
        circuit_code,
        provider=hw_provider,
        backend=backend or None,
        shots=shots,
        fallback_to_simulator=True
    )

# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Testing Quantum Hardware Module...")
        
        manager = QuantumHardwareManager()
        
        # Check available hardware
        status = await manager.get_available_hardware()
        print(f"Hardware status: {json.dumps(status, indent=2)}")
        
        # Test on simulator
        test_code = """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
        
        result = await manager.run_on_hardware(
            test_code,
            HardwareProvider.SIMULATOR,
            shots=1000
        )
        print(f"\nSimulator result: {json.dumps(result, indent=2)}")
    
    asyncio.run(main())
