"""
UniQ-MCP Quantum Hardware Module - Unified AWS Braket Access

This module provides unified access to quantum hardware through AWS Braket as the
primary interface for all supported providers:
- IonQ (trapped ion)
- Rigetti (superconducting)
- IQM (superconducting)
- QuEra (neutral atom)
- AQT (trapped ion)

IBM Quantum is available as a separate option (not on Braket).

The module handles:
- Unified device selection across providers
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

logger = logging.getLogger("uniq-mcp.quantum_hardware")

# ============================================================================
# Configuration
# ============================================================================

class HardwareProvider(Enum):
    """Supported quantum hardware providers."""
    # AWS Braket providers (unified access)
    IONQ = "ionq"
    RIGETTI = "rigetti"
    IQM = "iqm"
    QUERA = "quera"
    AQT = "aqt"
    # Braket simulators
    BRAKET_SV1 = "braket_sv1"  # State vector simulator
    BRAKET_DM1 = "braket_dm1"  # Density matrix simulator
    BRAKET_TN1 = "braket_tn1"  # Tensor network simulator
    # Local simulator
    LOCAL_SIMULATOR = "local_simulator"
    # IBM (separate, not on Braket)
    IBM_QUANTUM = "ibm_quantum"

@dataclass
class BraketDevice:
    """AWS Braket device configuration."""
    provider: HardwareProvider
    name: str
    arn: str
    qubits: int
    device_type: str  # "qpu" or "simulator"
    region: str
    status: str = "available"

# AWS Braket device catalog (as of 2025)
BRAKET_DEVICES = {
    # IonQ devices
    "ionq_aria": BraketDevice(
        provider=HardwareProvider.IONQ,
        name="IonQ Aria",
        arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
        qubits=25,
        device_type="qpu",
        region="us-east-1"
    ),
    "ionq_aria2": BraketDevice(
        provider=HardwareProvider.IONQ,
        name="IonQ Aria 2",
        arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2",
        qubits=25,
        device_type="qpu",
        region="us-east-1"
    ),
    "ionq_forte": BraketDevice(
        provider=HardwareProvider.IONQ,
        name="IonQ Forte",
        arn="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        qubits=32,
        device_type="qpu",
        region="us-east-1"
    ),
    # Rigetti devices
    "rigetti_ankaa2": BraketDevice(
        provider=HardwareProvider.RIGETTI,
        name="Rigetti Ankaa-2",
        arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2",
        qubits=84,
        device_type="qpu",
        region="us-west-1"
    ),
    # IQM devices
    "iqm_garnet": BraketDevice(
        provider=HardwareProvider.IQM,
        name="IQM Garnet",
        arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
        qubits=20,
        device_type="qpu",
        region="eu-north-1"
    ),
    # QuEra devices
    "quera_aquila": BraketDevice(
        provider=HardwareProvider.QUERA,
        name="QuEra Aquila",
        arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
        qubits=256,
        device_type="qpu",
        region="us-east-1"
    ),
    # Braket simulators
    "sv1": BraketDevice(
        provider=HardwareProvider.BRAKET_SV1,
        name="State Vector Simulator (SV1)",
        arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        qubits=34,
        device_type="simulator",
        region="us-east-1"
    ),
    "dm1": BraketDevice(
        provider=HardwareProvider.BRAKET_DM1,
        name="Density Matrix Simulator (DM1)",
        arn="arn:aws:braket:::device/quantum-simulator/amazon/dm1",
        qubits=17,
        device_type="simulator",
        region="us-east-1"
    ),
    "tn1": BraketDevice(
        provider=HardwareProvider.BRAKET_TN1,
        name="Tensor Network Simulator (TN1)",
        arn="arn:aws:braket:::device/quantum-simulator/amazon/tn1",
        qubits=50,
        device_type="simulator",
        region="us-west-2"
    ),
}

@dataclass
class HardwareConfig:
    """Configuration for quantum hardware access."""
    # AWS Braket
    aws_region: str = "us-east-1"
    aws_access_key: str = ""
    aws_secret_key: str = ""
    s3_bucket: str = ""  # For Braket results
    # IBM Quantum (separate)
    ibm_token: str = ""
    ibm_channel: str = "ibm_quantum"
    # General
    default_shots: int = 1000
    max_qubits: int = 10  # Safety limit
    timeout: float = 300.0  # 5 minutes
    
    @classmethod
    def from_env(cls) -> "HardwareConfig":
        return cls(
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            s3_bucket=os.getenv("BRAKET_S3_BUCKET", ""),
            ibm_token=os.getenv("IBM_QUANTUM_TOKEN", ""),
            ibm_channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
        )

# ============================================================================
# Qiskit to Braket Circuit Conversion
# ============================================================================

def qiskit_to_braket(qc) -> Any:
    """Convert Qiskit circuit to Braket circuit.
    
    Args:
        qc: Qiskit QuantumCircuit
    
    Returns:
        Braket Circuit
    """
    from braket.circuits import Circuit as BraketCircuit
    
    braket = BraketCircuit()
    
    # Gate mapping from Qiskit to Braket
    for instruction in qc.data:
        gate_name = instruction.operation.name.lower()
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        params = instruction.operation.params
        
        # Single-qubit gates
        if gate_name == 'h':
            braket.h(qubits[0])
        elif gate_name == 'x':
            braket.x(qubits[0])
        elif gate_name == 'y':
            braket.y(qubits[0])
        elif gate_name == 'z':
            braket.z(qubits[0])
        elif gate_name == 's':
            braket.s(qubits[0])
        elif gate_name == 'sdg':
            braket.si(qubits[0])
        elif gate_name == 't':
            braket.t(qubits[0])
        elif gate_name == 'tdg':
            braket.ti(qubits[0])
        elif gate_name == 'rx':
            braket.rx(qubits[0], params[0])
        elif gate_name == 'ry':
            braket.ry(qubits[0], params[0])
        elif gate_name == 'rz':
            braket.rz(qubits[0], params[0])
        # Two-qubit gates
        elif gate_name == 'cx' or gate_name == 'cnot':
            braket.cnot(qubits[0], qubits[1])
        elif gate_name == 'cz':
            braket.cz(qubits[0], qubits[1])
        elif gate_name == 'swap':
            braket.swap(qubits[0], qubits[1])
        elif gate_name == 'iswap':
            braket.iswap(qubits[0], qubits[1])
        # Three-qubit gates
        elif gate_name == 'ccx' or gate_name == 'toffoli':
            braket.ccnot(qubits[0], qubits[1], qubits[2])
        elif gate_name == 'cswap' or gate_name == 'fredkin':
            braket.cswap(qubits[0], qubits[1], qubits[2])
        elif gate_name == 'measure':
            continue  # Braket handles measurement differently
        elif gate_name == 'barrier':
            continue  # No equivalent in Braket
        else:
            logger.warning(f"Unsupported gate: {gate_name}, skipping")
    
    return braket

# ============================================================================
# AWS Braket Client (Unified Hardware Access)
# ============================================================================

class BraketClient:
    """Unified client for AWS Braket quantum hardware."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of Braket."""
        if self._initialized:
            return
        
        try:
            import boto3
            from braket.aws import AwsDevice, AwsSession
            
            # Create AWS session
            if self.config.aws_access_key and self.config.aws_secret_key:
                boto_session = boto3.Session(
                    aws_access_key_id=self.config.aws_access_key,
                    aws_secret_access_key=self.config.aws_secret_key,
                    region_name=self.config.aws_region
                )
            else:
                # Use default credentials
                boto_session = boto3.Session(region_name=self.config.aws_region)
            
            self._boto_session = boto_session
            self._initialized = True
            
        except ImportError:
            raise ImportError("amazon-braket-sdk not installed. Run: pip install amazon-braket-sdk")
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available Braket devices."""
        devices = []
        for device_id, device in BRAKET_DEVICES.items():
            devices.append({
                "id": device_id,
                "name": device.name,
                "provider": device.provider.value,
                "qubits": device.qubits,
                "type": device.device_type,
                "region": device.region,
                "arn": device.arn
            })
        return devices
    
    def get_device_status(self, device_id: str) -> Dict:
        """Get real-time status of a device."""
        self._ensure_initialized()
        
        if device_id not in BRAKET_DEVICES:
            return {"error": f"Unknown device: {device_id}"}
        
        device_config = BRAKET_DEVICES[device_id]
        
        try:
            from braket.aws import AwsDevice
            
            device = AwsDevice(device_config.arn, aws_session=self._boto_session)
            status = device.status
            
            return {
                "device_id": device_id,
                "name": device_config.name,
                "status": status,
                "is_available": status == "ONLINE",
                "queue_depth": getattr(device, 'queue_depth', 'unknown')
            }
        except Exception as e:
            return {
                "device_id": device_id,
                "status": "error",
                "error": str(e)
            }
    
    async def run_circuit(
        self,
        circuit_code: str,
        device_id: str = "sv1",
        shots: int = 0
    ) -> Dict[str, Any]:
        """Run a circuit on a Braket device.
        
        Args:
            circuit_code: Qiskit circuit code
            device_id: Braket device ID (e.g., "ionq_aria", "sv1")
            shots: Number of shots
        
        Returns:
            Dictionary with results
        """
        self._ensure_initialized()
        
        shots = shots or self.config.default_shots
        
        if device_id not in BRAKET_DEVICES:
            return {"error": f"Unknown device: {device_id}"}
        
        device_config = BRAKET_DEVICES[device_id]
        
        try:
            from braket.aws import AwsDevice
            
            # Execute Qiskit code to get circuit
            exec_globals = {}
            exec(circuit_code, exec_globals)
            qc = exec_globals.get('qc')
            
            if qc is None:
                return {"error": "No circuit 'qc' found in code"}
            
            # Check qubit limit
            if qc.num_qubits > self.config.max_qubits:
                return {"error": f"Circuit has {qc.num_qubits} qubits, max is {self.config.max_qubits}"}
            
            if qc.num_qubits > device_config.qubits:
                return {"error": f"Circuit has {qc.num_qubits} qubits, device supports {device_config.qubits}"}
            
            # Convert to Braket circuit
            braket_circuit = qiskit_to_braket(qc)
            
            # Get device
            device = AwsDevice(device_config.arn, aws_session=self._boto_session)
            
            # Determine S3 location for results
            s3_folder = self.config.s3_bucket or f"amazon-braket-{self.config.aws_region}"
            s3_location = (s3_folder, "uniq-mcp-results")
            
            # Run task
            task = device.run(braket_circuit, shots=shots, s3_destination_folder=s3_location)
            
            # Wait for result
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, task.result),
                timeout=self.config.timeout
            )
            
            # Extract counts
            counts = dict(result.measurement_counts)
            
            return {
                "success": True,
                "counts": counts,
                "shots": shots,
                "device": device_id,
                "device_name": device_config.name,
                "provider": device_config.provider.value,
                "task_id": task.id
            }
            
        except asyncio.TimeoutError:
            return {"error": "Job timed out", "timeout": self.config.timeout}
        except ImportError as e:
            return {"error": f"Braket SDK not available: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def run_on_best_device(
        self,
        circuit_code: str,
        provider: Optional[HardwareProvider] = None,
        shots: int = 0,
        prefer_simulator: bool = False
    ) -> Dict[str, Any]:
        """Run on the best available device.
        
        Args:
            circuit_code: Qiskit circuit code
            provider: Preferred provider (optional)
            shots: Number of shots
            prefer_simulator: Prefer simulator over QPU
        
        Returns:
            Dictionary with results
        """
        # Determine device priority
        if prefer_simulator:
            device_priority = ["sv1", "dm1", "tn1"]
        elif provider == HardwareProvider.IONQ:
            device_priority = ["ionq_aria", "ionq_aria2", "ionq_forte", "sv1"]
        elif provider == HardwareProvider.RIGETTI:
            device_priority = ["rigetti_ankaa2", "sv1"]
        elif provider == HardwareProvider.IQM:
            device_priority = ["iqm_garnet", "sv1"]
        else:
            # Default: try QPUs first, then simulators
            device_priority = ["ionq_aria", "rigetti_ankaa2", "iqm_garnet", "sv1"]
        
        # Try devices in priority order
        for device_id in device_priority:
            result = await self.run_circuit(circuit_code, device_id, shots)
            if result.get("success"):
                return result
            logger.warning(f"Device {device_id} failed: {result.get('error')}")
        
        return {"error": "All devices failed", "tried": device_priority}

# ============================================================================
# IBM Quantum Client (Separate from Braket)
# ============================================================================

class IBMQuantumClient:
    """Client for IBM Quantum hardware (not available on Braket)."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self._service = None
    
    def _ensure_initialized(self):
        """Lazy initialization of IBM Quantum service."""
        if self._service is not None:
            return
        
        if not self.config.ibm_token:
            raise ValueError("IBM_QUANTUM_TOKEN not configured")
        
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            try:
                self._service = QiskitRuntimeService()
            except:
                QiskitRuntimeService.save_account(
                    channel=self.config.ibm_channel,
                    token=self.config.ibm_token,
                    overwrite=True
                )
                self._service = QiskitRuntimeService()
                
        except ImportError:
            raise ImportError("qiskit-ibm-runtime not installed")
    
    def get_available_backends(self) -> List[Dict]:
        """Get available IBM Quantum backends."""
        self._ensure_initialized()
        
        backends = []
        for backend in self._service.backends():
            try:
                config = backend.configuration()
                status = backend.status()
                backends.append({
                    "name": backend.name,
                    "qubits": config.n_qubits,
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "simulator": config.simulator
                })
            except:
                continue
        return backends
    
    async def run_circuit(
        self,
        circuit_code: str,
        backend_name: Optional[str] = None,
        shots: int = 0
    ) -> Dict[str, Any]:
        """Run circuit on IBM Quantum."""
        self._ensure_initialized()
        shots = shots or self.config.default_shots
        
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit_ibm_runtime import least_busy
            
            # Execute code
            exec_globals = {}
            exec(circuit_code, exec_globals)
            qc = exec_globals.get('qc')
            
            if qc is None:
                return {"error": "No circuit 'qc' found"}
            
            # Add measurements
            if qc.num_clbits == 0:
                qc.measure_all()
            
            # Select backend
            if backend_name:
                backend = self._service.backend(backend_name)
            else:
                suitable = [
                    b for b in self._service.backends()
                    if b.configuration().n_qubits >= qc.num_qubits
                    and not b.configuration().simulator
                    and b.status().operational
                ]
                if suitable:
                    backend = least_busy(suitable)
                else:
                    return {"error": "No suitable backend available"}
            
            # Run
            sampler = Sampler(backend)
            job = sampler.run([qc], shots=shots)
            
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, job.result),
                timeout=self.config.timeout
            )
            
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
            
            return {
                "success": True,
                "counts": counts,
                "shots": shots,
                "backend": backend.name,
                "provider": "ibm_quantum",
                "job_id": job.job_id()
            }
            
        except asyncio.TimeoutError:
            return {"error": "Job timed out"}
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# Local Simulator (Fallback)
# ============================================================================

class LocalSimulator:
    """Local Qiskit simulator for testing and fallback."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
    
    async def run_circuit(
        self,
        circuit_code: str,
        shots: int = 0
    ) -> Dict[str, Any]:
        """Run circuit on local simulator."""
        shots = shots or self.config.default_shots
        
        try:
            from qiskit.quantum_info import Statevector
            import numpy as np
            
            exec_globals = {}
            exec(circuit_code, exec_globals)
            qc = exec_globals.get('qc')
            
            if qc is None:
                return {"error": "No circuit 'qc' found"}
            
            if qc.num_clbits == 0:
                qc.measure_all()
            
            sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
            probs = sv.probabilities_dict()
            
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
                "device": "local_simulator",
                "provider": "qiskit"
            }
            
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# Unified Hardware Manager
# ============================================================================

class QuantumHardwareManager:
    """Unified manager for all quantum hardware access."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or HardwareConfig.from_env()
        self._braket_client: Optional[BraketClient] = None
        self._ibm_client: Optional[IBMQuantumClient] = None
        self._local_simulator = LocalSimulator(self.config)
    
    def _get_braket_client(self) -> BraketClient:
        if self._braket_client is None:
            self._braket_client = BraketClient(self.config)
        return self._braket_client
    
    def _get_ibm_client(self) -> IBMQuantumClient:
        if self._ibm_client is None:
            self._ibm_client = IBMQuantumClient(self.config)
        return self._ibm_client
    
    async def run_on_hardware(
        self,
        circuit_code: str,
        provider: str = "local_simulator",
        device: str = "",
        shots: int = 0,
        fallback_to_simulator: bool = True
    ) -> Dict[str, Any]:
        """Run circuit on specified hardware.
        
        Args:
            circuit_code: Qiskit circuit code
            provider: Provider name (ionq, rigetti, iqm, ibm_quantum, local_simulator, etc.)
            device: Specific device ID (optional)
            shots: Number of shots
            fallback_to_simulator: Fall back to simulator on failure
        
        Returns:
            Dictionary with results
        """
        provider_lower = provider.lower()
        result = None
        
        # Route to appropriate client
        if provider_lower == "local_simulator":
            result = await self._local_simulator.run_circuit(circuit_code, shots)
        
        elif provider_lower == "ibm_quantum":
            try:
                client = self._get_ibm_client()
                result = await client.run_circuit(circuit_code, device or None, shots)
            except Exception as e:
                result = {"error": f"IBM Quantum failed: {str(e)}"}
        
        elif provider_lower in ["ionq", "rigetti", "iqm", "quera", "aqt", "braket_sv1", "braket_dm1", "braket_tn1"]:
            # AWS Braket providers
            try:
                client = self._get_braket_client()
                
                # Map provider to default device if not specified
                if not device:
                    device_map = {
                        "ionq": "ionq_aria",
                        "rigetti": "rigetti_ankaa2",
                        "iqm": "iqm_garnet",
                        "quera": "quera_aquila",
                        "braket_sv1": "sv1",
                        "braket_dm1": "dm1",
                        "braket_tn1": "tn1",
                    }
                    device = device_map.get(provider_lower, "sv1")
                
                result = await client.run_circuit(circuit_code, device, shots)
            except Exception as e:
                result = {"error": f"Braket failed: {str(e)}"}
        
        else:
            result = {"error": f"Unknown provider: {provider}"}
        
        # Fallback to local simulator
        if fallback_to_simulator and result and result.get("error"):
            logger.warning(f"Hardware failed, falling back to simulator: {result['error']}")
            sim_result = await self._local_simulator.run_circuit(circuit_code, shots)
            sim_result["fallback"] = True
            sim_result["original_error"] = result["error"]
            return sim_result
        
        return result or {"error": "Unknown error"}
    
    async def get_available_hardware(self) -> Dict[str, Any]:
        """Get status of all available hardware."""
        status = {
            "local_simulator": {
                "available": True,
                "status": "ready",
                "provider": "qiskit"
            },
            "aws_braket": {
                "available": False,
                "status": "checking",
                "devices": []
            },
            "ibm_quantum": {
                "available": False,
                "status": "not_configured"
            }
        }
        
        # Check AWS Braket
        try:
            client = self._get_braket_client()
            devices = client.get_available_devices()
            status["aws_braket"] = {
                "available": True,
                "status": "configured",
                "devices": devices,
                "note": "Pending AWS Braket activation" if not self.config.aws_access_key else "Ready"
            }
        except Exception as e:
            status["aws_braket"] = {
                "available": False,
                "status": "error",
                "error": str(e)
            }
        
        # Check IBM Quantum
        if self.config.ibm_token:
            try:
                client = self._get_ibm_client()
                backends = client.get_available_backends()
                status["ibm_quantum"] = {
                    "available": True,
                    "status": "connected",
                    "backends": backends[:5]
                }
            except Exception as e:
                status["ibm_quantum"] = {
                    "available": False,
                    "status": "error",
                    "error": str(e)
                }
        
        return status
    
    def list_all_devices(self) -> List[Dict]:
        """List all available devices across all providers."""
        devices = []
        
        # Local simulator
        devices.append({
            "id": "local_simulator",
            "name": "Local Qiskit Simulator",
            "provider": "qiskit",
            "qubits": 30,
            "type": "simulator",
            "status": "ready"
        })
        
        # AWS Braket devices
        for device_id, device in BRAKET_DEVICES.items():
            devices.append({
                "id": device_id,
                "name": device.name,
                "provider": device.provider.value,
                "qubits": device.qubits,
                "type": device.device_type,
                "region": device.region,
                "status": "available"  # Would need API call for real status
            })
        
        # IBM Quantum (placeholder)
        if self.config.ibm_token:
            devices.append({
                "id": "ibm_quantum",
                "name": "IBM Quantum (various)",
                "provider": "ibm_quantum",
                "qubits": "varies",
                "type": "qpu",
                "status": "configured"
            })
        
        return devices

# ============================================================================
# Helper Functions for MCP Integration
# ============================================================================

async def execute_on_quantum_hardware(
    circuit_code: str,
    provider: str = "local_simulator",
    device: str = "",
    shots: int = 1000
) -> Dict[str, Any]:
    """Helper function for MCP tool integration.
    
    Args:
        circuit_code: Qiskit circuit code
        provider: Provider name (ionq, rigetti, iqm, ibm_quantum, local_simulator)
        device: Specific device ID
        shots: Number of shots
    
    Returns:
        Execution results
    """
    manager = QuantumHardwareManager()
    return await manager.run_on_hardware(
        circuit_code,
        provider=provider,
        device=device,
        shots=shots,
        fallback_to_simulator=True
    )

def get_provider_info() -> Dict[str, Any]:
    """Get information about all supported providers."""
    return {
        "aws_braket": {
            "description": "Unified access to multiple quantum hardware providers",
            "providers": ["IonQ", "Rigetti", "IQM", "QuEra", "AQT"],
            "simulators": ["SV1", "DM1", "TN1"],
            "requires": "AWS credentials and Braket activation"
        },
        "ibm_quantum": {
            "description": "IBM Quantum hardware (separate from Braket)",
            "providers": ["IBM"],
            "requires": "IBM Quantum token"
        },
        "local_simulator": {
            "description": "Local Qiskit statevector simulator",
            "providers": ["Qiskit"],
            "requires": "Nothing (always available)"
        }
    }

# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Testing Unified Quantum Hardware Module...")
        print("=" * 60)
        
        manager = QuantumHardwareManager()
        
        # List all devices
        print("\n📋 Available Devices:")
        devices = manager.list_all_devices()
        for d in devices:
            print(f"  - {d['id']}: {d['name']} ({d['provider']}, {d['qubits']} qubits)")
        
        # Check hardware status
        print("\n🔍 Hardware Status:")
        status = await manager.get_available_hardware()
        print(json.dumps(status, indent=2, default=str))
        
        # Test on local simulator
        print("\n⚛️ Testing Local Simulator:")
        test_code = """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
        result = await manager.run_on_hardware(test_code, "local_simulator", shots=100)
        print(f"Result: {json.dumps(result, indent=2, default=str)}")
        
        # Provider info
        print("\n📚 Provider Information:")
        print(json.dumps(get_provider_info(), indent=2))
    
    asyncio.run(main())
