"""Internal implementation backing the layered dynamic runtime package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ...circuit import Circuit
from ...compilation.routing import CouplingMap, route_to_topology
from ...core.ir import Instruction
from ._conditions import classical_width, instruction_conditions
from .circuit import DynamicCircuit
from .result import DynamicExecutionResult

_classical_width = classical_width
_instruction_conditions = instruction_conditions


@dataclass(frozen=True)
class DynamicBackendCompatibility:
    """Fail-closed compatibility report for dynamic hardware execution."""

    compatible: bool
    required_classical_bits: int
    blockers: tuple[str, ...]
    qasm_version: float = 3.0
    dynamic_dialect: str = "openqasm3"

    def summary(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "required_classical_bits": self.required_classical_bits,
            "qasm_version": self.qasm_version,
            "dynamic_dialect": self.dynamic_dialect,
            "blockers": self.blockers,
        }


def _measure_wire(
    state: torch.Tensor,
    wire: int,
    n_wires: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    shaped = state.reshape((2,) * n_wires)
    axes = tuple(axis for axis in range(n_wires) if axis != wire)
    magnitudes = torch.abs(shaped) ** 2
    probabilities = magnitudes.sum(dim=axes) if axes else magnitudes
    bit = int(torch.multinomial(probabilities, 1, generator=generator).item())
    selector = torch.arange(2, device=state.device) == bit
    view_shape = [1] * n_wires
    view_shape[wire] = 2
    collapsed = shaped * selector.reshape(view_shape)
    norm = torch.linalg.vector_norm(collapsed)
    return (collapsed / norm).reshape(1, -1), bit


def _apply_instruction(
    state: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor:
    circuit = Circuit(
        n_qubits=n_wires,
        bsz=1,
        device=state.device,
        dtype=state.dtype,
        inputs=state,
    )
    circuit._instructions.append(instruction)
    return circuit.state()


def run_dynamic(
    circuit: DynamicCircuit,
    *,
    shots: int,
    seed: int | None = None,
) -> DynamicExecutionResult:
    """Execute independent statevector trajectories with classical feedback."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("run_dynamic requires an experimental DynamicCircuit")
    if int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    for instruction in circuit._instructions:
        for value in instruction.params.values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise RuntimeError("dynamic trajectory execution is not differentiable")

    classical_width = _classical_width(circuit)

    generator = torch.Generator(device=torch.device(circuit.device).type)
    if seed is not None:
        generator.manual_seed(int(seed))
    initial_states = circuit.initial_state()
    batched_states = []
    batched_samples = []
    batched_classical = []
    for batch_index in range(circuit.bsz):
        final_states = []
        final_samples = []
        classical_rows = []
        for _ in range(int(shots)):
            state = initial_states[batch_index : batch_index + 1]
            classical = [-1] * classical_width
            for instruction in circuit._instructions:
                conditions = _instruction_conditions(instruction)
                skip = False
                for bit_index, expected in conditions:
                    if classical[bit_index] < 0:
                        raise RuntimeError(
                            f"classical bit {bit_index} was read before measurement"
                        )
                    if classical[bit_index] != expected:
                        skip = True
                        break
                if skip:
                    continue
                if instruction.name == "measure":
                    state, bit = _measure_wire(
                        state,
                        instruction.wires[0],
                        circuit.n_wires,
                        generator=generator,
                    )
                    classical[int(instruction.metadata["classical_bit"])] = bit
                elif instruction.name == "reset":
                    state, bit = _measure_wire(
                        state,
                        instruction.wires[0],
                        circuit.n_wires,
                        generator=generator,
                    )
                    if bit:
                        state = _apply_instruction(
                            state,
                            Instruction("x", instruction.wires),
                            n_wires=circuit.n_wires,
                        )
                else:
                    state = _apply_instruction(
                        state,
                        instruction,
                        n_wires=circuit.n_wires,
                    )
            index = int(
                torch.multinomial(
                    torch.abs(state.reshape(-1)) ** 2,
                    1,
                    generator=generator,
                ).item()
            )
            bits = [
                (index >> (circuit.n_wires - wire - 1)) & 1
                for wire in range(circuit.n_wires)
            ]
            final_states.append(state.reshape(-1))
            final_samples.append(bits)
            classical_rows.append(classical)
        batched_states.append(torch.stack(final_states))
        batched_samples.append(
            torch.tensor(final_samples, dtype=torch.int64, device=circuit.device)
        )
        batched_classical.append(
            torch.tensor(classical_rows, dtype=torch.int64, device=circuit.device)
        )
    states_tensor = torch.stack(batched_states)
    samples_tensor = torch.stack(batched_samples)
    classical_tensor = torch.stack(batched_classical)
    if circuit.bsz == 1:
        states_tensor = states_tensor[0]
        samples_tensor = samples_tensor[0]
        classical_tensor = classical_tensor[0]
    return DynamicExecutionResult(
        samples=samples_tensor,
        classical_bits=classical_tensor,
        final_states=states_tensor,
        shots=int(shots),
        seed=seed,
    )


from .dialects.braket_iqm import export_braket_iqm_dynamic_qasm3  # noqa: E402
from .dialects.openqasm3 import (  # noqa: E402
    export_dynamic_qasm3,
    export_dynamic_qasm3_for_backend,
)


def assess_dynamic_backend(
    circuit: DynamicCircuit,
    backend: Any,
) -> DynamicBackendCompatibility:
    """Check provider-declared dynamic-circuit capabilities without guessing."""

    blockers = []
    if circuit.n_wires > int(getattr(backend, "n_wires", 0)):
        blockers.append("circuit_exceeds_backend_qubit_capacity")
    if not bool(getattr(backend, "supports_openqasm", False)):
        blockers.append("backend_does_not_support_openqasm")
    if not bool(getattr(backend, "supports_dynamic_circuits", False)):
        blockers.append("backend_does_not_declare_dynamic_circuit_support")
    required = _classical_width(circuit)
    maximum = getattr(backend, "max_classical_bits", None)
    if maximum is not None and required > int(maximum):
        blockers.append("required_classical_bits_exceed_backend_limit")
    dialect = getattr(backend, "dynamic_dialect", None) or "openqasm3"
    try:
        export_dynamic_qasm3_for_backend(circuit, backend)
    except (TypeError, ValueError) as exc:
        blockers.append(str(exc))
    return DynamicBackendCompatibility(
        compatible=not blockers,
        required_classical_bits=required,
        blockers=tuple(blockers),
        dynamic_dialect=dialect,
    )


def route_dynamic_circuit(
    circuit: DynamicCircuit,
    coupling_map: CouplingMap,
) -> DynamicCircuit:
    """Route each gate with restored layout across every dynamic boundary."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("dynamic routing requires DynamicCircuit")
    routed_ir = route_to_topology(
        circuit.to_ir(),
        coupling_map,
        strategy="restore_after_each_gate",
    )
    boundaries = tuple(
        index
        for index, instruction in enumerate(circuit._instructions)
        if instruction.metadata.get("is_dynamic")
    )
    metadata = dict(routed_ir.metadata)
    routing = dict(metadata["routing"])
    routing["dynamic_boundary_count"] = len(boundaries)
    routing["dynamic_boundary_source_indices"] = boundaries
    routing["dynamic_boundary_mapping_policy"] = "identity_restored_per_gate"
    routing["conditional_routing_semantics"] = "unconditional_swap_sandwich"
    metadata["routing"] = routing
    from dataclasses import replace

    routed_ir = replace(routed_ir, metadata=metadata)
    routed = DynamicCircuit(
        n_qubits=circuit.n_wires,
        bsz=circuit.bsz,
        device=circuit.device,
        dtype=circuit.dtype,
        inputs=circuit._inputs,
        config=circuit.runtime_config,
    )
    routed._instructions.extend(routed_ir.instructions)
    routed._ir_cache = routed_ir
    return routed


def create_dynamic_deployment_package(
    circuit: DynamicCircuit,
    *,
    backend: Any,
    name: str = "flagquantum_dynamic_job",
    shots: int = 1024,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Create a sealed OpenQASM 3 deployment package after capability checks."""

    from ...deployment.cloud import DeploymentPackage
    from ...deployment.routing_evidence import (
        DEPLOYMENT_PACKAGE_SCHEMA,
        build_deployment_routing_evidence,
        deployment_artifact_sha256,
        stable_payload_sha256,
    )

    if int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    if circuit.bsz != 1 or circuit._inputs is not None:
        raise ValueError(
            "dynamic hardware deployment requires one circuit with the default "
            "|0...0> initial state"
        )
    if circuit.parameter_names:
        raise ValueError("dynamic deployment requires all parameters to be bound")
    compatibility = assess_dynamic_backend(circuit, backend)
    if not compatibility.compatible:
        raise RuntimeError(
            "dynamic backend is incompatible: " + ", ".join(compatibility.blockers)
        )

    deployment_circuit = (
        route_dynamic_circuit(circuit, backend.coupling_map)
        if backend.coupling_map is not None
        else circuit
    )
    deployment_compatibility = assess_dynamic_backend(deployment_circuit, backend)
    if not deployment_compatibility.compatible:
        raise RuntimeError(
            "routed dynamic circuit is incompatible: "
            + ", ".join(deployment_compatibility.blockers)
        )
    deployment_ir = deployment_circuit.to_ir()
    qasm = export_dynamic_qasm3_for_backend(deployment_circuit, backend)
    routing_plan = dict(deployment_ir.metadata.get("routing", {}) or {})
    routing_evidence = build_deployment_routing_evidence(
        routing_plan,
        routing_reused=False,
        n_wires=circuit.n_wires,
        coupling_map=backend.coupling_map,
    )
    routing_hash = stable_payload_sha256(routing_evidence)
    package_metadata = {
        **dict(metadata or {}),
        "source": "flagquantum",
        "compiled": backend.coupling_map is not None,
        "dynamic_circuit": True,
        "dynamic_execution_semantics": "provider_mid_circuit_measurement",
        "dynamic_dialect": compatibility.dynamic_dialect,
        "mid_circuit_measurements_returned": (
            False if compatibility.dynamic_dialect == "braket_iqm" else None
        ),
        "target_provider": backend.provider,
        "target_backend": backend.name,
        "deployment_package_schema": DEPLOYMENT_PACKAGE_SCHEMA,
        "deployment_program_format": "openqasm-3",
        "routing_evidence": routing_evidence,
        "routing_plan": routing_evidence["routing_plan"],
        "routing_reused": False,
        "routing_evidence_sha256": routing_hash,
        "dynamic_backend_compatibility": compatibility.summary(),
    }
    package_metadata["deployment_artifact_sha256"] = deployment_artifact_sha256(
        name=name,
        backend_provider=backend.provider,
        backend_name=backend.name,
        shots=int(shots),
        program_format="openqasm-3",
        program=qasm,
        routing_evidence_sha256=routing_hash,
    )
    return DeploymentPackage(
        name=name,
        ir=deployment_ir,
        qasm=qasm,
        shots=int(shots),
        qasm_version=3.0,
        backend=backend,
        metadata=package_metadata,
    )


def deploy_dynamic_circuit(
    circuit: DynamicCircuit,
    provider: Any,
    *,
    backend: Any,
    name: str = "flagquantum_dynamic_job",
    shots: int = 1024,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Seal and submit a dynamic circuit through the provider run contract."""

    package = create_dynamic_deployment_package(
        circuit,
        backend=backend,
        name=name,
        shots=shots,
        metadata=metadata,
    )
    return provider.run(package)


__all__ = (
    "DynamicCircuit",
    "DynamicBackendCompatibility",
    "DynamicExecutionResult",
    "assess_dynamic_backend",
    "create_dynamic_deployment_package",
    "deploy_dynamic_circuit",
    "export_braket_iqm_dynamic_qasm3",
    "export_dynamic_qasm3",
    "export_dynamic_qasm3_for_backend",
    "route_dynamic_circuit",
    "run_dynamic",
)
