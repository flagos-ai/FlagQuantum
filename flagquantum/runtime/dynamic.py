"""Experimental local statevector execution for dynamic quantum circuits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

from ..circuit import Circuit
from ..compilation.routing import CouplingMap, route_to_topology
from ..core.ir import Instruction


@dataclass(frozen=True)
class DynamicExecutionResult:
    """Trajectory result containing final samples and classical registers."""

    samples: torch.Tensor
    classical_bits: torch.Tensor
    final_states: torch.Tensor
    shots: int
    seed: int | None
    execution_semantics: str = "local_statevector_trajectory"
    mid_circuit_measurements_available: bool = True
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_samples(self) -> torch.Tensor:
        return self.samples

    @property
    def classical_register(self) -> torch.Tensor:
        return self.classical_bits

    @property
    def mid_circuit_measurements(self) -> torch.Tensor | None:
        return (
            self.classical_bits
            if self.mid_circuit_measurements_available
            else None
        )

    def to_execution_result(self) -> Any:
        """Project dynamic shots into the canonical execution result contract."""

        from .result import ExecutionResult, MeasurementResult

        measurements = [
            MeasurementResult(
                "sample",
                tuple(range(self.samples.shape[-1])),
                self.samples,
                shots=self.shots,
                metadata={"stage": "final"},
            )
        ]
        if self.mid_circuit_measurements_available:
            measurements.append(
                MeasurementResult(
                    "mid_circuit_measurement",
                    tuple(range(self.classical_bits.shape[-1])),
                    self.classical_bits,
                    shots=self.shots,
                )
            )
        return ExecutionResult(
            samples=self.samples,
            measurements=tuple(measurements),
            runtime={
                "mode": self.execution_semantics,
                "shots": self.shots,
                "seed": self.seed,
                "mid_circuit_measurements_available": (
                    self.mid_circuit_measurements_available
                ),
            },
            provenance=dict(self.provider_metadata),
        )


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


class DynamicCircuit(Circuit):
    """Circuit builder for experimental mid-circuit measurement workflows."""

    def _append_dynamic(self, instruction: Instruction) -> "DynamicCircuit":
        if any(wire >= self.n_wires for wire in instruction.wires):
            raise ValueError("dynamic instruction wire is outside the circuit")
        self._instructions.append(instruction)
        self._state_cache = None
        self._ir_cache = None
        return self

    def measure(
        self,
        wire: int,
        *,
        classical_bit: int | None = None,
    ) -> "DynamicCircuit":
        bit = int(wire if classical_bit is None else classical_bit)
        if bit < 0:
            raise ValueError("classical_bit must be non-negative")
        return self._append_dynamic(
            Instruction(
                "measure",
                (int(wire),),
                metadata={"is_dynamic": True, "classical_bit": bit},
            )
        )

    def reset(self, wire: int) -> "DynamicCircuit":
        return self._append_dynamic(
            Instruction("reset", (int(wire),), metadata={"is_dynamic": True})
        )

    def conditional(
        self,
        name: str,
        wires: Iterable[int] | int,
        *,
        classical_bit: int | None = None,
        equals: int = 1,
        conditions: Mapping[int, int] | None = None,
        params: Mapping[str, Any] | None = None,
        matrix: Any | None = None,
    ) -> "DynamicCircuit":
        if conditions is not None and classical_bit is not None:
            raise ValueError("use either classical_bit or conditions, not both")
        raw_conditions = (
            {int(classical_bit): int(equals)}
            if classical_bit is not None
            else {int(bit): int(value) for bit, value in (conditions or {}).items()}
        )
        if not raw_conditions:
            raise ValueError("conditional gate requires at least one classical condition")
        if any(bit < 0 or value not in {0, 1} for bit, value in raw_conditions.items()):
            raise ValueError("conditions require non-negative bits and values 0 or 1")
        wire_tuple = (int(wires),) if isinstance(wires, int) else tuple(wires)
        return self._append_dynamic(
            Instruction(
                name,
                wire_tuple,
                params=dict(params or {}),
                matrix=matrix,
                metadata={"conditions": tuple(sorted(raw_conditions.items()))},
            )
        )

    def state(self, parameter_bindings: Mapping[Any, Any] | None = None) -> torch.Tensor:
        if any(
            instruction.metadata.get("is_dynamic")
            or instruction.metadata.get("condition")
            for instruction in self._instructions
        ):
            raise RuntimeError(
                "dynamic circuits require fq.experimental.run_dynamic(..., shots=...)"
            )
        return super().state(parameter_bindings)


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


def _instruction_conditions(instruction: Instruction) -> tuple[tuple[int, int], ...]:
    if "conditions" in instruction.metadata:
        return tuple(
            (int(bit), int(value))
            for bit, value in instruction.metadata["conditions"]
        )
    legacy = instruction.metadata.get("condition")
    if legacy:
        return ((int(legacy["bit"]), int(legacy["equals"])),)
    return ()


def _classical_width(circuit: DynamicCircuit) -> int:
    width = 0
    for instruction in circuit._instructions:
        if instruction.name == "measure":
            width = max(width, int(instruction.metadata["classical_bit"]) + 1)
        for bit, _value in _instruction_conditions(instruction):
            width = max(width, bit + 1)
    return width


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


def export_dynamic_qasm3(circuit: DynamicCircuit) -> str:
    """Export the supported dynamic subset to OpenQASM 3."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("dynamic QASM export requires DynamicCircuit")
    classical_width = _classical_width(circuit)
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{circuit.n_wires}] q;",
        *([f"bit[{classical_width}] c;"] if classical_width else []),
    ]
    for instruction in circuit._instructions:
        wire_text = ", ".join(f"q[{wire}]" for wire in instruction.wires)
        if instruction.name == "measure":
            bit = int(instruction.metadata["classical_bit"])
            statement = f"c[{bit}] = measure {wire_text};"
        elif instruction.name == "reset":
            statement = f"reset {wire_text};"
        else:
            params = ""
            if instruction.params:
                params = "(" + ", ".join(
                    str(value) for value in instruction.params.values()
                ) + ")"
            statement = f"{instruction.name}{params} {wire_text};"
        conditions = _instruction_conditions(instruction)
        if conditions:
            expression = " && ".join(
                f"c[{bit}] == {'true' if value else 'false'}"
                for bit, value in conditions
            )
            statement = (
                f"if ({expression}) {{ {statement} }}"
            )
        lines.append(statement)
    return "\n".join(lines) + "\n"


def _qasm_angle(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.requires_grad:
            raise ValueError("braket_iqm requires bound scalar gate parameters")
        value = value.detach().cpu().item()
    return repr(float(value))


def _iqm_qubit_groups(backend: Any) -> tuple[frozenset[int], ...]:
    metadata = getattr(backend, "metadata", {}) or {}
    raw_groups = metadata.get("dynamic_qubit_groups")
    if raw_groups is None:
        return ()
    return tuple(frozenset(int(wire) for wire in group) for group in raw_groups)


def export_braket_iqm_dynamic_qasm3(
    circuit: DynamicCircuit,
    *,
    qubit_groups: Iterable[Iterable[int]] | None = None,
) -> str:
    """Lower FlagQuantum feedback to IQM ``measure_ff``/``cc_prx`` QASM."""

    if not isinstance(circuit, DynamicCircuit):
        raise TypeError("Braket IQM dynamic export requires DynamicCircuit")
    groups = tuple(frozenset(int(wire) for wire in group) for group in (qubit_groups or ()))
    if not groups:
        raise ValueError("braket_iqm_dynamic_qubit_groups_are_required")

    latest_key: dict[int, int] = {}
    measured_wire: dict[int, int] = {}
    feed_forward_keys: set[int] = set()
    target_controller: dict[int, int] = {}
    next_key = 0
    body: list[str] = []

    def ensure_group(control: int, target: int) -> None:
        if not any(control in group and target in group for group in groups):
            raise ValueError("braket_iqm_feedback_pair_outside_dynamic_qubit_group")

    for instruction in circuit._instructions:
        conditions = _instruction_conditions(instruction)
        if instruction.name == "measure":
            if conditions:
                raise ValueError("braket_iqm_conditional_measurement_is_unsupported")
            bit = int(instruction.metadata["classical_bit"])
            key = next_key
            next_key += 1
            latest_key[bit] = key
            measured_wire[key] = instruction.wires[0]
            body.append(f"measure_ff({key}) ${instruction.wires[0]};")
            continue
        if instruction.name == "reset":
            if conditions:
                raise ValueError("braket_iqm_conditional_reset_is_unsupported")
            wire = instruction.wires[0]
            key = next_key
            next_key += 1
            measured_wire[key] = wire
            body.extend(
                (f"measure_ff({key}) ${wire};", f"cc_prx({math.pi!r}, 0.0, {key}) ${wire};")
            )
            feed_forward_keys.add(key)
            target_controller[wire] = wire
            continue
        if conditions:
            if len(conditions) != 1 or conditions[0][1] != 1:
                raise ValueError("braket_iqm_conditions_require_one_bit_equal_to_one")
            bit = conditions[0][0]
            if bit not in latest_key:
                raise ValueError("braket_iqm_feedback_bit_was_read_before_measurement")
            if len(instruction.wires) != 1 or instruction.name not in {"x", "rx"}:
                raise ValueError("braket_iqm_conditional_gate_cannot_lower_to_cc_prx")
            key = latest_key[bit]
            control = measured_wire[key]
            target = instruction.wires[0]
            ensure_group(control, target)
            previous = target_controller.setdefault(target, control)
            if previous != control:
                raise ValueError("braket_iqm_target_has_multiple_feedback_controllers")
            angle = math.pi if instruction.name == "x" else next(iter(instruction.params.values()))
            body.append(f"cc_prx({_qasm_angle(angle)}, 0.0, {key}) ${target};")
            feed_forward_keys.add(key)
            continue
        wires = ", ".join(f"${wire}" for wire in instruction.wires)
        if instruction.name == "x":
            body.append(f"prx({math.pi!r}, 0.0) {wires};")
        elif instruction.name == "rx":
            angle = next(iter(instruction.params.values()))
            body.append(f"prx({_qasm_angle(angle)}, 0.0) {wires};")
        elif instruction.name in {"rz", "cz"}:
            params = (
                "(" + ", ".join(_qasm_angle(value) for value in instruction.params.values()) + ")"
                if instruction.params
                else ""
            )
            body.append(f"{instruction.name}{params} {wires};")
        else:
            raise ValueError(f"braket_iqm_unsupported_native_gate:{instruction.name}")

    unused_keys = set(measured_wire) - feed_forward_keys
    if unused_keys:
        raise ValueError("braket_iqm_mid_circuit_measurement_requires_feed_forward")

    lines = [
        "OPENQASM 3.0;",
        f"bit[{circuit.n_wires}] b;",
        "#pragma braket verbatim",
        "box{",
        *(f"    {line}" for line in body),
        "}",
        *(f"b[{wire}] = measure ${wire};" for wire in range(circuit.n_wires)),
    ]
    return "\n".join(lines) + "\n"


def export_dynamic_qasm3_for_backend(circuit: DynamicCircuit, backend: Any) -> str:
    """Export using the backend-declared dynamic OpenQASM dialect."""

    dialect = getattr(backend, "dynamic_dialect", None) or "openqasm3"
    if dialect == "openqasm3":
        return export_dynamic_qasm3(circuit)
    if dialect == "braket_iqm":
        return export_braket_iqm_dynamic_qasm3(
            circuit,
            qubit_groups=_iqm_qubit_groups(backend),
        )
    raise ValueError(f"unsupported_dynamic_dialect:{dialect}")


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

    from ..deployment.cloud import DeploymentPackage
    from ..deployment.routing_evidence import (
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
