"""Provider-neutral capability and conformance contracts for dynamic circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ._conditions import classical_width as _classical_width
from ._conditions import instruction_conditions as _instruction_conditions
from .circuit import DynamicCircuit
from .execution import run_dynamic
from .result import DynamicExecutionResult


@dataclass(frozen=True)
class DynamicFeatureSet:
    """Declarative dynamic-circuit features implemented by one execution path."""

    name: str
    mid_circuit_measurement: bool = True
    reset: bool = True
    supported_conditional_gates: tuple[str, ...] = ()
    max_condition_bits: int | None = None
    condition_values: tuple[int, ...] = (0, 1)
    returns_mid_circuit_measurements: bool = True
    supports_repeated_measurement: bool = True


@dataclass(frozen=True)
class DynamicFeatureReport:
    compatible: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class DynamicConformanceCase:
    name: str
    circuit: DynamicCircuit
    expected_sample: tuple[int, ...]
    expected_classical: tuple[int, ...]


@dataclass(frozen=True)
class DynamicConformanceResult:
    implementation: str
    passed: bool
    cases: tuple[tuple[str, bool], ...]


LOCAL_TRAJECTORY_FEATURES = DynamicFeatureSet(
    name="flagquantum_local_trajectory",
    supported_conditional_gates=(
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "cx",
        "cz",
        "swap",
    ),
)

QISKIT_AER_DYNAMIC_FEATURES = DynamicFeatureSet(
    name="qiskit_aer",
    supported_conditional_gates=LOCAL_TRAJECTORY_FEATURES.supported_conditional_gates,
)

BRAKET_IQM_DYNAMIC_FEATURES = DynamicFeatureSet(
    name="braket_iqm",
    supported_conditional_gates=("x", "rx"),
    max_condition_bits=1,
    condition_values=(1,),
    returns_mid_circuit_measurements=False,
)


def assess_dynamic_features(
    circuit: DynamicCircuit,
    features: DynamicFeatureSet,
) -> DynamicFeatureReport:
    """Assess circuit semantics independently of provider transport details."""

    blockers: list[str] = []
    measured: set[int] = set()
    for instruction in circuit._instructions:
        conditions = _instruction_conditions(instruction)
        if instruction.name == "measure":
            if not features.mid_circuit_measurement:
                blockers.append("mid_circuit_measurement_unsupported")
            bit = int(instruction.metadata["classical_bit"])
            if bit in measured and not features.supports_repeated_measurement:
                blockers.append("repeated_measurement_unsupported")
            measured.add(bit)
        if instruction.name == "reset" and not features.reset:
            blockers.append("reset_unsupported")
        if conditions:
            if (
                features.max_condition_bits is not None
                and len(conditions) > features.max_condition_bits
            ):
                blockers.append("condition_width_exceeds_limit")
            if any(value not in features.condition_values for _, value in conditions):
                blockers.append("condition_value_unsupported")
            if instruction.name not in features.supported_conditional_gates:
                blockers.append(f"conditional_gate_unsupported:{instruction.name}")
            if any(bit not in measured for bit, _ in conditions):
                blockers.append("classical_bit_read_before_measurement")
    return DynamicFeatureReport(not blockers, tuple(dict.fromkeys(blockers)))


def dynamic_conformance_cases() -> tuple[DynamicConformanceCase, ...]:
    reset = DynamicCircuit(1).x(0).reset(0)
    feedback = DynamicCircuit(2).x(0).measure(0, classical_bit=0)
    feedback.conditional("x", 1, classical_bit=0)
    repeated = DynamicCircuit(1).x(0).measure(0, classical_bit=0)
    repeated.reset(0).measure(0, classical_bit=0)
    return (
        DynamicConformanceCase("active_reset", reset, (0,), ()),
        DynamicConformanceCase("conditional_flip", feedback, (1, 1), (1,)),
        DynamicConformanceCase("qubit_reuse", repeated, (0,), (0,)),
    )


def run_dynamic_conformance(
    executor: Callable[..., DynamicExecutionResult] = run_dynamic,
    *,
    implementation: str = "flagquantum_local_trajectory",
    shots: int = 8,
) -> DynamicConformanceResult:
    """Run deterministic semantic vectors against a dynamic executor."""

    outcomes = []
    for index, case in enumerate(dynamic_conformance_cases()):
        result = executor(case.circuit, shots=shots, seed=100 + index)
        samples = result.samples.reshape(-1, case.circuit.n_wires)
        width = _classical_width(case.circuit)
        classical = (
            result.classical_bits.reshape(-1, width)
            if width
            else result.classical_bits.reshape(shots, 0)
        )
        passed = bool(torch.all(samples == torch.tensor(case.expected_sample)).item())
        if case.expected_classical:
            passed = passed and bool(
                torch.all(classical == torch.tensor(case.expected_classical)).item()
            )
        outcomes.append((case.name, passed))
    return DynamicConformanceResult(
        implementation=implementation,
        passed=all(passed for _, passed in outcomes),
        cases=tuple(outcomes),
    )


__all__ = (
    "BRAKET_IQM_DYNAMIC_FEATURES",
    "DynamicConformanceCase",
    "DynamicConformanceResult",
    "DynamicFeatureReport",
    "DynamicFeatureSet",
    "LOCAL_TRAJECTORY_FEATURES",
    "QISKIT_AER_DYNAMIC_FEATURES",
    "assess_dynamic_features",
    "dynamic_conformance_cases",
    "run_dynamic_conformance",
)
