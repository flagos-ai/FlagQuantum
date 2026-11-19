"""Execute and independently check the original quantum programming exercises."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from random import Random

import torch

import flagquantum as fq
from flagquantum import synth

from . import quantum, reference

SHOTS = 1_000
BINDINGS = ((0.2, 1.0), (1.5, 0.5))


@dataclass(frozen=True)
class CheckedState:
    vector: torch.Tensor
    error: float
    workspace_probability: float


def compare(
    full: torch.Tensor, expected: torch.Tensor, wires: Sequence[int]
) -> CheckedState:
    """Compare absolute amplitudes and verify unobserved temporary bits are clean."""
    width = full.numel().bit_length() - 1
    workspace = tuple(wire for wire in range(width) if wire not in wires)
    aligned = full.reshape((2,) * width).permute((*wires, *workspace))
    columns = aligned.reshape(expected.numel(), -1)
    vector = columns[:, 0]
    leakage = float(columns[:, 1:].abs().square().sum())
    error = float((vector - expected).abs().max())
    if error > 1e-12 or leakage > 1e-24:
        raise ValueError(f"Full-state check failed: {error=}, {leakage=}")
    return CheckedState(vector, error, leakage)


def run_native(
    entry: str,
    expected: torch.Tensor,
    registers: tuple[str, ...],
    *,
    max_qubits: int,
    max_workspace: int = 0,
    angles: bool = False,
) -> tuple[synth.Program, fq.ExecutionResult, CheckedState]:
    """Synthesize under an explicit width and operation bound and execute once."""
    definitions = {
        "product_phase": quantum.product_phase,
        "solution_oracle": quantum.solution_oracle,
        "local_comparison": quantum.local_comparison,
        "fourier_add": quantum.fourier_add,
        "controlled_fourier_add": quantum.controlled_fourier_add,
        "pauli_evolution": quantum.pauli_evolution,
        "plus_pair": quantum.plus_pair,
        "estimate_phase": quantum.estimate_phase,
    }
    program = synth.synthesize(
        definitions[entry],
        classical_inputs=(
            {"angles": [math.pi * (1 << i) / 8 for i in range(6)]} if angles else None
        ),
        max_qubits=max_qubits,
        max_workspace=max_workspace,
        max_operations=180,
        memory_limit_bytes=1024**2,
    )
    result = fq.run(program)
    wires = tuple(wire for name in registers for wire in program.registers[name])
    return program, result, compare(result.statevector(), expected, wires)


def draw(
    probabilities: Mapping[tuple[int, ...], float],
    *,
    seed: int,
) -> Counter[tuple[int, ...]]:
    """Take finite seeded draws from the measured native-program distribution."""
    return Counter(
        Random(seed).choices(
            list(probabilities), weights=list(probabilities.values()), k=SHOTS
        )
    )


def read_counts(
    program: synth.Program,
    result: fq.ExecutionResult,
    registers: tuple[str, ...],
    *,
    seed: int,
) -> Counter[tuple[int, ...]]:
    return draw(dict(program.probabilities(result, registers=registers)), seed=seed)


def observe_plus() -> tuple[float, float]:
    program, _, checked = run_native(
        "plus_pair",
        torch.full((4,), 0.5, dtype=torch.complex128),
        ("qarr",),
        max_qubits=2,
    )
    observed = fq.run(
        program.circuit, outputs=fq.expectation(fq.X(0) + fq.X(1), name="plus")
    )
    energy = float(observed.expectation("plus").reshape(()))
    if abs(energy - 2) > 1e-12:
        raise ValueError("The plus state must give two")
    return energy, checked.error


def read_callback() -> tuple[Counter[tuple[int, ...]], float]:
    circuit = quantum.apply_to_all(fq.Circuit(3, dtype=torch.complex128), fq.Circuit.h)
    result = fq.run(circuit)
    checked = compare(
        result.statevector(),
        torch.full((8,), 1 / math.sqrt(8), dtype=torch.complex128),
        (0, 1, 2),
    )
    weights: dict[tuple[int, ...], float] = {
        (i,): float(z) for i, z in enumerate(checked.vector.abs().square())
    }
    return draw(weights, seed=18), checked.error


def read_qpe() -> tuple[Mapping[tuple[Fraction, ...], float], CheckedState]:
    program, result, checked = run_native(
        "estimate_phase", reference.qpe_state(), ("theta", "state"), max_qubits=6
    )
    return program.numeric_probabilities(result, registers=("theta",)), checked


def bound(
    template: fq.Circuit,
    alpha: float,
    beta: float,
) -> tuple[fq.Circuit, CheckedState]:
    """Bind a copy while retaining the original two symbolic angles."""
    circuit = template.bind_parameters({"alpha": alpha, "beta": beta})
    checked = compare(
        fq.run(circuit).statevector(), reference.ansatz_state(alpha, beta), (0, 1)
    )
    if circuit.parameter_names or template.parameter_names != ("alpha", "beta"):
        raise ValueError(
            "Each concrete circuit must leave the symbolic template intact"
        )
    return circuit, checked


def read_bindings(
    template: fq.Circuit,
) -> list[tuple[tuple[float, float], dict[str | int, int], float]]:
    rows = []
    for seed, (alpha, beta) in enumerate(BINDINGS, start=20):
        circuit, checked = bound(template, alpha, beta)
        counts = circuit.counts(SHOTS, generator=torch.Generator().manual_seed(seed))[0]
        rows.append(((alpha, beta), counts, checked.error))
    return rows


def observe_binding(template: fq.Circuit) -> tuple[float, float]:
    alpha, beta = math.pi / 4, math.pi / 3
    circuit, checked = bound(template, alpha, beta)
    observable = fq.X(0) + 0.5 * (fq.Z(0) @ fq.Z(1))
    result = fq.run(circuit, outputs=fq.expectation(observable, name="energy"))
    value = float(result.expectation("energy").reshape(()))
    expected = math.sin(alpha) * math.sin(beta) + 0.5 * math.cos(beta)
    if abs(value - expected) > 1e-12:
        raise ValueError("The expectation must match the independent analytic value")
    return value, checked.error
