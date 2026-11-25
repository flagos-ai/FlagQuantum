"""Original-size bent-function hidden shift and exact binary reconstruction."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from sympy import GF, Matrix
from sympy.polys.matrices import DomainMatrix

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    X,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

Route = Literal["simple", "dual", "no_dual"]
SHIFT = (1, 3, 9)
SOURCE_SHOTS = 2048
_permutation = list(range(8))
random.Random(1).shuffle(_permutation)
PERMUTATION = tuple(_permutation)
INVERSE = tuple(PERMUTATION.index(i) for i in range(8))


@qfunc
def simple(s: Output[QArray[QBit]]) -> None:
    allocate(4, s)
    hadamard_transform(s)
    phase((1 - s[0]) * s[1] + s[2] * s[3], math.pi)
    hadamard_transform(s)
    phase(s[0] * s[1] + s[2] * s[3], math.pi)
    hadamard_transform(s)


@qfunc
def bent_phase(s: QArray[QBit], permutation: list[int]) -> None:
    for i in range(8):
        phase(s[i] * s[8 + permutation[i]], math.pi)
    phase(s[8] * s[9] * s[10] * s[11] * s[12] * s[13] * s[14] * s[15], math.pi)


@qfunc
def dual_phase(s: QArray[QBit], inverse: list[int]) -> None:
    for i in range(8):
        phase(s[inverse[i]] * s[8 + i], math.pi)
    phase(s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7], math.pi)


@qfunc
def dual(permutation: list[int], inverse: list[int], s: Output[QArray[QBit]]) -> None:
    allocate(16, s)
    hadamard_transform(s)
    X(s[1])
    X(s[3])
    X(s[9])
    bent_phase(s, permutation)
    X(s[1])
    X(s[3])
    X(s[9])
    hadamard_transform(s)
    dual_phase(s, inverse)
    hadamard_transform(s)


@qfunc
def bent_marker(s: QArray[QBit], ind: QBit, permutation: list[int]) -> None:
    for i in range(8):
        ind ^= s[i] & s[8 + permutation[i]]
    ind ^= s[8] & s[9] & s[10] & s[11] & s[12] & s[13] & s[14] & s[15]


@qfunc
def no_dual(
    permutation: list[int], target: Output[QArray[QBit]], ind: Output[QBit]
) -> None:
    allocate(16, target)
    allocate(ind)
    hadamard_transform(target)
    bent_marker(target, ind, permutation)
    phase(ind, math.pi)
    X(target[1])
    X(target[3])
    X(target[9])
    bent_marker(target, ind, permutation)
    X(target[1])
    X(target[3])
    X(target[9])
    hadamard_transform(target)


def build_program(route: Route) -> synth.Program:
    """Compile source dimensions with explicit physical and workspace limits."""
    if route == "simple":
        return synth.synthesize(
            simple, max_qubits=4, max_workspace=0, memory_limit_bytes=64 * 1024**2
        )
    if route == "dual":
        return synth.synthesize(
            dual,
            classical_inputs={
                "permutation": list(PERMUTATION),
                "inverse": list(INVERSE),
            },
            max_qubits=16,
            max_workspace=0,
            memory_limit_bytes=64 * 1024**2,
        )
    if route != "no_dual":
        raise ValueError("use simple, dual or no_dual")
    return synth.synthesize(
        no_dual,
        classical_inputs={"permutation": list(PERMUTATION)},
        max_qubits=21,
        max_workspace=4,
        memory_limit_bytes=64 * 1024**2,
    )


def reference_circuit(route: Route) -> fq.Circuit:
    """Build independent local CZ/Toffoli oracles without Compiler lowering."""
    if route not in {"simple", "dual", "no_dual"}:
        raise ValueError("use simple, dual or no_dual")
    bits = 4 if route == "simple" else 16
    circuit = fq.Circuit(bits + int(route == "no_dual"), dtype=torch.complex128)
    for wire in range(bits):
        circuit.h(wire)
    if route == "simple":
        circuit.x(3).cz(3, 2).x(3).cz(1, 0)
        for wire in range(bits):
            circuit.h(wire)
        circuit.cz(3, 2).cz(1, 0)
    elif route == "dual":
        for bit in SHIFT:
            circuit.x(15 - bit)
        _reference_phase(circuit, dual_function=False)
        for bit in SHIFT:
            circuit.x(15 - bit)
        for wire in range(bits):
            circuit.h(wire)
        _reference_phase(circuit, dual_function=True)
    else:
        _reference_marker(circuit)
        circuit.z(16)
        for bit in SHIFT:
            circuit.x(15 - bit)
        _reference_marker(circuit)
        for bit in SHIFT:
            circuit.x(15 - bit)
    for wire in range(bits):
        circuit.h(wire)
    return circuit


def _reference_phase(circuit: fq.Circuit, *, dual_function: bool) -> None:
    for i in range(8):
        a, b = (
            (15 - INVERSE[i], 7 - i) if dual_function else (15 - i, 7 - PERMUTATION[i])
        )
        circuit.cz(a, b)
    diagonal = torch.ones(256, dtype=torch.complex128)
    diagonal[-1] = -1
    start = 8 if dual_function else 0
    circuit.any(*range(start, start + 8), matrix=torch.diag(diagonal))


def _reference_marker(circuit: fq.Circuit) -> None:
    for i in range(8):
        circuit.ccx(15 - i, 7 - PERMUTATION[i], 16)
    # One bounded 9-bit permutation; never construct a dense 17-bit oracle.
    permutation = torch.eye(512, dtype=torch.complex128)
    permutation[510, 510] = permutation[511, 511] = 0
    permutation[510, 511] = permutation[511, 510] = 1
    circuit.any(*range(8), 16, matrix=permutation)


def recover_shift(
    samples: Sequence[tuple[Sequence[int], int]], *, bits: int = 16
) -> tuple[int, ...]:
    """Select independent measured rows and solve A*s=b exactly over GF(2)."""
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError("use 1..16 binary variables")
    rows: list[list[int]] = []
    rhs: list[int] = []
    for vector, indicator in samples:
        if (
            len(vector) != bits
            or any(type(bit) is not int or bit not in (0, 1) for bit in vector)
            or type(indicator) is not int
            or indicator not in (0, 1)
        ):
            raise ValueError("samples must contain binary vectors and indicators")
        candidate = [*rows, list(vector)]
        if len(rows) < bits and DomainMatrix.from_list(candidate, GF(2)).rank() > len(
            rows
        ):
            rows.append(list(vector))
            rhs.append(indicator)
    if len(rows) != bits:
        raise ValueError("insufficient independent samples to recover the shift")
    answer = tuple(int(value) % 2 for value in Matrix(rows).inv_mod(2) * Matrix(rhs))
    if any(
        sum(a * b for a, b in zip(vector, answer, strict=True)) % 2 != indicator
        for vector, indicator in samples
    ):
        raise ValueError("inconsistent binary equations")
    return answer


@dataclass(frozen=True)
class Estimate:
    """Actual observations, recovered shift and full physical-state evidence."""

    route: Route
    seed: int
    shots: int
    counts: tuple[tuple[int, int], ...]
    recovered: tuple[int, ...]
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


def run_estimate(
    route: Route, *, shots: int = SOURCE_SHOTS, seed: int = 1500
) -> Estimate:
    """Check the complete state, sample 2048 source shots and recover the shift."""
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("use 1..100000 shots")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("use an integer seed in 0..2**32-1")
    program = build_program(route)
    result = fq.run(program)
    state = result.statevector().flatten()
    expected = fq.run(reference_circuit(route)).statevector().flatten()
    logical_bits = 4 if route == "simple" else 17 if route == "no_dual" else 16
    workspace = program.circuit.n_wires - logical_bits
    ordered = state.reshape(1 << logical_bits, 1 << workspace)
    error = float((ordered[:, 0] - expected).abs().max())
    torch.testing.assert_close(ordered[:, 0], expected, atol=2e-11, rtol=0)
    dirty = float(ordered[:, 1:].abs().square().sum())
    if dirty > 1e-24:
        raise AssertionError("hidden-shift workspace did not return to zero")
    registers = ("target", "ind") if route == "no_dual" else ("s",)
    probabilities = dict(
        program.probabilities(result, registers=registers, max_bits=17, max_keys=131072)
    )
    dense = torch.zeros(1 << logical_bits, dtype=torch.float64)
    for labels, probability in probabilities.items():
        word = 2 * labels[0] + labels[1] if route == "no_dual" else labels[0]
        dense[word] = probability
    draws = torch.multinomial(
        dense, shots, replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    independent_draws = torch.multinomial(
        expected.abs().square(),
        shots,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    if not torch.equal(draws, independent_draws):
        raise AssertionError("same-seed independent Circuit observations disagree")
    counts = tuple(sorted(Counter(int(word) for word in draws).items()))
    bits = 4 if route == "simple" else 16
    if route == "no_dual":
        equations = [
            (tuple((word // 2) >> bit & 1 for bit in range(bits)), word & 1)
            for word, _ in counts
        ]
        recovered = recover_shift(equations)
    else:
        recovered = tuple(counts[0][0] >> bit & 1 for bit in range(bits))
    expected_shift = tuple(
        int(bit in ((0,) if route == "simple" else SHIFT)) for bit in range(bits)
    )
    if recovered != expected_shift:
        raise AssertionError(
            "measured hidden shift disagrees with the source assertion"
        )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Estimate(route, seed, shots, counts, recovered, error, dirty, resources)
