"""Native Bloch and repeated Z/ZZ feature maps and their overlap circuits."""

import math
from enum import Enum
from typing import Literal

import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.qsvm.main import encode_on_bloch
from flagquantum import synth
from flagquantum.synth import (
    RZ,
    RZZ,
    H,
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    qfunc,
)

Point = tuple[float, float]


class FeatureMap(Enum):
    BLOCH = "Bloch"
    PAULI = "Pauli Z/ZZ"


@qfunc
def bloch_state(data: list[float], q: Output[QBit]) -> None:
    allocate(q)
    encode_on_bloch(data, q)


@qfunc
def pauli_encode(angles: list[float], q: QArray[QBit]) -> None:
    for repeat in range(2):
        H(q[0])
        H(q[1])
        RZ(angles[0], q[0])
        RZ(angles[1], q[1])
        RZZ(angles[2], q)


@qfunc
def pauli_state(angles: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    pauli_encode(angles, q)


@qfunc
def bloch_overlap(a: list[float], b: list[float], q: Output[QBit]) -> None:
    allocate(q)
    encode_on_bloch(a, q)
    with adjoint():
        encode_on_bloch(b, q)


@qfunc
def pauli_overlap(a: list[float], b: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    pauli_encode(a, q)
    with adjoint():
        pauli_encode(b, q)


def bindings(kind: FeatureMap, point: Point) -> list[float]:
    """Map the source affine Pauli coefficients to rotation angles."""
    x, y = point
    if not all(math.isfinite(value) for value in point):
        raise ValueError("Each feature coordinate must be finite")
    if kind is FeatureMap.BLOCH:
        return [x, y]
    if kind is FeatureMap.PAULI:
        return [2 * x, 2 * y, 2 * (x - math.pi) * (y - math.pi)]
    raise ValueError("Unknown feature map")


def build(kind: FeatureMap, a: Point, b: Point | None = None) -> synth.Program:
    """Bind a single feature state or the source overlap U(b)† U(a)."""
    width: Literal[1, 2] = 1 if kind is FeatureMap.BLOCH else 2
    values = bindings(kind, a)
    if b is None:
        root = bloch_state if kind is FeatureMap.BLOCH else pauli_state
        classical = {"data" if kind is FeatureMap.BLOCH else "angles": values}
    else:
        root = bloch_overlap if kind is FeatureMap.BLOCH else pauli_overlap
        classical = {"a": values, "b": bindings(kind, b)}
    return synth.synthesize(
        root,
        classical_inputs=classical,
        max_qubits=width,
        max_workspace=0,
        max_operations=64,
        memory_limit_bytes=1024**2,
    )


def state(kind: FeatureMap, point: Point) -> torch.Tensor:
    statevector: torch.Tensor = fq.run(build(kind, point)).statevector().flatten()
    return statevector


def independent_circuit(kind: FeatureMap, point: Point) -> fq.Circuit:
    """Compose the Bloch rotations or an independently exponentiated diagonal."""
    x, y = point
    if kind is FeatureMap.BLOCH:
        return (
            fq.Circuit(1, dtype=torch.complex128).rx(0, math.pi * x).rz(0, math.pi * y)
        )
    coupling = (x - math.pi) * (y - math.pi)
    # qfunc array index 0 is the least-significant data wire.
    energies = torch.tensor(
        [x + y + coupling, -x + y - coupling, x - y - coupling, -x - y + coupling],
        dtype=torch.float64,
    )
    diagonal = torch.diag(torch.exp(-1j * energies))
    circuit = fq.Circuit(2, dtype=torch.complex128)
    for _ in range(2):
        circuit.h(0).h(1).any(0, 1, matrix=diagonal)
    return circuit
