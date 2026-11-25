"""Source W and custom W circuits, fresh Pauli decomposition, and resource sweeps."""

import math
from collections.abc import Sequence
from typing import Literal, cast

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import (
    ResourceEstimate,
    estimate_resources,
)
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .physics import Problem

Variant = Literal["proposed", "custom", "trotter"]
Term = tuple[str, float]


def controlled_rz(
    circuit: fq.Circuit, controls: Sequence[int], target: int, angle: float
) -> None:
    if not controls:
        circuit.rz(target, theta=angle)
        return
    size = 1 << len(controls)
    for i in range(size):
        gray = i ^ (i >> 1)
        circuit.rz(target, theta=(-1) ** gray.bit_count() * angle / size)
        next_i = (i + 1) % size
        changed = (gray ^ (next_i ^ (next_i >> 1))).bit_length() - 1
        circuit.cx(controls[changed], target)


def mcx(
    circuit: fq.Circuit, controls: Sequence[int], target: int, borrowed: Sequence[int]
) -> None:
    if len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        circuit.ccx(*controls, target)
    else:
        if len(controls) != 3 or not borrowed:
            raise ValueError(
                "source custom controls need at most three controls and one borrowed wire"
            )
        a, b, c = controls
        work = borrowed[0]
        for _ in range(2):
            circuit.ccx(a, b, work).ccx(work, c, target)


def custom_rz(
    circuit: fq.Circuit, controls: Sequence[int], target: int, angle: float
) -> None:
    if len(controls) < 2:
        controlled_rz(circuit, controls, target, angle)
        return
    split = len(controls) // 2
    first, second = controls[:split], controls[split:]
    for _ in range(2):
        mcx(circuit, first, target, second)
        circuit.rz(target, theta=-angle / 4)
        mcx(circuit, second, target, first)
        circuit.rz(target, theta=angle / 4)


def v_step(
    circuit: fq.Circuit, width: int, dx: float, steps: int, custom: bool
) -> None:
    wires = tuple(reversed(range(width)))
    angle = 1 / (dx * steps)
    for index in range(width):
        target = wires[index]
        for control in wires[:index]:
            circuit.cx(target, control)
        circuit.phase(target, theta=-math.pi / 2).h(target)
        if custom:
            custom_rz(circuit, wires[:index], target, angle)
        else:
            controlled_rz(circuit, wires[:index], target, angle)
        circuit.h(target).phase(target, theta=math.pi / 2)
        for control in reversed(wires[:index]):
            circuit.cx(target, control)


def paulis(problem: Problem) -> tuple[Term, ...]:
    matrix = (-1j * problem.derivative).to(torch.complex128)
    size = matrix.shape[0]
    indices = torch.arange(size)
    output = []
    for mask in range(size):
        coefficients = matrix[indices, indices ^ mask] / size
        step = 1
        while step < size:
            for start in range(0, size, 2 * step):
                first = coefficients[start : start + step].clone()
                second = coefficients[start + step : start + 2 * step].clone()
                coefficients[start : start + step] = first + second
                coefficients[start + step : start + 2 * step] = first - second
            step *= 2
        for index, value in enumerate(coefficients):
            if abs(complex(value)) <= 1e-12:
                continue
            coefficient = complex(value) * (1j) ** ((index & mask).bit_count())
            word = "".join(
                (
                    ("Y" if index >> bit & 1 else "X")
                    if mask >> bit & 1
                    else ("Z" if index >> bit & 1 else "I")
                )
                for bit in reversed(range(problem.width))
            )
            output.append((word, float(coefficient.real)))
    return tuple(output)


def append_term(circuit: fq.Circuit, word: str, angle: float) -> None:
    support = tuple(i for i, pauli in enumerate(word) if pauli != "I")
    for i in support:
        if word[i] == "Y":
            circuit.sdg(i).h(i)
        elif word[i] == "X":
            circuit.h(i)
    target = support[-1]
    for control in support[:-1]:
        circuit.cx(control, target)
    circuit.rz(target, theta=2 * angle)
    for control in reversed(support[:-1]):
        circuit.cx(control, target)
    for i in reversed(support):
        if word[i] == "Y":
            circuit.h(i).s(i)
        elif word[i] == "X":
            circuit.h(i)


@qfunc
def initial_state(
    bits: int, weights: list[float], target: Output[QArray[QBit]]
) -> None:
    allocate(bits, target)
    inplace_prepare_amplitudes(weights, 0, target)


def build(
    problem: Problem,
    variant: Variant = "proposed",
    steps: int = 10,
    prepare: bool = True,
) -> fq.Circuit:
    circuit = fq.Circuit(problem.width, dtype=torch.complex128)
    if prepare:
        initialized = synth.synthesize(
            initial_state,
            classical_inputs={
                "bits": problem.width,
                "weights": problem.initial.tolist(),
            },
            max_qubits=problem.width,
            max_workspace=0,
            memory_limit_bytes=1024**2,
        )
        circuit.compose(initialized.circuit)
    terms = paulis(problem) if variant == "trotter" else ()
    for _ in range(steps):
        if variant == "trotter":
            for word, coefficient in terms:
                append_term(circuit, word, coefficient / steps)
        else:
            v_step(circuit, problem.width, problem.dx, steps, variant == "custom")
    return circuit


def state(circuit: fq.Circuit) -> torch.Tensor:
    return cast(torch.Tensor, circuit.state()).reshape(-1)


def metrics(circuit: fq.Circuit) -> ResourceEstimate:
    lowered = convert_basis(
        circuit, gates=("u3", "cx"), max_added_operations=500_000
    ).program
    return estimate_resources(lowered)
