"""Compose native Magnus steps, an amplifier, and a coherent success counter."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.main import (
    step,
)
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc

from .approximation import COMPLEX, REAL

SLICE_COUNT = 4
END = 2.0
AMPLIFIER = 7
COUNTER = (8, 9, 10)
BASE_WIDTH = 7
WIDTH = 11


@qfunc
def magnus(
    even: list[float],
    odd: list[float],
    start: float,
    end: float,
    index: Output[QArray[QBit]],
    time: Output[QArray[QBit]],
    block: Output[QBit],
    signal: Output[QBit],
    selector: Output[QBit],
) -> None:
    allocate(2, index)
    allocate(2, time)
    allocate(block)
    allocate(signal)
    allocate(selector)
    step(even, odd, start, end, index, time, block, signal, selector)


def build_slice(
    start: float,
    end: float,
    even: tuple[float, ...],
    odd: tuple[float, ...],
) -> fq.Circuit:
    program = synth.synthesize(
        magnus,
        classical_inputs={
            "start": start,
            "end": end,
            "even": [2 * x for x in even],
            "odd": [2 * x for x in odd],
        },
        max_qubits=BASE_WIDTH,
        max_workspace=0,
        max_operations=10_000,
        memory_limit_bytes=2**20,
    )
    return fq.Circuit.from_ir(program.circuit, dtype=COMPLEX)


def projector_phase(angle: float) -> torch.Tensor:
    """Phase the all-zero Magnus block with opposite signs on both signal branches."""
    basis = torch.arange(64)
    success = (basis >> 1) == 0
    positive = success == (basis % 2 == 0)
    signs = 2 * positive.to(REAL) - 1
    return torch.diag((1j * angle * signs).exp())


def counter_permutation(signal_bits: int) -> torch.Tensor:
    """Subtract one from the three-bit counter conditional on a zero signal word."""
    dimension = 1 << (signal_bits + len(COUNTER))
    columns = torch.arange(dimension)
    rows = columns.clone()
    rows[:8] = (columns[:8] - 1) % 8
    unitary = torch.zeros((dimension, dimension), dtype=COMPLEX)
    unitary[rows, columns] = 1
    return unitary


def assemble(
    slices: tuple[fq.Circuit, ...],
    amplitude_phases: tuple[float, ...] | None,
) -> fq.Circuit:
    """Start at counter four and decrement once on each successful slice."""
    if len(slices) != SLICE_COUNT or any(
        item.n_qubits != BASE_WIDTH for item in slices
    ):
        raise ValueError("Expected four original seven-bit Magnus slices")
    if amplitude_phases is not None and len(amplitude_phases) != 8:
        raise ValueError("The degree-seven amplifier requires eight angles")
    circuit = fq.Circuit(WIDTH, dtype=COMPLEX).h(0).h(1).x(COUNTER[0])
    block = tuple(range(2, 7 if amplitude_phases is None else 8))
    decrement = counter_permutation(len(block))
    for segment in slices:
        if amplitude_phases is None:
            circuit.compose(segment, qubits=tuple(range(BASE_WIDTH)))
        else:
            adjoint = segment.adjoint()
            circuit.h(AMPLIFIER)
            circuit.any(*block, matrix=projector_phase(amplitude_phases[0]))
            for order, angle in enumerate(amplitude_phases[1:]):
                circuit.compose(
                    segment if order % 2 == 0 else adjoint,
                    qubits=tuple(range(BASE_WIDTH)),
                )
                circuit.any(*block, matrix=projector_phase(angle))
            circuit.h(AMPLIFIER)
        circuit.any(*block, *COUNTER, matrix=decrement)
    return circuit


def exact_ode(times: torch.Tensor) -> torch.Tensor:
    index = torch.arange(4, dtype=REAL)
    return (torch.sin(times[:, None] + index) - torch.sin(index)).exp()


def expected_block(start: float, end: float) -> torch.Tensor:
    """Exact left-Riemann slice target for the four quantum indices."""
    coordinates = torch.arange(4, dtype=REAL)[:, None]
    samples = start + torch.arange(4, dtype=REAL) * (end - start) / 4
    return (coordinates + samples).cos().mean(dim=1)
