"""Prepare, select rotations, then uncompute the time-dependent encoding."""

import math

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.main import (
    evolution,
)
from flagquantum import synth
from flagquantum.synth import (
    PHASE,
    Output,
    QArray,
    QBit,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    qfunc,
    unitary,
    within,
)

from .model import INITIAL, Encoding


def build_program(encoding: Encoding, *, max_qubits: int = 7) -> synth.Program:
    """Specialize the original three-bit Taylor selector at one time."""
    encoding_matrix = encoding.householder
    norm = math.hypot(*INITIAL)
    initial_amplitudes = tuple(value / norm for value in INITIAL)

    @qfunc
    def oscillator(
        initial: list[float],
        x: Output[QBit],
        ancilla: Output[QArray[QBit]],
        y: Output[QBit],
    ) -> None:
        allocate(x)
        allocate(3, ancilla)
        allocate(y)
        inplace_prepare_amplitudes(initial, 0.01, x)
        with within():
            PHASE(math.pi, y)
            control(~y, lambda: unitary(encoding_matrix, ancilla))
        with apply():
            evolution(1.0, ancilla, x)

    return synth.synthesize(
        oscillator,
        classical_inputs={"initial": initial_amplitudes},
        max_qubits=max_qubits,
        max_workspace=2,
        max_operations=2048,
        memory_limit_bytes=1024**2,
    )


def logical_state(program: synth.Program) -> torch.Tensor:
    """Read the named logical ports in source order and verify clean workspace."""
    native: torch.Tensor = fq.run(program).statevector().flatten()
    width = native.numel().bit_length() - 1
    wires = tuple(
        wire for name in ("x", "ancilla", "y") for wire in program.registers[name]
    )
    temporary = tuple(wire for wire in range(width) if wire not in wires)
    aligned = native.reshape((2,) * width).permute((*wires, *temporary)).reshape(32, -1)
    if float(aligned[:, 1:].abs().square().sum()) > 1e-24:
        raise ValueError("The synthesis workspace must return to zero")
    return aligned[:, 0]
