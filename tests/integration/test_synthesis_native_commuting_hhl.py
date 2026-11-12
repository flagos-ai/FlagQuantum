"""Native commuting HHL, signed raw selection and independent complete-state checks."""

import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.comparisons.circuit import (
    inversion_network,
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.data import (
    CONTROL_BITS,
    INVERSION_ANGLES,
    RHS,
    ROTATIONS,
)
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.main import (
    build_program,
    main,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    select_rotation,
)

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/quantum_linear_solvers/hhl/hhl.qmod"
)


def verify(
    program: synth.Program, expected: torch.Tensor, logical_size: int
) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.reshape(logical_size, -1)
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=1e-11, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    return state[:, 0]


@pytest.mark.parametrize(
    "rhs",
    [
        RHS,
        tuple(x / math.sqrt(30) for x in (1.0, 2.0, 3.0, 4.0)),
        tuple(x / math.sqrt(30) for x in (-2.0, 1.0, 4.0, 3.0)),
    ],
)
def test_hhl_complete_nonzero_rhs_and_signed_counter(rhs: tuple[float, ...]) -> None:
    program = build_program(rhs)
    result = fq.run(program)
    verify(program, fq.run(reference_circuit(rhs)).statevector(), 128)
    assert program.report["workspace_qubits"] == 0
    raw = program.probabilities(result, registers=("phase_var",))
    numeric = program.numeric_probabilities(result, registers=("phase_var",))
    assert dict(numeric) == {
        (Fraction(word[0] - 16 if word[0] >= 8 else word[0], 16),): p
        for word, p in raw.items()
    }
    assert any(
        value[0] < 0 and probability > 1e-8 for value, probability in numeric.items()
    )
    if rhs == RHS:
        verify(
            synth.synthesize(ORIGINAL), fq.run(reference_circuit()).statevector(), 128
        )


def test_retained_rotation_coefficients_and_parity_restoration() -> None:
    source = ORIGINAL.read_text()
    assert (
        tuple(float(x) for x in re.findall(r"RY\(([-0-9.]+), indicator\)", source))
        == ROTATIONS
    )
    assert (
        tuple(int(x) for x in re.findall(r"CX\(value\[([0-3])\], indicator\)", source))
        == CONTROL_BITS
    )
    for word in range(16):
        parity = 0
        angle = 0.0
        for theta, bit in zip(ROTATIONS, CONTROL_BITS, strict=True):
            angle += (-1 if parity else 1) * theta
            parity ^= (word >> bit) & 1
        assert parity == 0 and INVERSION_ANGLES[word] == angle


@qfunc
def inversion_only(
    theta: float,
    phi: float,
    counter: Output[Annotated[QNum, (4, True, 4)]],
    indicator: Output[QBit],
) -> None:
    allocate(counter)
    allocate(indicator)
    hadamard_transform(counter)
    R(theta, math.pi / 2, indicator)
    phase(indicator, phi)
    select_rotation("Y", INVERSION_ANGLES, counter, indicator)


@pytest.mark.parametrize("theta,phi", [(0.73, -0.41), (1.12, 0.37)])
def test_inversion_table_on_coherent_signed_words_and_complex_nonzero_target(
    theta: float, phi: float
) -> None:
    program = synth.synthesize(
        inversion_only, classical_inputs={"theta": theta, "phi": phi}
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.ry(4, theta=theta).phase(4, theta=phi)
    inversion_network(reference, (3, 2, 1, 0), 4)
    verify(program, fq.run(reference).statevector(), 32)


@qfunc
def inverse_rotation(
    counter: Output[Annotated[QNum, (4, True, 4)]], indicator: Output[QBit]
) -> None:
    allocate(counter)
    allocate(indicator)
    hadamard_transform(counter)
    H(indicator)
    phase(indicator, 0.37)
    select_rotation("Y", INVERSION_ANGLES, counter, indicator)
    with adjoint():
        select_rotation("Y", INVERSION_ANGLES, counter, indicator)


def test_checked_inverse_restores_all_coherent_words() -> None:
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.phase(4, theta=0.37)
    verify(synth.synthesize(inverse_rotation), fq.run(reference).statevector(), 32)


@pytest.mark.parametrize(
    "kind", ["qubits", "operations", "depth", "memory", "workspace"]
)
def test_hhl_budget_refusals(kind: str) -> None:
    inputs = {"amplitudes": list(RHS)}
    with pytest.raises((synth.SynthesisError, PlanningError, ValueError)):
        if kind == "qubits":
            synth.synthesize(main, classical_inputs=inputs, max_qubits=6)
        elif kind == "operations":
            synth.synthesize(main, classical_inputs=inputs, max_operations=1)
        elif kind == "depth":
            synth.synthesize(main, classical_inputs=inputs, max_depth=1)
        elif kind == "memory":
            synth.synthesize(main, classical_inputs=inputs, memory_limit_bytes=16)
        else:
            synth.synthesize(main, classical_inputs=inputs, max_workspace=-1)


@pytest.mark.parametrize(
    "rhs", [(), (1.0, 2.0), (0.0, 0.0, 0.0, 0.0), (math.nan, 1.0, 1.0, 1.0)]
)
def test_bad_rhs_refuses(rhs: tuple[float, ...]) -> None:
    with pytest.raises(
        (ValueError, synth.SynthesisError),
        match="amplitudes|vector|finite|normalization",
    ):
        build_program(rhs)
