"""Full Poisson states, spectral phase, spatial transforms and input validation."""

import ast
import math
import re
from pathlib import Path
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver import (
    main as native,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.comparisons.circuit import (
    reference_circuit,
    spatial_transforms,
    spectrum,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.physics import (
    AMPLITUDES,
    ANGLES,
    COSINE,
    HAMILTONIAN,
    SINE,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    R,
    adjoint,
    allocate,
    control,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
    suzuki_trotter,
    unitary,
)

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/discrete_poisson_solver.qmod"
)


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def state(program: synth.Program, size: int = 8192) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    viewed = raw.reshape(size, -1)
    assert float(viewed[:, 1:].abs().square().sum()) < 1e-22
    return viewed[:, 0]


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_native_frozen_and_independent_full_states(variant: int) -> None:
    amplitudes = (
        tuple(AMPLITUDES)
        if variant == 0
        else tuple(
            ((-1) ** (i + variant))
            * (i + variant + 1)
            / math.sqrt(sum((j + variant + 1) ** 2 for j in range(64)))
            for i in range(64)
        )
    )
    program = native.build_program(amplitudes)
    expected = fq.run(reference_circuit(amplitudes)).statevector().flatten()
    torch.testing.assert_close(state(program), expected, atol=1e-10, rtol=0)
    assert (
        len(program.registers["xy"]) == 6 and len(program.registers["phase_var"]) == 6
    )
    assert program.report["workspace_qubits"] == 0
    text = ORIGINAL.read_text()
    match = re.search(r"prepare_amplitudes\((\[.*?\]), 0.0, xy_variable", text, re.S)
    assert match
    assert tuple(ast.literal_eval(match[1])) == tuple(AMPLITUDES)
    if variant:
        text = text[: match.start(1)] + repr(list(amplitudes)) + text[match.end(1) :]
    source = (
        synth.synthesize(ORIGINAL)
        if variant == 0
        else synth.synthesize(text, source_format="qmod")
    )
    torch.testing.assert_close(state(source), expected, atol=1e-10, rtol=0)


def test_retained_table_operator_and_independent_spectrum() -> None:
    text = ORIGINAL.read_text()
    table = re.search(r"select_rotation\(Pauli::Y, (\[.*?\]), index", text, re.S)
    assert table and tuple(ast.literal_eval(table[1])) == ANGLES and len(ANGLES) == 64
    terms = []
    for axes, coefficient in re.findall(
        r"PauliTerm \{\s*pauli=(\[.*?\]),\s*coefficient=([-0-9.]+)", text, re.S
    ):
        support = tuple(
            (i, axis.lower())
            for i, axis in enumerate(re.findall(r"Pauli::([IXYZ])", axes))
            if axis != "I"
        )
        terms.append((support, float(coefficient)))
    assert tuple(
        (term.factors, term.coefficient) for term in HAMILTONIAN.terms
    ) == tuple(terms)
    assert len(terms) == 16 and sum(not support for support, _ in terms) == 2
    observed = torch.tensor(
        [
            math.fsum(
                coefficient * ((-1) ** sum((word >> i) & 1 for i, _ in support))
                for support, coefficient in terms
            )
            for word in range(64)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(observed, spectrum(), atol=1e-13, rtol=0)
    for native_matrix, reference in zip(
        (SINE, COSINE), spatial_transforms(), strict=True
    ):
        torch.testing.assert_close(
            torch.tensor(native_matrix, dtype=torch.complex128),
            reference,
            atol=1e-14,
            rtol=0,
        )


@qfunc
def transforms(
    amplitudes: list[float], phi: float, undo: int, xy: Output[QArray[QBit]]
) -> None:
    allocate(6, xy)
    inplace_prepare_amplitudes(amplitudes, 0, xy)
    for i in range(6):
        phase(xy[i], phi * (i + 1))
    unitary(SINE, xy[:3])
    unitary(COSINE, xy[3:])
    if undo == 1:
        with adjoint():
            unitary(SINE, xy[:3])
            unitary(COSINE, xy[3:])


@pytest.mark.parametrize("phi", [0.37, -0.61])
@pytest.mark.parametrize("undo", [False, True])
def test_spatial_transform_bit_order_and_inverse(phi: float, undo: bool) -> None:
    amplitudes = tuple(
        (i + 1) / math.sqrt(sum(j * j for j in range(1, 65))) for i in range(64)
    )
    program = synth.synthesize(
        transforms,
        classical_inputs={
            "amplitudes": list(amplitudes),
            "phi": phi,
            "undo": int(undo),
        },
    )
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    angles = torch.tensor(
        [sum(((word >> i) & 1) * (i + 1) for i in range(6)) for word in range(64)],
        dtype=torch.float64,
    )
    expected = vector * torch.exp(1j * phi * angles)
    if not undo:
        sine, cosine = spatial_transforms()
        expected = torch.kron(cosine, sine) @ expected
    torch.testing.assert_close(state(program, 64), expected, atol=1e-11, rtol=0)


@qfunc
def propagate(time: float, xy: QArray[QBit]) -> None:
    suzuki_trotter(HAMILTONIAN, time, 1, 1, xy)


@qfunc
def evolution(
    amplitudes: list[float],
    time: float,
    undo: int,
    xy: Output[QArray[QBit]],
    bit: Output[QBit],
) -> None:
    allocate(6, xy)
    allocate(bit)
    inplace_prepare_amplitudes(amplitudes, 0, xy)
    R(0.93, math.pi / 2, bit)
    phase(bit, 0.37)
    control(bit, lambda: propagate(time, xy))
    if undo == 1:
        with adjoint():
            control(bit, lambda: propagate(time, xy))


@pytest.mark.parametrize("time", [-0.73, -2 * math.pi * 0.101321183642338])
@pytest.mark.parametrize("undo", [False, True])
def test_controlled_diagonal_evolution_keeps_identity_phase(
    time: float, undo: bool
) -> None:
    program = synth.synthesize(
        evolution,
        classical_inputs={"amplitudes": AMPLITUDES, "time": time, "undo": int(undo)},
    )
    vector = torch.tensor(AMPLITUDES, dtype=torch.complex128)
    expected = torch.stack(
        (
            vector * math.cos(0.93 / 2),
            vector * math.sin(0.93 / 2) * complex(math.cos(0.37), math.sin(0.37)),
        ),
        dim=1,
    )
    if not undo:
        expected[:, 1] *= torch.exp(-1j * time * spectrum())
    torch.testing.assert_close(
        state(program, 128), expected.flatten(), atol=1e-11, rtol=0
    )


@qfunc
def aliased(xy: Output[QArray[QBit]]) -> None:
    allocate(6, xy)
    control(xy[0], lambda: propagate(-1.0, xy))


def test_control_alias_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(aliased)


@pytest.mark.parametrize(
    "amplitudes",
    [
        (1.0,),
        tuple([0.0] * 64),
        (float("nan"),) + tuple(AMPLITUDES[1:]),
        (True,) + tuple(AMPLITUDES[1:]),
    ],
)
def test_invalid_spatial_input_refuses(amplitudes: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        native.build_program(amplitudes)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 12},
        {"max_operations": 100},
        {"max_depth": 100},
        {"memory_limit_bytes": 1024},
    ],
)
def test_resource_budgets_refuse(limits: Limits) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError),
        match="budget|limit|requires|exceed|infeasible",
    ):
        synth.synthesize(
            native.main, classical_inputs={"amplitudes": AMPLITUDES}, **limits
        )
