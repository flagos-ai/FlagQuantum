"""Exact power-specific molecular QPE, complete phase and original model identity."""

import math
import re
from pathlib import Path
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.qpe_for_molecules import main as native
from examples.synthesis.applications.chemistry.qpe_for_molecules.comparisons.circuit import (
    pauli_rotation,
    reference_circuit,
)
from examples.synthesis.applications.chemistry.qpe_for_molecules.physics import (
    RUNS,
    STEPS,
    TERMS,
    ZERO_STATE,
    commute,
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
)

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/chemistry/qpe_for_molecules/qpe_for_molecules.qmod"
)


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def state(program: synth.Program, logical_size: int | None = None) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    if logical_size is None:
        assert program.report["workspace_qubits"] == 0
        return raw.flatten()
    viewed = raw.reshape(logical_size, -1)
    assert float(viewed[:, 1:].abs().square().sum()) < 1e-22
    return viewed[:, 0]


@pytest.mark.parametrize("scale,changed", [(1.0, False), (0.73, True)])
def test_native_source_and_independent_full_states(scale: float, changed: bool) -> None:
    amplitudes = (
        ZERO_STATE
        if not changed
        else tuple(
            ((-1) ** i) * (i + 1) / math.sqrt(sum(k * k for k in range(1, 65)))
            for i in range(64)
        )
    )
    program = native.build_program(scale, amplitudes)
    expected = fq.run(reference_circuit(scale, amplitudes)).statevector().flatten()
    torch.testing.assert_close(state(program), expected, atol=1e-10, rtol=0)
    assert len(program.registers["state"]) == 6 and len(program.registers["phase"]) == 7
    source = ORIGINAL.read_text()
    source = source.replace(
        "(-6.283185307179586) * p", repr(-2 * math.pi * scale) + " * p"
    )
    if changed:
        source = source.replace(
            "qperm prepare_basis_state", "qfunc prepare_basis_state"
        )
        source = source.replace(
            "allocate(6, arr);",
            "allocate(6, arr); inplace_prepare_amplitudes("
            + repr(list(amplitudes))
            + ", 0, arr);",
        )
    original = (
        synth.synthesize(ORIGINAL)
        if scale == 1 and not changed
        else synth.synthesize(source, source_format="qmod")
    )
    torch.testing.assert_close(state(original), expected, atol=1e-10, rtol=0)


def test_complete_model_commuting_partition_and_step_schedule() -> None:
    source = ORIGINAL.read_text()
    terms = []
    for body, coefficient in re.findall(
        r"SparsePauliTerm \{(.*?)coefficient=([-0-9.]+)\s*\}", source, re.S
    ):
        support = tuple(
            (int(bit), axis)
            for axis, bit in re.findall(r"pauli=Pauli::([XYZ]),\s*index=(\d+)", body)
        )
        terms.append((support, float(coefficient)))
    assert tuple(terms) == TERMS and len(TERMS) == 49
    assert tuple(term for group in RUNS for term in group) == TERMS
    assert tuple(map(len, RUNS)) == (10, 4, 4, 6, 6, 1, 7, 11)
    for group in RUNS:
        assert all(commute(a, b) for a in group for b in group)
    assert tuple(math.ceil(0.05 * (2**i) ** 1.5) for i in range(7)) == STEPS
    assert "ceiling(0.05 * (p ** 1.5))" in source and "qnum<7, True, 7>" in source
    # This regrouping saves effects without changing any noncommuting order.
    naive = sum(2 * len(TERMS) * n for n in STEPS)
    grouped = sum(
        (2 * len(TERMS) - len(RUNS[0]) - len(RUNS[-1])) * n + len(RUNS[0])
        for n in STEPS
    )
    assert naive > 4096 and grouped < 4096


@qfunc
def controlled_power(
    time: float,
    n: int,
    amplitudes: list[float],
    phi: float,
    undo: int,
    state: Output[QArray[QBit]],
    bit: Output[QBit],
) -> None:
    allocate(6, state)
    allocate(bit)
    inplace_prepare_amplitudes(amplitudes, 0, state)
    R(0.93, math.pi / 2, bit)
    phase(bit, phi)
    control(bit, lambda: native.power(time, n, state))
    if undo == 1:
        with adjoint():
            control(bit, lambda: native.power(time, n, state))


@pytest.mark.parametrize(
    "time,n,phi", [(-2 * math.pi, 1, 0.37), (-16 * math.pi, 2, -0.61)]
)
@pytest.mark.parametrize("undo", [False, True])
def test_controlled_power_keeps_identity_branch_phase_and_inverse(
    time: float, n: int, phi: float, undo: bool
) -> None:
    amplitudes = tuple(
        ((-1) ** i) * (i + 1) / math.sqrt(sum(k * k for k in range(1, 65)))
        for i in range(64)
    )
    program = synth.synthesize(
        controlled_power,
        classical_inputs={
            "time": time,
            "n": n,
            "amplitudes": list(amplitudes),
            "phi": phi,
            "undo": int(undo),
        },
        max_workspace=1,
    )
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    eye = torch.eye(64, dtype=torch.complex128)
    delta = eye[:, 0] - vector
    circuit = fq.Circuit(7, dtype=torch.complex128).any(
        *range(6), matrix=eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    )
    circuit.ry(6, theta=0.93).phase(6, theta=phi)
    if not undo:
        for _ in range(n):
            for term in TERMS:
                pauli_rotation(circuit, term, time / n / 2, 6)
            for term in reversed(TERMS):
                pauli_rotation(circuit, term, time / n / 2, 6)
    torch.testing.assert_close(
        state(program, 128), fq.run(circuit).statevector().flatten(), atol=1e-10, rtol=0
    )


@qfunc
def aliased(state: Output[QArray[QBit]]) -> None:
    allocate(6, state)
    control(state[0], lambda: native.power(-1.0, 1, state))


def test_callback_alias_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(aliased)


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0, float("nan"), float("inf"), True])
def test_invalid_scale_refuses(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        native.build_program(scale)


@pytest.mark.parametrize(
    "amplitudes",
    [
        (1.0,),
        tuple([0.0] * 64),
        (float("nan"),) + ZERO_STATE[1:],
        (True,) + ZERO_STATE[1:],
    ],
)
def test_invalid_state_refuses(amplitudes: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        native.build_program(amplitudes=amplitudes)


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
            native.main, classical_inputs=native.classical_inputs(), **limits
        )
