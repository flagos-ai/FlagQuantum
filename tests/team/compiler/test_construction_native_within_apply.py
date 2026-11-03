"""Native scoped arithmetic reuses checked register cleanup."""

import math
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    apply,
    hadamard_transform,
    phase,
    qfunc,
    within,
)

pytestmark = pytest.mark.unit


@qfunc
def mark(score: Const[QNum[1]]) -> None:
    phase(score, math.pi)


@qfunc
def scoped(x: Const[QBit], y: Const[QBit]) -> None:
    with within():
        score: QNum[1] = x * y
    with apply():
        mark(score)


@qfunc
def main(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    scoped(x, y)
    with within():
        score: QNum[1] = y
    with apply():
        mark(score)


QMOD = """qfunc mark(const score:qnum<1>) { phase(score,pi); }
qfunc scoped(const x:qbit,const y:qbit) {
    score:qnum<1>;
    within { score=x*y; } apply { mark(score); }
}
qfunc main(output x:qbit,output y:qbit) {
    allocate(x); allocate(y); H(x); H(y); scoped(x,y);
    score:qnum<1>;
    within { score=y; } apply { mark(score); }
}"""


def test_scoped_native_cleanup_matches_qmod_and_independent_amplitudes() -> None:
    for program in (
        synth.synthesize(main),
        synth.synthesize(QMOD, source_format="qmod"),
    ):
        state = fq.run(program).statevector().reshape(-1)
        expected = torch.zeros_like(state)
        scratch = program.circuit.n_wires - 2
        assert set(program.registers) == {"x", "y"}
        for x, y in product(range(2), repeat=2):
            expected[(2 * x + y) << scratch] = (-1) ** (x * y + y) / 2
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@qfunc
def mutate_dependency(x: Output[QBit]) -> None:
    allocate(x)
    with within():
        score: QNum[1] = x
    with apply():
        mark(score)
        hadamard_transform(x)


@qfunc
def escape_local(x: Output[QBit]) -> None:
    allocate(x)
    with within():
        score: QNum[1] = x
    with apply():
        mark(score)
    mark(score)


@pytest.mark.parametrize(
    ("definition", "reason"),
    [(mutate_dependency, "Const-only"), (escape_local, "declared quantum ports")],
)
def test_rejects_dependency_mutation_and_local_escape(definition, reason) -> None:
    with pytest.raises(HybridVerificationError, match=reason):
        capture_program(definition)


@qfunc
def weighted_cleanup(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    with within():
        score: QNum[4] = x * x + y
    with apply():
        phase(score, 0.8)


def test_weighted_local_score_cleans_value_and_phase_work() -> None:
    program = synth.synthesize(weighted_cleanup, max_qubits=24)
    state = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    scratch = program.circuit.n_wires - 4
    assert scratch >= 4
    for x, y in product(range(4), repeat=2):
        angle = 0.8 * (x * x + y)
        expected[(4 * x + y) << scratch] = complex(math.cos(angle), math.sin(angle)) / 4
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@qfunc
def dependent_phase(x: Const[QBit], y: Const[QBit]) -> None:
    with within():
        product_value: QNum[1] = x * y
        score: QNum[2] = product_value + y
    with apply():
        phase(score, 0.37)
        phase(1, -0.13)


@qfunc
def inverse_only(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    with adjoint():
        scoped(x, y)
        dependent_phase(x, y)


@qfunc
def round_trip(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    scoped(x, y)
    dependent_phase(x, y)
    with adjoint():
        scoped(x, y)
        dependent_phase(x, y)


@pytest.mark.parametrize("definition", [inverse_only, round_trip])
def test_checked_adjoint_of_local_and_nested_calls_has_exact_phase(definition) -> None:
    program = synth.synthesize(definition, max_qubits=16)
    state = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    scratch = program.circuit.n_wires - 2
    for x, y in product(range(2), repeat=2):
        angle = 0.0
        if definition is inverse_only:
            angle = -(math.pi * x * y + 0.37 * (x * y + y) - 0.13)
        expected[(2 * x + y) << scratch] = complex(math.cos(angle), math.sin(angle)) / 2
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@qfunc
def inverse_of_output_program(x: Output[QBit], y: Output[QBit]) -> None:
    with adjoint():
        main(x, y)


def test_adjoint_rejects_a_callable_that_allocates_outputs() -> None:
    with pytest.raises(HybridVerificationError, match="static Const-only"):
        capture_program(inverse_of_output_program)


@qfunc
def scheduled_phase(x: Output[QBit], y: Output[QBit]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    with within():
        score: QNum[3] = (x * y + x) * (x * y + y)
    with apply():
        phase(score, 0.37)


def test_cleanup_schedules_agree_on_state_and_report_different_widths() -> None:
    from flagquantum.compiler._construction.selection import select_register_program

    root, functions = capture_program(scheduled_phase)
    selection = select_register_program(root, functions)
    widths = set()
    for row in selection.rows:
        assert row.target is not None and row.arithmetic is not None
        widths.add(row.arithmetic.peak_workspace)
        circuit = row.target.executable
        state = fq.run(circuit).statevector().reshape(-1)
        expected = torch.zeros_like(state)
        scratch = circuit.n_wires - 2
        for x, y in product(range(2), repeat=2):
            angle = 0.37 * (x * y + x) * (x * y + y)
            expected[(2 * x + y) << scratch] = (
                complex(math.cos(angle), math.sin(angle)) / 2
            )
        torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert len(widths) == 2


@qfunc
def adjoint_in_apply(x: Output[QBit]) -> None:
    allocate(x)
    hadamard_transform(x)
    with within():
        score: QNum[1] = x
    with apply():
        mark(score)
        with adjoint():
            mark(score)


def test_apply_accepts_checked_inverse_of_a_local_borrow() -> None:
    program = synth.synthesize(adjoint_in_apply)
    state = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    expected[0] = expected[len(state) // 2] = 2**-0.5
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
