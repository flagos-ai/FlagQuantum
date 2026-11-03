"""Native scoped arithmetic reuses checked register cleanup."""

import math
from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.program_types import apply, within
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
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
