"""Check weighted coherent phase against the fourth-section amplitude oracle."""

import cmath

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


@qfunc
def cost(x: Const[QNum[2]], y: Const[QNum[2]]) -> None:
    phase(x * x + y, 0.8)


@qfunc
def main(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    cost(x, y)
    control(x == 1, lambda: phase(y, 0.3))


def test_weighted_control_composes_with_arithmetic_phase():
    program = synth.synthesize(main, max_qubits=24, max_workspace=20)
    expected = {
        (x, y): cmath.exp(1j * (0.8 * (x * x + y) + (0.3 * y if x == 1 else 0))) / 4
        for x in range(4)
        for y in range(4)
    }
    check_state(program, fq.run(program), expected)


def test_qmod_controlled_weighted_phase():
    source = """qfunc main(output x: qnum, output y: qnum) {
        allocate(2, x); allocate(2, y);
        hadamard_transform(x); hadamard_transform(y);
        control (x == 1) { phase(y, 0.3); }
    }"""
    program = synth.synthesize(source, source_format="qmod")
    expected = {
        (x, y): cmath.exp(0.3j * y) / 4 if x == 1 else 0.25
        for x in range(4)
        for y in range(4)
    }
    check_state(program, fq.run(program), expected)


@pytest.mark.parametrize(
    "effect", ["allocate(t);", "measure(t);", "release(t);", "consume(t);"]
)
def test_control_rejects_branch_local_effects(effect):
    source = f"qfunc main(output q: qbit, output t: qbit) {{ allocate(q); control(q) {{ {effect} }} }}"
    with pytest.raises(ValueError, match="expected a scalar Hadamard or phase"):
        synth.synthesize(source, source_format="qmod")


def test_weighted_control_budget_counts_both_lowered_phase_gates():
    source = """qfunc main(output q: qbit, output y: qnum) {
        allocate(q); allocate(2, y);
        control(q) { phase(y, 0.3); }
    }"""
    program = synth.synthesize(source, source_format="qmod", max_operations=2)
    assert len(program.circuit.instructions) == 2
    with pytest.raises(ValueError, match="operation"):
        synth.synthesize(source, source_format="qmod", max_operations=1)
