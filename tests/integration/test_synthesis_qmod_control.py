"""Check a Qmod control region and native bounded comparison against an oracle."""

import math

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration

SOURCE = """qfunc main(output x: qnum, output target: qbit) {
  allocate(2, x);
  allocate(target);
  hadamard_transform(x);
  hadamard_transform(target);
  control(x < 2) { phase(target, pi); }
}"""


@qfunc
def native(x: Output[QNum], target: Output[QBit]) -> None:
    allocate(2, x)
    allocate(target)
    hadamard_transform(x)
    hadamard_transform(target)
    control(x < 2, lambda: phase(target, math.pi))


@pytest.mark.parametrize("source", [SOURCE, native])
def test_predicate_state_matches_oracle_and_cleans_workspace(source):
    options = {"source_format": "qmod"} if isinstance(source, str) else {}
    program = synth.synthesize(source, **options)
    expected = {
        (x, t): (-1 if x < 2 and t else 1) / math.sqrt(8)
        for x in range(4)
        for t in range(2)
    }
    check_state(program, fq.run(program), expected)
    with pytest.raises(ValueError, match="qubit|budget|workspace|width"):
        synth.synthesize(source, max_qubits=3, **options)


def test_qmod_scalar_controlled_global_phase_interferes():
    source = """qfunc main(output q: qbit) {
      allocate(q);
      H(q);
      control(q) { phase(1, pi); }
      H(q);
    }"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(1,): 1})


@pytest.mark.parametrize(
    "body",
    [
        "control(q == 1) { H(q); }",
        "control(q) { control(q) { phase(1, pi); } }",
    ],
)
def test_qmod_control_refuses_overlap_and_nesting(body):
    source = f"qfunc main(output q: qbit) {{ allocate(q); {body} }}"
    with pytest.raises(ValueError, match="disjoint|function scope"):
        synth.synthesize(source, source_format="qmod")
