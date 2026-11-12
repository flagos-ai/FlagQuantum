"""Native predicates phase coherent inputs through existing Compiler selection."""

import math
import operator
from typing import Literal

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
RELATIONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_relations_and_inputs_keep_absolute_sign_and_zero_work(
    relation: str, objective: Literal["width", "depth"], x: int, y: int
) -> None:
    source = f"qfunc main(output x:qnum<2>,output y:qnum<2>) {{x={x};y={y};phase(x{relation}y,pi);}}"
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(p, fq.run(p), {(x, y): (-1) ** int(RELATIONS[relation](x, y))})


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("helper", [False, True])
def test_nested_comparison_control_keeps_complex_reference_entanglement(
    objective: Literal["width", "depth"], helper: bool
) -> None:
    source = """qfunc mark(const a:qnum<2>,const b:qnum<2>,const e:qbit) {phase((a<b)*e,pi);}
    qfunc main(output x:qnum<2>,output y:qnum<2>,output e:qbit,output ref:qbit) {
      allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
      allocate(e);H(e);S(e);ref=e; ACTION
    }""".replace(
        "ACTION", "mark(x,y,e);" if helper else "phase((x<y)*e,pi);"
    )
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        p,
        fq.run(p),
        {
            (x, y, e, e): 1j**e * (-1) ** int(x < y and e) / math.sqrt(32)
            for x in range(4)
            for y in range(4)
            for e in range(2)
        },
    )


def test_direct_and_explicit_local_preserve_phase_under_measured_budgets() -> None:
    prefix = "qfunc main(output x:qnum<2>,output y:qnum<2>) {allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);"
    direct = prefix + "phase(x<y,pi);}"
    scoped = prefix + "f:qbit;within {f=x<y;} apply {phase(f,pi);}}"
    programs = [synth.synthesize(s, source_format="qmod") for s in (direct, scoped)]
    for p in programs:
        check_state(
            p,
            fq.run(p),
            {(x, y): (-1) ** int(x < y) / 4 for x in range(4) for y in range(4)},
        )
    direct_resources = programs[0].report["resources"]
    scoped_resources = programs[1].report["resources"]
    assert isinstance(direct_resources, ResourceEstimate)
    assert isinstance(scoped_resources, ResourceEstimate)
    assert direct_resources.n_qubits == 9
    assert scoped_resources.n_qubits == 6
    synth.synthesize(direct, source_format="qmod", max_qubits=9)
    synth.synthesize(scoped, source_format="qmod", max_qubits=6)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(scoped, source_format="qmod", max_qubits=5)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(direct, source_format="qmod", max_qubits=8)


def test_predicate_phase_is_visible_in_interference() -> None:
    source = "qfunc main(output q:qbit) {allocate(q);H(q);phase(q==1,pi);H(q);}"
    p = synth.synthesize(source, source_format="qmod")
    check_state(p, fq.run(p), {(1,): 1})
