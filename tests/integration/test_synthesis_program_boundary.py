"""UP-00 compatibility uses existing complete workflows, not a new DSL engine."""

from typing import Literal

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from tests.synthesis_workflow_helpers import check_state
from tests.team.compiler.test_construction_python import _cell

pytestmark = pytest.mark.integration


def energy(x: int, y: int) -> int:
    return x * x + y


@synth.qfunc
def captured_energy(x: int, y: int) -> int:
    return x * x + y


ARITHMETIC_QMOD = """qfunc energy(const x:qnum<2>,const y:qnum<2>,output z:qnum) {
    z=x*x+y;
}
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}"""


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_frontends_replace_each_other_for_same_named_coherent_consumer(
    objective: Literal["width", "depth"],
) -> None:
    expected = {(x, y, x * x + y): 0.25 for x in range(4) for y in range(4)}
    for producer in (energy, captured_energy, ARITHMETIC_QMOD):
        options = (
            {"source_format": "qmod"}
            if isinstance(producer, str)
            else {
                "inputs": {"x": synth.UInt(2), "y": synth.UInt(2)},
                "input_state": synth.InputState.uniform(),
            }
        )
        p = synth.synthesize(
            producer,
            **options,
            objective=objective,
            max_qubits=24,
            memory_limit_bytes=2**30,
        )
        result = fq.run(p)
        assert isinstance(p, synth.Program)
        assert tuple(p.registers) == ("x", "y", "result")
        assert len(p.registers["result"]) == 4
        check_state(p, result, expected)
        assert dict(
            p.probabilities(result, registers=("x", "y", "result"))
        ) == pytest.approx(dict.fromkeys(expected, 1 / 16), abs=1e-12)


@pytest.mark.parametrize("annotation", ["bool", "int", ""])
def test_existing_predicate_returns_still_encode_pi(annotation: str) -> None:
    hint = f" -> {annotation}" if annotation else ""
    function = _cell(f"def main(q: int){hint}:\n    return q==1\n")
    p = synth.synthesize(
        function,
        inputs={"q": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
    )
    check_state(p, fq.run(p), {(0,): 2**-0.5, (1,): -(2**-0.5)})
    assert tuple(p.registers) == ("q",)
    assert p.report["result_width"] == 0
    assert p.report["scope"] == "actual_cpu_preparation_predicate_phase_cleanup"


@pytest.mark.parametrize("mode", ["value", "phase"])
@pytest.mark.parametrize("width", [1, 2])
def test_withdrawn_radians_profile_is_refused_instead_of_reinterpreted(
    mode: Literal["value", "phase"], width: int
) -> None:
    function = _cell("def main(q: int) -> float:\n    return 1.9*(q-0.5)\n")
    with pytest.raises(HybridVerificationError, match="construction.python.annotation"):
        synth.synthesize(
            function,
            inputs={"q": synth.UInt(width)},
            input_state=synth.InputState.basis(q=0),
            output_mode=mode,
        )


@pytest.mark.parametrize(
    "source",
    [
        # Input is consumption, not a synonym for Const.
        "qfunc f(input q:qbit) {} qfunc main(output q:qbit) {allocate(q);f(q);}",
        # A live borrowed value cannot be silently overwritten.
        "qfunc f(const q:qnum<2>) {q=1;} qfunc main(output q:qnum<2>) {q=1;f(q);}",
        "qfunc f(output q:qbit) {q=1;} qfunc main(output q:qbit) {allocate(q);f(q);}",
        # General free/drop and coherent blocks are not current admission.
        "qfunc main(output q:qbit) {allocate(q);free(q);}",
        "qfunc main(output q:qbit) {allocate(q);drop(q);}",
        "qfunc main(output c:qbit,output q:qbit) {allocate(c);allocate(q);control(c){X(q);}}",
    ],
)
def test_future_role_and_control_semantics_fail_closed_today(source: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(source, source_format="qmod")


def test_basis_preserving_const_can_gain_relative_phase_and_entanglement() -> None:
    source = """qfunc mark(const x:qbit,const y:qbit) {phase(x*y,pi);}
    qfunc main(output x:qbit,output y:qbit) {
        allocate(x);allocate(y);H(x);H(y);mark(x,y);
    }"""
    p = synth.synthesize(source, source_format="qmod")
    check_state(
        p, fq.run(p), {(x, y): (-1) ** (x * y) / 2 for x in range(2) for y in range(2)}
    )


def test_current_within_apply_cleans_local_without_exporting_or_free() -> None:
    source = """qfunc main(output x:qnum<2>) {
        allocate(x);hadamard_transform(x);
        f:qbit;
        within { f=x<2; } apply { phase(f,pi); }
    }"""
    p = synth.synthesize(source, source_format="qmod")
    assert tuple(p.registers) == ("x",)
    check_state(p, fq.run(p), {(x,): (-1) ** int(x < 2) / 2 for x in range(4)})
