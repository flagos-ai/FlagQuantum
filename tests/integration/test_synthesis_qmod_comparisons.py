"""Native comparisons reuse checked execution, clean work and named outputs."""

import math
import operator
from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis_qmod_comparisons import SOURCE
from flagquantum import _api, synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
RELATIONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_six_relations_over_all_two_bit_inputs(
    relation: str, x: int, y: int
) -> None:
    source = f"""qfunc compare(const a:qnum<2>,const b:qnum<2>,output flag:qbit) {{ flag=a{relation}b; }}
    qfunc main(output x:qnum<2>,output y:qnum<2>,output flag:qbit) {{ x={x}; y={y}; compare(x,y,flag); }}"""
    program = synth.synthesize(source, source_format="qmod")
    flag = int(RELATIONS[relation](x, y))
    result = fq.run(program)
    check_state(program, result, {(x, y, flag): 1})
    assert dict(
        program.probabilities(result, registers=("x", "y", "flag"))
    ) == pytest.approx({(x, y, flag): 1})


@pytest.mark.parametrize("relation", RELATIONS)
def test_unequal_quantum_widths_are_zero_extended(relation: str) -> None:
    source = f"""qfunc main(output x:qnum<1>,output y:qnum<3>,output flag:qbit) {{
    allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y); flag=x{relation}y;
    }}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(
        program,
        fq.run(program),
        {
            (x, y, int(RELATIONS[relation](x, y))): 0.25
            for x in range(2)
            for y in range(8)
        },
    )


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("left_literal", [False, True])
def test_literal_on_either_side_preserves_the_full_quantum_domain(
    relation: str, left_literal: bool
) -> None:
    expression = f"2{relation}x" if left_literal else f"x{relation}2"
    program = synth.synthesize(
        f"qfunc main(output x:qnum<2>,output flag:qbit) {{ allocate(x); hadamard_transform(x); flag={expression}; }}",
        source_format="qmod",
    )
    check_state(
        program,
        fq.run(program),
        {
            (
                x,
                int(
                    RELATIONS[relation](2, x)
                    if left_literal
                    else RELATIONS[relation](x, 2)
                ),
            ): 0.5
            for x in range(4)
        },
    )


@pytest.mark.parametrize("relation", RELATIONS)
def test_aliased_const_actuals_do_not_copy_quantum_state(relation: str) -> None:
    source = f"""qfunc compare(const a:qnum<2>,const b:qnum<2>,output flag:qbit) {{ flag=a{relation}b; }}
    qfunc main(output x:qnum<2>,output flag:qbit) {{ allocate(x); hadamard_transform(x); compare(x,x,flag); }}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(
        program,
        fq.run(program),
        {(x, int(RELATIONS[relation](x, x))): 0.5 for x in range(4)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complex_correlated_flags_can_be_phased_as_exported_outputs(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc copy(const x:qbit,output reference:qbit) { reference=x; }
    qfunc compare(const x:qbit,const y:qbit,output flag:qbit) { flag=x+1<y*2; }
    qfunc main(output x:qbit,output y:qbit,output flag:qbit,output reference:qbit) {
      allocate(x); H(x); S(x); allocate(y); H(y); copy(x,reference);
      compare(x,y,flag); phase(flag,pi);
    }"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {
            (x, y, int(x + 1 < y * 2), x): 1j**x * (-1) ** (x + 1 < y * 2) / 2
            for x in range(2)
            for y in range(2)
        },
    )
    assert "flag" in program.registers


def predicate(x: int, y: int) -> bool:
    return x < y


@pytest.mark.parametrize("producer", ["native", "python"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_both_producers_share_the_same_execution_consumer(
    producer: str, objective: Literal["width", "depth"]
) -> None:
    if producer == "native":
        program = synth.synthesize(
            SOURCE.replace("flag", "result"), source_format="qmod", objective=objective
        )
    else:
        program = synth.synthesize(
            predicate,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            input_state=synth.InputState.uniform(),
            objective=objective,
        )
    result = fq.run(program)
    check_state(
        program, result, {(x, y, int(x < y)): 0.25 for x in range(4) for y in range(4)}
    )
    assert dict(
        program.probabilities(result, registers=("x", "y", "result"))
    ) == pytest.approx({(x, y, int(x < y)): 1 / 16 for x in range(4) for y in range(4)})
    assert program.report["resources"] == program._checked.verify_result(result)


@pytest.mark.parametrize("producer", ["file", "text"])
def test_new_original_is_verified_at_unchanged_declared_size(
    tmp_path: Path, producer: str
) -> None:
    path = CORPUS / "comparator_integer_example.qmod"
    original = path.read_bytes()
    if producer == "file":
        program = synth.synthesize(path, max_qubits=24)
    else:
        program = synth.synthesize(
            original.decode(),
            source_format="qmod",
            source_name=str(path),
            max_qubits=24,
        )
    result = fq.run(program)
    check_state(program, result, {(a, int(a <= 2)): 1 / math.sqrt(8) for a in range(8)})
    assert dict(program.report["register_widths"]) == {"a": 3, "res": 1}
    assert dict(program.probabilities(result, registers=("a", "res"))) == pytest.approx(
        {(a, int(a <= 2)): 1 / 8 for a in range(8)}
    )
    assert path.read_bytes() == original


@pytest.mark.parametrize("statement", ["flag=x<y<2;", "phase(x<y,pi);", "flag=x is y;"])
def test_unsupported_conditions_fail_before_planning(
    monkeypatch: pytest.MonkeyPatch, statement: str
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid condition planned")
    )
    with pytest.raises(InteropConversionError) as error:
        synth.synthesize(
            f"qfunc main(output x:qnum<2>,output y:qnum<2>,output flag:qbit) {{ x=0; y=0; {statement} }}",
            source_format="qmod",
            source_name="comparison.qmod",
        )
    assert "comparison.qmod:1:" in error.value.report.issues[0].message


def test_existing_target_budget_and_memory_guards_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    assert synth.synthesize(SOURCE, source_format="qmod").execution is not None
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(SOURCE, source_format="qmod", max_qubits=4)
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(SOURCE, source_format="qmod", memory_limit_bytes=1)
    with pytest.raises(ValueError, match="phase output admits only"):
        synth.synthesize(
            SOURCE.replace("compare(x,y,flag);", "compare(x,y,flag); phase(flag,pi);"),
            source_format="qmod",
            topology="line",
        )
