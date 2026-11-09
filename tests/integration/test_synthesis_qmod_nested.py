"""Lexical apply scopes preserve outer dependencies and clean all hidden work."""

import math
from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis.tutorials.qmod.qmod_nested.main import SOURCE as ENERGY_SOURCE
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_user_example_marks_sum_and_hides_both_temporaries(
    objective: Literal["width", "depth"],
) -> None:
    program = synth.synthesize(ENERGY_SOURCE, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {(x, y): (-1) ** int(x + y < 4) / 4 for x in range(4) for y in range(4)},
    )
    assert tuple(program.registers) == ("x", "y")


SOURCE = """qfunc main(output x:qnum<2>,output y:qnum<2>) {
 allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
 outer:qbit; inner:qbit;
 within {outer=x<y;} apply {
   within {inner=outer*(x!=y);} apply {phase(inner,pi);}
   phase(outer,pi);
   within {inner=x==y;} apply {phase(inner,pi);}
 }
}"""


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_nested_all_inputs_preserve_parent_and_reuse_inner_name(
    objective: Literal["width", "depth"], x: int, y: int
) -> None:
    source = SOURCE.replace(
        "allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);",
        f"x={x}; y={y};",
    )
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(program, fq.run(program), {(x, y): (-1) ** int(x == y)})
    assert tuple(program.registers) == ("x", "y")


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_complex_state_keeps_external_reference(
    objective: Literal["width", "depth"],
) -> None:
    # A qbit control carries a complex phase and is correlated with a reference.
    source = """qfunc main(output q:qbit,output reference:qbit) {
      allocate(q); H(q); S(q); reference=q; a:qbit; b:qbit;
      within {a=q;} apply {
        within {b=a;} apply {phase(b,pi);}
      }
    }"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program, fq.run(program), {(0, 0): 1 / math.sqrt(2), (1, 1): -1j / math.sqrt(2)}
    )


@pytest.mark.parametrize(
    "body",
    [
        "within {a=q;} apply {within {b=a;} apply {} phase(b,pi);}",
        "within {a=q;} apply {within {a=q;} apply {}}",
        "within {a=q;} apply {within {b=a;} apply {X(a);}}",
        "within {within {a=q;} apply {}} apply {}",
    ],
)
def test_nested_scope_rejects_escape_parent_write_and_bad_preparation(
    body: str,
) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            "qfunc main(output q:qbit) {allocate(q); a:qbit; b:qbit;" + body + "}",
            source_format="qmod",
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_lexical_and_helper_scopes_share_consumer_and_resource_limits(
    objective: Literal["width", "depth"],
) -> None:
    helper = "qfunc mark(const a:qbit) { b:qbit; within {b=a;} apply {phase(b,pi);} }"
    main = """qfunc main(output q:qbit) {allocate(q); H(q); a:qbit;
       within {a=q;} apply { ACTION }
    }"""
    programs = [
        synth.synthesize(source, source_format="qmod", objective=objective)
        for source in (
            main.replace("ACTION", "within {b=a;} apply {phase(b,pi);}").replace(
                "a:qbit;", "a:qbit; b:qbit;"
            ),
            helper + main.replace("ACTION", "mark(a);"),
        )
    ]
    for program in programs:
        check_state(
            program, fq.run(program), {(0,): 1 / math.sqrt(2), (1,): -1 / math.sqrt(2)}
        )
        assert tuple(program.registers) == ("q",)
    assert programs[0].report["resources"] == programs[1].report["resources"]
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(SOURCE, source_format="qmod", max_workspace=1)
