"""Native Const calls share checked execution and a full complex-state consumer."""

import math
from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError
from flagquantum.services._synthesis_registers import synthesize_registers
from tests.integration.test_synthesis_register_workflow import check_state
from tests.register_program_cases import tutorial_program
from tools.synthesis_import_examples.qmod_const.main import SOURCE

pytestmark = pytest.mark.integration


def test_output_first_signature_keeps_the_declared_const_argument_order() -> None:
    source = """qfunc compute(output result:qnum,const y:qnum<2>,const x:qnum<2>) { result=x*x+y; }
    qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) { x=1; y=2; compute(result,y,x); }"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(1, 2, 3): 1})


def test_fresh_helper_outputs_become_initialized_const_actuals() -> None:
    source = """qfunc prepare(output x:qnum<2>,output y:qnum<2>) {
      allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y);
    }
    qfunc compute(const x:qnum<2>,const y:qnum<2>,output energy:qnum) { energy=x*x+y; }
    qfunc main(output x:qnum,output y:qnum,output energy:qnum) { prepare(x,y); compute(x,y,energy); }"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(
        program,
        fq.run(program),
        {(x, y, x * x + y): 0.25 for x in range(4) for y in range(4)},
    )
    assert dict(program.report["register_widths"]) == {"x": 2, "y": 2, "energy": 4}


@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_sixteen_inputs_with_interleaved_const_and_output(x: int, y: int) -> None:
    source = f"""qfunc compute(const x:qnum<2>,output result:qnum,const y:qnum<2>) {{ result=x*x+y; }}
    qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {{ x={x}; y={y}; compute(x,result,y); }}"""
    program = synth.synthesize(source, source_format="qmod")
    result = fq.run(program)
    check_state(program, result, {(x, y, x * x + y): 1})
    assert dict(
        program.probabilities(result, registers=("result", "x", "y"))
    ) == pytest.approx({(x * x + y, x, y): 1})


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("producer", ["text", "file"])
def test_public_source_workflow_needs_no_gate_or_wire_design(
    tmp_path: Path, objective: Literal["width", "depth"], producer: str
) -> None:
    if producer == "file":
        path = tmp_path / "const.qmod"
        path.write_text(SOURCE)
        program = synth.synthesize(path, objective=objective, max_qubits=24)
    else:
        program = synth.synthesize(
            SOURCE, source_format="qmod", objective=objective, max_qubits=24
        )
    result = fq.run(program)
    check_state(
        program, result, {(x, y, x * x + y): 0.25 for x in range(4) for y in range(4)}
    )
    assert dict(
        program.probabilities(result, registers=("x", "y", "energy"))
    ) == pytest.approx({(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)})
    assert dict(program.report["register_widths"]) == {"x": 2, "y": 2, "energy": 4}
    assert program.report["source_format"] == "qmod"
    assert program.report["resources"] == program._checked.verify_result(result)


@pytest.mark.parametrize("producer", ["native", "recorded"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_native_const_replaces_direct_producer_behind_unchanged_consumer(
    producer: str, objective: Literal["width", "depth"]
) -> None:
    if producer == "native":
        program = synth.synthesize(
            """qfunc square(const x:qnum<2>,output y:qnum) { y=x*x+1; }
        qfunc main(output x:qnum<2>,output y:qnum) { allocate(x); hadamard_transform(x); square(x,y); }""",
            source_format="qmod",
            objective=objective,
        )
    else:
        root, functions = tutorial_program(2)
        program = synthesize_registers(
            root, functions, objective=objective, memory_limit_bytes=256 * 1024**2
        )
    result = fq.run(program)
    check_state(program, result, {(x, x * x + 1): 0.5 for x in range(4)})
    assert dict(program.probabilities(result, registers=("x", "y"))) == pytest.approx(
        {(x, x * x + 1): 0.25 for x in range(4)}
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_phase_preserves_external_correlation_and_allows_caller_mutation(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc copy(const q:qbit,output reference:qbit) { reference=q; }
    qfunc mark(const a:qbit,const b:qbit) { phase(a*b,pi); }
    qfunc apply(const a:qbit,const b:qbit) { mark(a,b); }
    qfunc main(output x:qbit,output y:qbit,output reference:qbit) {
      allocate(x); H(x); S(x); allocate(y); H(y); copy(x,reference);
      apply(x,y); X(x); H(y);
    }"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {(1, 0, 0): 1 / math.sqrt(2), (0, 1, 1): 1j / math.sqrt(2)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_const_actual_aliases_are_one_encoding_and_remain_usable(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc mark(const a:qbit,const b:qbit) { phase(a*b,pi); }
    qfunc main(output x:qbit) { allocate(x); H(x); mark(x,x); H(x); }"""
    program = synth.synthesize(
        source, source_format="qmod", max_qubits=1, objective=objective
    )
    check_state(program, fq.run(program), {(1,): 1})
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_global_sign_is_not_discarded(
    value: int, objective: Literal["width", "depth"]
) -> None:
    source = f"""qfunc mark(const a:qbit,const b:qbit) {{ phase(a*0+b*0+{value},pi); }}
    qfunc main(output x:qbit) {{ allocate(x); H(x); S(x); mark(x,x); }}"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {(0,): (-1) ** value / math.sqrt(2), (1,): (-1) ** value * 1j / math.sqrt(2)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.usefixtures("legacy_register_selection")
def test_repeated_calls_reuse_clean_work_and_initialize_distinct_outputs(
    objective: Literal["width", "depth"],
) -> None:
    source = """qfunc product(const a:qbit,const b:qbit,output out:qbit) { out=(a*b)*(a*b); }
    qfunc main(output x:qbit,output y:qbit,output first:qbit,output second:qbit) {
      allocate(x); H(x); S(x); allocate(y); H(y);
      product(x,y,first); product(x,y,second);
    }"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {(x, y, x * y, x * y): 1j**x / 2 for x in range(2) for y in range(2)},
    )
    row = program.selection.selected.arithmetic
    assert row.peak_workspace > 0
    assert row.peak_workspace == max(s.peak_workspace for s in row.segments)
    assert row.peak_workspace < sum(s.peak_workspace for s in row.segments)


@pytest.mark.parametrize(
    "source",
    [
        "qfunc f(const q:qbit) {} qfunc main(output q:qbit) { f(q); allocate(q); }",
        "qfunc f(const q:qbit) { H(q); } qfunc main(output q:qbit) { allocate(q); }",
        "qfunc f(const q:qbit,output out:qbit) { out=q; } qfunc main(output q:qbit) { allocate(q); f(q,q); }",
        "qfunc f(const q:qnum) {} qfunc main(output q:qbit) { allocate(q); }",
        "qfunc f(output const q:qbit) {} qfunc main(output q:qbit) { allocate(q); }",
        "qfunc main(const q:qbit) {}",
    ],
)
def test_bad_const_source_is_located_and_never_plans(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid Const source planned")
    )
    with pytest.raises(InteropConversionError) as error:
        synth.synthesize(source, source_format="qmod", source_name="bad_const.qmod")
    assert "bad_const.qmod:1:" in error.value.report.issues[0].message


def test_hidden_phase_still_uses_exact_target_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "qfunc f(const q:qbit) { phase(q,pi); } qfunc main(output q:qbit) { allocate(q); f(q); }"
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("unsupported phase target planned")
    )
    with pytest.raises(ValueError, match="phase output admits only"):
        synth.synthesize(source, source_format="qmod", topology="line")


def test_value_const_helpers_keep_line_target_with_an_unused_phase_helper() -> None:
    source = """qfunc unused(const q:qbit) { phase(q,pi); }
    qfunc add(const x:qnum<1>,const y:qnum<1>,output result:qnum) { result=x*x+y; }
    qfunc main(output x:qnum<1>,output y:qnum<1>,output result:qnum) {
      allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y); add(x,y,result);
    }"""
    program = synth.synthesize(source, source_format="qmod", topology="line")
    check_state(
        program,
        fq.run(program),
        {(x, y, x + y): 0.5 for x in range(2) for y in range(2)},
    )


def test_budget_memory_and_synthesis_only_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    program = synth.synthesize(SOURCE, source_format="qmod", source_name="energy.qmod")
    assert program.execution is not None
    assert program.report["source_name"] == "energy.qmod"
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(SOURCE, source_format="qmod", max_qubits=7)
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(SOURCE, source_format="qmod", memory_limit_bytes=1)
