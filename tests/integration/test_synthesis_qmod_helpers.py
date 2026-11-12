"""Composed native preparations share checked execution and full-state readers."""

import pytest

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.ecosystem import InteropConversionError
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_sixteen_basis_inputs_of_output_arithmetic_helper(x, y):
    source = f"""qfunc calculate(output a:qnum<2>,output b:qnum<2>,output value:qnum) {{
      a={x}; b={y}; value=a*a+b;
    }}
    qfunc main(output x:qnum,output y:qnum,output result:qnum) {{ calculate(x,y,result); }}"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(x, y, x * x + y): 1})
    assert dict(program.report["register_widths"]) == {"x": 2, "y": 2, "result": 4}


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("producer", ["helpers", "flat", "file"])
def test_nested_helpers_replace_flat_source_behind_same_consumer(
    tmp_path, objective, producer
):
    from tools.synthesis_import_examples.qmod_helpers.main import SOURCE

    if producer == "file":
        path = tmp_path / "helpers.qmod"
        path.write_text(SOURCE)
        program = synth.synthesize(path, objective=objective)
    else:
        source = (
            SOURCE
            if producer == "helpers"
            else """qfunc main(output x:qnum<2>,output y:qnum<2>,output energy:qnum) {
        allocate(x); allocate(y); hadamard_transform(x); hadamard_transform(y); energy=x*x+y;
        }"""
        )
        program = synth.synthesize(source, source_format="qmod", objective=objective)
    result = fq.run(program)
    check_state(
        program, result, {(x, y, x * x + y): 0.25 for x in range(4) for y in range(4)}
    )
    assert dict(
        program.probabilities(result, registers=("energy", "y", "x"))
    ) == pytest.approx({(x * x + y, y, x): 1 / 16 for x in range(4) for y in range(4)})
    assert program.report["resources"] == program._checked.verify_result(result)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_repeated_helpers_preserve_complex_phases_and_disjoint_outputs(objective):
    source = """qfunc pair(output x:qbit,output y:qbit) {
       allocate(x); H(x); S(x); allocate(y); H(y); phase(x*y,pi);
    }
    qfunc main(output a:qbit,output b:qbit,output c:qbit,output d:qbit) {
       pair(a,b); pair(c,d); H(b); H(d);
    }"""
    program = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        program,
        fq.run(program),
        {(a, a, c, c): 1j ** (a + c) / 2 for a in range(2) for c in range(2)},
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "body", ["seed(x); seed(x);", "x=0; seed(x);", "seed(x+1);", "missing(x);"]
)
def test_invalid_calls_never_plan(monkeypatch, body):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid helper reached plan")
    )
    with pytest.raises(InteropConversionError) as error:
        synth.synthesize(
            "qfunc seed(output q:qbit) { allocate(q); }\n"
            + f"qfunc main(output x:qbit) {{ {body} }}",
            source_format="qmod",
            source_name="bad_helpers.qmod",
        )
    assert "bad_helpers.qmod:2:" in error.value.report.issues[0].message


def test_synthesis_never_executes_and_source_metadata_is_retained(monkeypatch):
    from tools.synthesis_import_examples.qmod_helpers.main import SOURCE

    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    program = synth.synthesize(SOURCE, source_format="qmod", source_name="helpers.qmod")
    assert program.report["source_name"] == "helpers.qmod"
    assert program.execution is not None


def test_phase_coefficient_is_resolved_in_helper_scope_not_caller_output_scope():
    source = """qfunc seed(output q:qbit) { allocate(q); H(q); phase(q,pi); }
    qfunc main(output pi:qbit) { seed(pi); H(pi); }"""
    program = synth.synthesize(source, source_format="qmod")
    check_state(program, fq.run(program), {(1,): 1})


def test_helper_cannot_bypass_existing_exact_phase_target_guard(monkeypatch):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid target planned")
    )
    source = """qfunc seed(output q:qbit) { allocate(q); H(q); phase(q,pi); }
    qfunc main(output x:qbit) { seed(x); }"""
    with pytest.raises(ValueError, match="phase output admits only"):
        synth.synthesize(source, source_format="qmod", topology="line")
