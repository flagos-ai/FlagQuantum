"""Public native phase workflows preserve complex signs and clean workspace."""

import math

import pytest

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration


def native(body, declarations="output x:qbit, output y:qbit", **options):
    return synth.synthesize(
        f"qfunc main({declarations}) {{ {body} }}", source_format="qmod", **options
    )


def product(x: int, y: int) -> int:
    return x * y


@pytest.mark.parametrize("producer", ["python", "qmod"])
def test_native_adapter_replaces_producer_behind_unchanged_execution_consumer(producer):
    if producer == "python":
        program = synth.synthesize(
            product,
            inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            output_mode="phase",
        )
    else:
        program = native("allocate(x); H(x); allocate(y); H(y); phase(x*y,pi);")
    result = fq.run(program)
    check_state(
        program,
        result,
        {(x, y): (-1) ** (x * y) / 2 for x in range(2) for y in range(2)},
    )
    assert tuple(program.registers) == ("x", "y")
    assert program.report["resources"].n_qubits == 2


def test_synthesis_only_plans_and_never_executes(monkeypatch):
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    program = native("allocate(x); H(x); allocate(y); H(y); phase(x*y,pi);")
    assert program.execution is not None


@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [0, 1])
def test_every_basis_input_preserves_values_and_exact_sign(x, y):
    program = native(f"x={x}; y={y}; phase(x*y,pi);")
    check_state(program, fq.run(program), {(x, y): (-1) ** (x * y)})
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("producer", ["text", "file"])
def test_complex_superposition_text_and_file_share_execution(
    tmp_path, producer, objective
):
    source = """qfunc main(output x:qbit, output y:qbit) {
      allocate(x); H(x); S(x); allocate(y); H(y); phase(x*y,pi);
    }"""
    if producer == "file":
        path = tmp_path / "phase.qmod"
        path.write_text(source)
        program = synth.synthesize(path, objective=objective)
    else:
        program = synth.synthesize(source, source_format="qmod", objective=objective)
    result = fq.run(program)
    check_state(
        program,
        result,
        {(x, y): 1j**x * (-1) ** (x * y) / 2 for x in range(2) for y in range(2)},
    )
    assert dict(program.probabilities(result, registers=("y", "x"))) == pytest.approx(
        {(y, x): 0.25 for x in range(2) for y in range(2)}
    )
    assert program.report["source_format"] == "qmod"
    assert program.report["resources"] == program._checked.verify_result(result)
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "expression,values",
    [("x*x", (0, 1)), ("x**2", (0, 1)), ("x*0", (0, 0)), ("x*0+1", (1, 1))],
)
def test_aliases_and_quantum_dependent_constants_preserve_global_phase(
    expression, values
):
    program = native(
        f"allocate(x); H(x); S(x); phase({expression},pi);", "output x:qbit"
    )
    check_state(
        program,
        fq.run(program),
        {(x,): 1j**x * (-1) ** values[x] / math.sqrt(2) for x in range(2)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_cleanup_precedes_mutation_subsequent_arithmetic_and_interference(objective):
    program = native(
        "allocate(x); H(x); S(x); allocate(y); H(y); "
        "phase((x*y)*(x*y),pi); X(x); res=x*y; phase(res*y,pi); H(y);",
        "output x:qbit, output y:qbit, output res:qnum",
        objective=objective,
    )
    expected = {}
    for x in range(2):
        for y in range(2):
            for z in range(2):
                key = (1 - x, z, (1 - x) * y)
                expected[key] = expected.get(key, 0) + 1j**x * (-1) ** (
                    x * y + (1 - x) * y + y * z
                ) / math.sqrt(8)
    check_state(program, fq.run(program), expected)
    assert program.report["workspace_qubits"] > 0


def test_double_phase_cancels_and_interference_reveals_sign():
    for calls, expected in [(1, {(1,): 1}), (2, {(0,): 1})]:
        program = native(
            "allocate(x); H(x); " + "phase(x,pi); " * calls + "H(x);", "output x:qbit"
        )
        check_state(program, fq.run(program), expected)


def test_gate_free_public_example_entangles_two_named_registers():
    from tools.synthesis_import_examples.qmod_phase.main import SOURCE

    program = synth.synthesize(SOURCE, source_format="qmod", max_qubits=2)
    result = fq.run(program)
    check_state(program, result, {(0, 0): 1 / math.sqrt(2), (1, 1): 1 / math.sqrt(2)})
    assert dict(program.probabilities(result, registers=("x", "y"))) == pytest.approx(
        {(0, 0): 0.5, (1, 1): 0.5}
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "options,error",
    [
        ({"max_qubits": 1}, synth.SynthesisError),
        ({"topology": "line"}, ValueError),
        ({"output_mode": "phase"}, ValueError),
    ],
)
def test_phase_constraints_and_target_refusals_precede_planning(
    monkeypatch, options, error
):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid source reached planning")
    )
    with pytest.raises(error):
        native("allocate(x); allocate(y); phase(x*y,pi);", **options)


def test_memory_guard_refuses_at_existing_planning_boundary():
    with pytest.raises(PlanningError, match="memory"):
        native("allocate(x); allocate(y); phase(x*y,pi);", memory_limit_bytes=1)


@pytest.mark.parametrize("statement", ["phase(x,1);", "phase(x+1,pi);", "phase(x,pi);"])
def test_public_invalid_or_uninitialized_phase_has_located_conversion_report(
    monkeypatch, statement
):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("invalid source reached planning")
    )
    with pytest.raises(InteropConversionError) as error:
        native(statement, "output x:qbit", source_name="invalid_phase.qmod")
    assert "invalid_phase.qmod:1:" in error.value.report.issues[0].message
