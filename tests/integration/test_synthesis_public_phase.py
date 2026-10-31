"""Public phase synthesis preserves absolute amplitudes and clean workspace."""

import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.errors import PlanningError
from flagquantum.services import synthesis
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


def equal(x: int, y: int) -> bool:
    return x == y


def identity(x: int) -> int:
    return x


def true(x: int) -> int:
    return x * 0 + 1


def false(x: int) -> int:
    return x * 0


def less_than(x: int, y: int) -> bool:
    return x < y


def helper_marked(x: int, y: int, threshold: int) -> bool:
    return less_than(x * x + y, threshold)


def phase(function=marked, *, objective="width", **kwargs):
    return synth.synthesize(
        function,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        output_mode="phase",
        objective=objective,
        max_qubits=22,
        memory_limit_bytes=64 * 1024**2,
        **kwargs,
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("predicate", [marked, equal])
def test_public_all_16_basis_inputs_have_exact_sign_and_clean_workspace(
    objective, predicate
):
    for x in range(4):
        for y in range(4):
            program = phase(
                predicate,
                objective=objective,
                input_state=synth.InputState.basis(x=x, y=y),
            )
            result = fq.run(program)
            check_state(program, result, {(x, y): (-1) ** int(predicate(x, y))})
            assert tuple(program.registers) == ("x", "y")
            assert dict(program.probabilities(result, registers=("x", "y"))) == {
                (x, y): 1.0
            }
            assert program.report["result_width"] == 0
            assert "arithmetic_resources" not in program.report


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("predicate", [marked, equal])
def test_public_coherent_preparation_retains_complex_phase_and_actual_resources(
    objective, predicate
):
    preparation = fq.Circuit(4, dtype=torch.complex128)
    preparation.h(0).h(1).h(2).h(3).s(0).t(2)
    program = phase(
        predicate,
        objective=objective,
        preparation=preparation.to_ir(),
        preparation_map={"x": (0, 1), "y": (2, 3)},
    )
    result = fq.run(program)
    expected = {
        (x, y): (
            (1j if x & 2 else 1)
            * (complex(math.cos(math.pi / 4), math.sin(math.pi / 4)) if y & 2 else 1)
            * (-1) ** int(predicate(x, y))
            / 4
        )
        for x in range(4)
        for y in range(4)
    }
    check_state(program, result, expected)
    assert program.report["output_mode"] == "phase"
    assert program.report["scope"] == "actual_cpu_preparation_predicate_phase_cleanup"
    assert program.report["register_widths"] == {"x": 2, "y": 2}
    assert program.report["resources"] == program._checked.verify_result(result)
    assert program.report["source_resources"].n_qubits == program.circuit.n_wires
    assert program.report["workspace_qubits"] == program.circuit.n_wires - 4
    assert program.report["memory_limit_origin"] == "explicit"
    assert result.plan.identity == program.execution.identity
    assert dict(program.probabilities(result, registers=("x", "y"))) == pytest.approx(
        {(x, y): 1 / 16 for x in range(4) for y in range(4)}, abs=1e-12
    )
    with pytest.raises(ValueError, match="available names"):
        program.probabilities(result)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_public_uniform_helper_constants_and_qfunc_share_the_existing_pipeline(
    objective,
):
    function = synth.qfunc(helper_marked)
    program = phase(
        function,
        objective=objective,
        constants={"threshold": 5},
        functions=(less_than,),
        input_state=synth.InputState.uniform(),
    )
    check_state(
        program,
        fq.run(program),
        {(x, y): (-1) ** int(x * x + y < 5) / 4 for x in range(4) for y in range(4)},
    )
    assert type(program) is synth.Program
    assert program.report["constants"] == {"threshold": 5}


@pytest.mark.parametrize("predicate,sign", [(true, -1), (false, 1), (identity, None)])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_public_constant_and_direct_one_bit_returns_preserve_absolute_sign(
    predicate, sign, objective
):
    program = synth.synthesize(
        predicate,
        inputs={"x": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        objective=objective,
    )
    check_state(
        program,
        fq.run(program),
        {
            (x,): (sign if sign is not None else (-1) ** x) / math.sqrt(2)
            for x in range(2)
        },
    )
    if predicate is identity:
        assert program.report["workspace_qubits"] == 0
        assert program.report["resources"].n_qubits == 1


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_public_oracle_interference_and_double_application(objective):
    program = synth.synthesize(
        identity,
        inputs={"x": synth.UInt(1)},
        output_mode="phase",
        objective=objective,
    )
    single = fq.Circuit(1, dtype=torch.complex128).h(0)
    single = single.compose(program.circuit).h(0)
    result = fq.run(single, options=fq.ExecutionOptions(precision="complex128"))
    torch.testing.assert_close(
        result.statevector().reshape(-1),
        torch.tensor([0, 1], dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    double = fq.Circuit(1, dtype=torch.complex128).h(0).s(0)
    double = double.compose(program.circuit).compose(program.circuit)
    result = fq.run(double, options=fq.ExecutionOptions(precision="complex128"))
    torch.testing.assert_close(
        result.statevector().reshape(-1),
        torch.tensor([1, 1j], dtype=torch.complex128) / math.sqrt(2),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("mode", [None, True, 1, [], {}, object()])
def test_invalid_mode_types_refuse_before_capture_or_planning(mode, monkeypatch):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned invalid mode")
    )
    monkeypatch.setattr(
        synthesis, "capture", lambda *a, **k: pytest.fail("captured invalid mode")
    )
    with pytest.raises(TypeError, match="output_mode"):
        synth.synthesize(marked, output_mode=mode)


@pytest.mark.parametrize("mode", ["", "sign", "PHASE"])
def test_unknown_mode_values_refuse_before_work(mode):
    with pytest.raises(ValueError, match="output_mode"):
        synth.synthesize(marked, output_mode=mode)


@pytest.mark.parametrize("kind", ["file", "text"])
def test_native_phase_is_refused_before_source_acquisition(kind, tmp_path, monkeypatch):
    from flagquantum.ecosystem._qmod import source

    monkeypatch.setattr(
        source, "load_qmod", lambda *a, **k: pytest.fail("read native phase")
    )
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned native phase")
    )
    value = tmp_path / "absent.qmod" if kind == "file" else "invalid source"
    with pytest.raises(ValueError, match="Python definitions only"):
        synth.synthesize(value, source_format="qmod", output_mode="phase")


@pytest.mark.parametrize(
    "target", [{"topology": "line"}, {"basis": ("x", "cx")}, {"basis": []}]
)
def test_phase_targets_refuse_before_capture_or_basis_iteration(target, monkeypatch):
    from flagquantum.services import synthesis

    monkeypatch.setattr(
        synthesis, "capture", lambda *a, **k: pytest.fail("captured unsupported target")
    )
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned unsupported target")
    )
    with pytest.raises(ValueError, match="logical CPU targets"):
        synth.synthesize(marked, output_mode="phase", **target)


def test_phase_refuses_multibit_return_before_plan(monkeypatch):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned multibit phase")
    )
    with pytest.raises(ValueError, match="exactly one-bit"):
        synth.synthesize(identity, inputs={"x": synth.UInt(2)}, output_mode="phase")


@pytest.mark.parametrize(
    "constraint",
    [
        {"max_operations": 1},
        {"max_depth": 1},
        {"max_workspace": 0},
        {"max_qubits": 4},
        {"construction_budget": synth.ConstructionBudget(candidate_checks=1)},
    ],
)
def test_phase_constraints_refuse_before_plan(constraint, monkeypatch):
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned rejected candidate")
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            marked,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            output_mode="phase",
            **constraint,
        )


def test_phase_memory_refusal_does_not_execute(monkeypatch):
    monkeypatch.setattr(
        fq, "run", lambda *a, **k: pytest.fail("executed during synthesis")
    )
    with pytest.raises(PlanningError, match="memory"):
        synth.synthesize(
            marked,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            output_mode="phase",
            memory_limit_bytes=1,
        )


def test_default_and_explicit_value_preserve_circuit_report_and_named_result():
    args = dict(inputs={"x": synth.UInt(1)}, input_state=synth.InputState.uniform())
    default = synth.synthesize(identity, **args)
    explicit = synth.synthesize(identity, output_mode="value", **args)
    assert default.execution.identity == explicit.execution.identity
    assert default.report == explicit.report
    assert default.report["output_mode"] == "value"
    assert default.report["result_width"] == 1
    assert "arithmetic_resources" in default.report
    assert dict(default.probabilities(fq.run(default))) == pytest.approx(
        {(0,): 0.5, (1,): 0.5}
    )


def test_public_example_has_no_private_imports_gates_or_decoding():
    source = (
        Path(__file__).resolve().parents[2] / "examples/synthesis_phase_oracle.py"
    ).read_text()
    assert 'output_mode="phase"' in source
    assert "fq.run(program)" in source
    assert "._" not in source and "fq.Circuit" not in source
    assert "physical_of" not in source and "statevector()" not in source
