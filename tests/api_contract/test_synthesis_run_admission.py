"""Production synthesis admission through the unchanged Runtime execution path."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.algorithms.synthesis_trial import pressure
from flagquantum import synth
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.result import ExecutionResult

pytestmark = pytest.mark.integration
OVERRIDES = (
    "options",
    "outputs",
    "noise_model",
    "compiler",
    "target",
    "target_qubits",
    "shots",
    "name",
)


def _program(topology: str | None = None, objective: str = "width") -> synth.Program:
    preparation = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).s(0).to_ir()
    return synth.synthesize(
        pressure,
        inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
        preparation=preparation,
        preparation_map={"x": (0,), "y": (1,)},
        topology=topology,
        objective=objective,
    )


@pytest.mark.parametrize("topology", [None, "line"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_root_run_preserves_actual_phase_identity_resources_and_names(
    topology, objective
) -> None:
    program = _program(topology, objective)
    result = fq.run(program)
    assert isinstance(result, ExecutionResult)
    assert result.plan.identity == program.execution.identity
    assert program._checked.verify_result(result) == program.report["resources"]
    observed = program.probabilities(result, registers=("x", "y", "result"))
    assert dict(observed) == pytest.approx(
        {(x, y, pressure(x, y)): 0.25 for x in range(2) for y in range(2)}, abs=1e-10
    )
    row = program.selection.selected
    state = result.statevector().reshape(-1)
    expected = torch.zeros_like(state)
    n = program.circuit.n_wires
    for x in range(2):
        for y in range(2):
            logical = (
                (((x << 1) | y) << 2) | pressure(x, y)
            ) << row.arithmetic.peak_workspace
            physical = sum(
                ((logical >> (n - 1 - q)) & 1)
                << (n - 1 - row.target.layout.physical_of(q))
                for q in range(n)
            )
            expected[physical] = torch.tensor(1j if x else 1, dtype=state.dtype) / 2
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("control", OVERRIDES)
def test_root_refuses_every_override_before_resolving_or_executing(
    control, monkeypatch
) -> None:
    program = _program()

    def forbidden(self):
        raise AssertionError("override refusal must precede plan access")

    monkeypatch.setattr(synth.Program, "execution", property(forbidden))
    with pytest.raises(TypeError, match="overrides"):
        fq.run(program, **{control: object()})


def test_execution_like_unrelated_objects_are_not_duck_typed() -> None:
    class Unrelated:
        @property
        def execution(self):
            raise AssertionError("unrelated execution properties must not be read")

    class Derived(synth.Program):
        pass

    with pytest.raises(TypeError):
        fq.run(Unrelated())
    with pytest.raises(TypeError, match="canonical"):
        fq.run(object.__new__(Derived))


def test_recorded_plan_replaces_resolution_without_changing_the_consumer(
    monkeypatch,
) -> None:
    import flagquantum._api as api
    import flagquantum.compiler as compiler
    import flagquantum.runtime.execution as execution
    import flagquantum.runtime.planner as planner

    program = _program("line")
    live = fq.run(program)
    recorded = program.execution.to_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("bound execution cannot plan, compile or synthesize")

    # Resolve the lazy root alias before patching its owning API function.
    monkeypatch.setattr(fq, "plan", forbidden)
    monkeypatch.setattr(api, "plan", forbidden)
    monkeypatch.setattr(compiler, "compile", forbidden)
    monkeypatch.setattr(execution, "compile_program", forbidden)
    monkeypatch.setattr(planner, "compile_program", forbidden)
    monkeypatch.setattr(synth, "synthesize", forbidden)

    restored = ExecutionPlan.from_json(recorded)
    product = replace(program, _checked=replace(program._checked, plan=restored))
    replayed = fq.run(product)
    assert replayed.plan.identity == live.plan.identity
    torch.testing.assert_close(
        replayed.statevector(), live.statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.probabilities(replayed)) == pytest.approx(
        dict(program.probabilities(live))
    )


def test_ordinary_circuit_execution_does_not_load_synthesis_services() -> None:
    code = """
import sys
import flagquantum as fq
assert "flagquantum.synth" not in sys.modules
assert "flagquantum.services" not in sys.modules
result=fq.run(fq.Circuit(1).h(0))
assert result.state is not None
assert "flagquantum.synth" not in sys.modules
assert "flagquantum.services" not in sys.modules
assert "flagquantum.services.synthesis" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_changed_product_is_rejected_before_existing_runtime_dispatch(
    monkeypatch,
) -> None:
    import flagquantum.runtime.execution as execution

    program = _program()
    program.circuit.metadata["changed"] = True

    def forbidden(*args, **kwargs):
        raise AssertionError("a stale product must fail before numerical dispatch")

    monkeypatch.setattr(execution, "run", forbidden)
    with pytest.raises(ValueError, match="circuit changed"):
        fq.run(program)
