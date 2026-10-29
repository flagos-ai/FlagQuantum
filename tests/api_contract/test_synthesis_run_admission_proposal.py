"""Proposal-only reference semantics; production fq.run(Program) is not admitted."""

import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.algorithms.synthesis_trial import pressure
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.result import ExecutionResult
from flagquantum.services import _synthesis as synth

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads(
    (ROOT / "contracts/synthesis-program-run-admission-proposal.json").read_text()
)


def _reference_consumer(
    program: object,
    *,
    resolve: Callable[[synth.Program], ExecutionPlan],
    **controls: object,
) -> ExecutionResult:
    """Test-only proposed dispatch; never installed as a production entry."""
    if set(controls) - set(CONTRACT["non_none_overrides_rejected"]) or any(
        value is not None for value in controls.values()
    ):
        raise TypeError("a bound synthesis program does not accept execution overrides")
    if type(program) is not synth.Program:
        raise TypeError("only the canonical synthesis Program is admitted")
    return fq.run(resolve(program))


def _live(program: synth.Program) -> ExecutionPlan:
    return program.execution


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


def test_proposal_does_not_self_authorize_or_modify_the_stable_root() -> None:
    assert CONTRACT["status"] == "proposed"
    assert CONTRACT["implementation_authorized"] is False
    assert CONTRACT["production_implemented"] is False
    assert CONTRACT["api_owner_approval"] is None
    assert not CONTRACT["human_trial_records"]
    assert (
        str(inspect.signature(fq.run, eval_str=False))
        == CONTRACT["current_run_signature"]
    )
    assert CONTRACT["proposed_run_signature"] == CONTRACT[
        "current_run_signature"
    ].replace(" | LindbladPlan", " | LindbladPlan | SynthesisProgram")
    assert not (ROOT / "flagquantum/synth").exists()
    # This guard must be replaced by production admission tests only through
    # the subsequent approved implementation change, not by a proposal flag.
    with pytest.raises(TypeError):
        fq.run(_program())


@pytest.mark.parametrize("topology", [None, "line"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_reference_route_preserves_actual_phase_identity_resources_and_names(
    topology, objective
) -> None:
    program = _program(topology, objective)
    result = _reference_consumer(program, resolve=_live)
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


@pytest.mark.parametrize("control", CONTRACT["non_none_overrides_rejected"])
def test_reference_refuses_every_override_before_resolving_or_executing(
    control,
) -> None:
    def forbidden(program):
        raise AssertionError("override refusal must precede plan access")

    with pytest.raises(TypeError, match="overrides"):
        _reference_consumer(object(), resolve=forbidden, **{control: object()})


def test_execution_like_unrelated_objects_are_not_duck_typed() -> None:
    class Unrelated:
        @property
        def execution(self):
            raise AssertionError("unrelated execution properties must not be read")

    class Derived(synth.Program):
        pass

    for value in (Unrelated(), object.__new__(Derived)):
        with pytest.raises(TypeError, match="canonical"):
            _reference_consumer(value, resolve=_live)


def test_recorded_plan_replaces_resolution_without_changing_the_consumer(
    monkeypatch,
) -> None:
    import flagquantum._api as api
    import flagquantum.compiler as compiler
    import flagquantum.runtime.execution as execution
    import flagquantum.runtime.planner as planner

    program = _program("line")
    live = _reference_consumer(program, resolve=_live)
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

    def replay(product):
        assert product is program
        return ExecutionPlan.from_json(recorded)

    replayed = _reference_consumer(program, resolve=replay)
    assert replayed.plan.identity == live.plan.identity
    torch.testing.assert_close(
        replayed.statevector(), live.statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.probabilities(replayed)) == pytest.approx(
        dict(program.probabilities(live))
    )
