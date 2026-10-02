"""Planning contract for the stabilizer execution mode.

These tests cover what the plan and the option vocabulary promise about a
stabilizer run: the representation it names, the cost it reports, the refusal of
everything the representation cannot hold, and that planning one of those
refusals fails before a route is selected. The engine's own numerical contract
is tested in `tests/team/simulation/`.
"""

from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime.execution_plan_contract import (
    plan_from_dict,
    plan_to_dict,
)
from flagquantum.runtime.planner import plan_advanced
from flagquantum.runtime.planner.estimates import (
    estimate_stabilizer_bytes,
    estimate_state_bytes,
)
from flagquantum.runtime.planner.execution_policy import (
    normalize_execution_state_mode,
)

pytestmark = pytest.mark.unit


def _ghz(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires - 1):
        circuit.cx(wire, wire + 1)
    return circuit


def _options(**overrides: object) -> fq.ExecutionOptions:
    values = {"mode": "stabilizer", "shots": 8, "seed": 3}
    values.update(overrides)
    return fq.ExecutionOptions(**values)


def test_the_estimate_is_the_packed_clifford_tableau() -> None:
    """`2n` generators of `2n` Paulis plus a sign, bit-packed, rounded up."""

    for n_wires in (0, 1, 2, 3, 17, 1024, 16384):
        expected = -(-(4 * n_wires**2 + 2 * n_wires) // 8)
        assert estimate_stabilizer_bytes(n_wires) == expected


def test_the_estimate_refuses_a_negative_wire_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_stabilizer_bytes(-1)


def test_the_mode_the_planner_reports_is_the_one_that_was_asked_for() -> None:
    plan = fq.plan(_ghz(4), options=_options())

    assert plan.state_mode == "stabilizer"
    assert plan.recommended_mode == "stabilizer"
    assert plan.state_bytes == estimate_stabilizer_bytes(4)


def test_the_tableau_estimate_replaces_the_amplitude_estimate() -> None:
    """The gap between the two grows with the width, which is the point."""

    plan = fq.plan(_ghz(24), options=_options())

    assert plan.state_bytes == estimate_stabilizer_bytes(24)
    assert plan.state_bytes < estimate_state_bytes(24)


def test_planning_succeeds_past_the_width_an_amplitude_count_cannot_serialize() -> None:
    """A 16384-wire tableau is 134 MB; a 16384-wire amplitude store is 2**16387."""

    plan = fq.plan(_ghz(16384), options=_options())

    assert plan.state_bytes == 134221824
    assert len(str(plan.state_bytes)) == 9


def test_automatic_selection_never_reaches_the_stabilizer_representation() -> None:
    """Selection stays explicit, so no existing plan changes meaning."""

    for n_wires in (2, 24):
        plan = fq.plan(_ghz(n_wires), options=fq.ExecutionOptions(shots=8))

        assert plan.state_mode == "statevector"
        assert plan.recommended_mode != "stabilizer"


def test_normalizing_the_state_mode_reports_the_stabilizer_representation() -> None:
    assert normalize_execution_state_mode("stabilizer") == "stabilizer"


def test_a_non_clifford_gate_fails_planning_with_the_gate_named() -> None:
    circuit = fq.Circuit(2).h(0).t(1)

    with pytest.raises(
        CapabilityError, match=r"instruction 1 't' is not a Clifford gate"
    ):
        fq.plan(circuit, options=_options())


def test_planning_refuses_a_channel_before_a_route_is_selected() -> None:
    circuit = fq.Circuit(2).h(0).depolarizing(1)

    with pytest.raises(CapabilityError, match="is a noise channel"):
        fq.plan(circuit, options=_options())


def test_the_planning_refusal_is_the_engine_refusal() -> None:
    """One owner for the Clifford gate set, so the two texts cannot drift."""

    from flagquantum.simulation.stabilizer import require_clifford_program

    circuit = fq.Circuit(2).h(0).t(1)
    with pytest.raises(CapabilityError) as engine:
        require_clifford_program(circuit)
    with pytest.raises(CapabilityError) as planner:
        fq.plan(circuit, options=_options())

    assert str(planner.value) == str(engine.value)


def test_a_noisy_program_is_refused_by_the_stable_noise_rule() -> None:
    import flagquantum.noise as noise

    model = noise.NoiseModel()
    model.add(("h",), noise.depolarizing_channel(0.01))

    with pytest.raises(ValidationError, match="mode='auto' or mode='density_matrix'"):
        fq.plan(fq.Circuit(1).h(0), options=_options(), noise_model=model)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device": "cuda"}, "runs on cpu only"),
        ({"require_gradients": True}, "has no gradient route"),
        ({"target": "expectation"}, "has no stabilizer route"),
    ],
)
def test_a_request_the_tableau_cannot_serve_fails_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((CapabilityError, ValueError), match=message):
        fq.plan(_ghz(2), options=_options(**overrides))


def test_an_oversized_batch_is_refused_by_the_program_constraint() -> None:
    """The batch is pinned to the program's, so the refusal names that conflict."""

    with pytest.raises(ValueError, match="conflicts with the program batch constraint"):
        fq.plan(_ghz(2), options=_options(batch_size=3))


def test_a_batched_program_is_refused() -> None:
    """A tableau holds one circuit; a batch would be the same circuit again."""

    circuit = fq.Circuit(2, bsz=3).h(0).cx(0, 1)

    with pytest.raises(CapabilityError, match="bsz must be 1, got 3"):
        fq.plan(circuit, options=_options())


def test_a_run_without_shots_is_refused() -> None:
    with pytest.raises((CapabilityError, ValidationError), match="requires shots"):
        fq.run(_ghz(2), options=fq.ExecutionOptions(mode="stabilizer"))


def test_a_sharded_world_size_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tableau has no sharded layout, so the mode refuses rather than replicating."""

    monkeypatch.setenv("FQ_LOCAL_WORLD_SIZE", "2")

    with pytest.raises(CapabilityError, match="single-device representation"):
        fq.plan(_ghz(2), options=_options())


def test_the_expert_planner_owns_the_structural_refusals() -> None:
    """One authority: these fields belong to `plan_advanced`, so it checks them.

    The three checks below describe the representation rather than the request
    that asked for it, and every caller that builds a stabilizer plan -- `fq.plan`,
    the expert planner, and a bare `run_native` -- passes through this one
    function. Without them the expert planner would return a plan recording a
    batch, a gradient, or a rank count that the engine that runs it never
    produced.
    """

    with pytest.raises(CapabilityError, match="bsz must be 1, got 2"):
        plan_advanced(_ghz(4), state_mode="stabilizer", bsz=2)

    with pytest.raises(CapabilityError, match="has no gradient route"):
        plan_advanced(_ghz(4), state_mode="stabilizer", require_gradients=True)

    with pytest.raises(CapabilityError, match="single-device representation"):
        plan_advanced(_ghz(4), state_mode="stabilizer", world_size=2)


def test_a_bare_native_run_cannot_build_the_plan_the_planner_refuses() -> None:
    """The executor's own plan build reaches the same authority, not a bypass."""

    from dataclasses import replace

    from flagquantum.core.ir import MeasurementNode
    from flagquantum.runtime import run_native

    prepared = replace(
        _ghz(4).to_ir(),
        measurements=(MeasurementNode("sample", (0, 1, 2, 3), shots=8),),
    )

    with pytest.raises(CapabilityError, match="bsz must be 1, got 2"):
        run_native(prepared, mode="stabilizer", bsz=2)


def test_the_evidence_route_states_its_own_limits() -> None:
    plan = plan_from_dict(plan_to_dict(fq.plan(_ghz(3), options=_options())))
    summary = plan.summary()

    assert summary["state_mode"] == "stabilizer"
    assert summary["distribution_semantics"] == "single_device_fast_path"
    assert summary["scalability_claim_allowed"] is False
    assert summary["release_gate_allowed"] is False


def test_the_low_level_planner_records_the_representation() -> None:
    plan = plan_advanced(_ghz(5), state_mode="stabilizer", bsz=1)

    assert plan.state_mode == "stabilizer"
    assert plan.state_bytes == estimate_stabilizer_bytes(5)


def test_planning_refuses_a_measurement_the_tableau_cannot_answer() -> None:
    """The planner reads the program's measurement kinds, not just the target.

    `fq.plan` derives its target from the requested outputs, so an
    `expectation` node can only arrive inside the program -- which is exactly
    where `fq.plan` has to look for it rather than trusting the target name.
    """

    from dataclasses import replace

    from flagquantum.core.ir import MeasurementNode

    prepared = replace(
        _ghz(2).to_ir(),
        measurements=(MeasurementNode("expectation", (0,), shots=8),),
    )

    with pytest.raises(CapabilityError, match=r"measurement kind\(s\) expectation"):
        fq.plan(prepared, options=_options())


def test_the_executor_refuses_the_requests_planning_cannot_see() -> None:
    """The executor serves `run_native` directly, so it re-checks the request.

    A bare `run_native(..., mode="stabilizer")` never passes through
    `fq.plan`, so the option-level guard has not run. Both refusals below are
    therefore reachable, and each names the field the caller set wrongly rather
    than letting the engine raise something unrelated.
    """

    from dataclasses import replace

    from flagquantum.core.ir import MeasurementNode
    from flagquantum.runtime import run_native

    unsampled = replace(
        _ghz(2).to_ir(),
        measurements=(MeasurementNode("sample", (0, 1), shots=None),),
    )
    with pytest.raises(ValidationError, match="requires a positive shot count"):
        run_native(unsampled, mode="stabilizer")

    unanswerable = replace(
        _ghz(2).to_ir(),
        measurements=(MeasurementNode("expectation", (0,), shots=8),),
    )
    with pytest.raises(CapabilityError, match="cannot serve measurement kind"):
        run_native(unanswerable, mode="stabilizer")


def test_the_sampling_target_refuses_an_encoding_it_does_not_produce() -> None:
    """The sampler protocol allows `index`; this route only has bit strings.

    Returning bits under the `index` name would be a wrong answer rather than a
    missing one, so the target refuses the encoding it cannot honour instead of
    ignoring the argument.

    This is the one test in the file that draws samples rather than planning
    them, so it is also the one that needs the optional backend. The planning
    contract above holds with the engine absent and keeps running there.
    """

    pytest.importorskip("stim")

    from flagquantum.runtime.executors.stabilizer import StabilizerSamplingTarget

    target = StabilizerSamplingTarget(_ghz(2).to_ir(), seed=3)

    assert target.sample(4, format="bits").shape == (1, 4, 2)
    with pytest.raises(CapabilityError, match="no stabilizer route"):
        target.sample(4, format="index")
