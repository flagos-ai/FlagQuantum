"""Contract for the stochastic SWAP measurement W7-06 records.

The benchmark module answers one question: should FlagQuantum expose the SWAP
search Qiskit removed in 2.0, Bravyi's randomized layer permutation, as another
routing strategy? The recorded answer is no, and these tests hold the evidence for
that answer in place. They fail if the port stops producing plans whose recorded
placements are the replay of their own SWAPs, if it starts emitting a two-wire
instruction off the device, or if it starts beating the shipped ``sabre_layout``
strategy.

Only a small seeded basis is measured here, so this stays a fast
``benchmark_contract`` test. The basis includes ``grid4x4`` because that is where
the trial count changes the outcome; the test asserts that it does, so it cannot
pass on a basis that can never distinguish one trial from twenty.
"""

import dataclasses

import pytest

from benchmarks.compiler_stochastic_swap import (
    DEFAULT_CONFIGURATIONS,
    Configuration,
    check_plan,
    plan_stochastic_swaps,
    run_benchmark,
)
from flagquantum.compiler import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.benchmark_contract

# ``grid4x4`` is where the trial count matters; ``line4`` and ``grid3x3`` keep a
# connected small device and a grid in the basis.
_BASIS_TOPOLOGIES = ("line4", "grid3x3", "grid4x4")
_BASIS_SEEDS = (0, 1)
_BASIS_CASE_COUNT = 24

_SHIPPED = Configuration("sabre", "sabre"), Configuration(
    "sabre_layout", "sabre_layout"
)
_CONFIGURATIONS = (
    *_SHIPPED,
    Configuration("stochastic/t1", "sabre", 1),
    Configuration("stochastic/t20", "sabre", 20),
    Configuration("layout/stochastic/t20", "sabre_layout", 20),
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark(
        topologies=_BASIS_TOPOLOGIES,
        seeds=_BASIS_SEEDS,
        configurations=_CONFIGURATIONS,
    )


def _row(payload: dict, label: str) -> dict:
    rows = [row for row in payload["configurations"] if row["label"] == label]
    assert len(rows) == 1, f"expected exactly one row for {label}"
    return rows[0]


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == "flagquantum_compiler_stochastic_swap_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert (
        payload["reference_algorithm"]
        == "qiskit_stochastic_swap_randomized_layer_permutation"
    )
    # The pass this ports was removed in Qiskit 2.0, which the repository certifies.
    assert payload["reference_revision"].startswith("qiskit 1.3.0")


def test_measurement_covers_the_declared_basis(payload: dict) -> None:
    assert payload["case_count"] == _BASIS_CASE_COUNT
    assert payload["topologies"] == sorted(_BASIS_TOPOLOGIES)
    assert payload["seeds"] == list(_BASIS_SEEDS)
    # Every configuration is measured on every case and every compiled program is
    # compared with its source state, so a configuration cannot silently drop out
    # of the basis. A configuration that fails closed on a case produces no
    # compiled program to compare, so it is counted out of both.
    assert payload["verified_case_count"] == sum(
        row["planned_case_count"] for row in payload["configurations"]
    )
    assert len(payload["case_records"]) == _BASIS_CASE_COUNT * len(_CONFIGURATIONS)
    assert payload["max_state_difference"] < 1e-12


def test_shipped_strategies_are_the_baseline_every_ratio_uses(payload: dict) -> None:
    shipped = payload["shipped"]
    assert shipped["sabre"]["retained_inserted_swap_count"] > 0
    assert (
        shipped["sabre_layout"]["retained_inserted_swap_count"]
        < shipped["sabre"]["retained_inserted_swap_count"]
    )
    assert _row(payload, "sabre_layout")["ratio_to_sabre_layout"] == 1.0
    assert _row(payload, "sabre")["ratio_to_sabre"] == 1.0


def test_every_plan_the_port_returns_is_the_replay_of_its_own_swaps(
    payload: dict,
) -> None:
    replay = payload["plan_replay"]
    expected = _BASIS_CASE_COUNT * sum(
        1 for configuration in _CONFIGURATIONS if configuration.trials
    )
    # Non-vacuity: the check compares a plan against an independent replay, so a
    # plan the port refuses to produce would show up as a shortfall here rather
    # than as agreement.
    assert replay["case_count"] == expected
    assert replay["placements_replay_case_count"] == expected
    assert replay["two_wire_placements_on_device_case_count"] == expected


def test_the_port_loses_to_the_shipped_layout_search(payload: dict) -> None:
    routed = _row(payload, "stochastic/t20")
    # A planner that failed closed would retain fewer SWAPs for the wrong reason,
    # so the loss only counts if the planner produced a route for every case.
    assert routed["failed_case_count"] == 0
    assert routed["planned_case_count"] == _BASIS_CASE_COUNT
    assert routed["planned_inserted_swap_count"] > 0
    assert routed["better_than_sabre_layout_case_count"] == 0
    assert routed["ratio_to_sabre_layout"] > 1.5
    # It does not even beat the greedy SWAP search the layout search generalizes.
    assert routed["ratio_to_sabre"] > 1.3


def test_more_trials_do_not_close_the_gap(payload: dict) -> None:
    one = _row(payload, "stochastic/t1")
    twenty = _row(payload, "stochastic/t20")
    assert one["failed_case_count"] == 0
    assert twenty["ratio_to_sabre_layout"] > 1.5
    # More trials do help, and not enough to matter. The basis must show the
    # difference, or this test would pass on a basis where the trial count is not
    # exercised at all -- which is an assertion about the basis, not the port.
    assert twenty["ratio_to_sabre_layout"] < one["ratio_to_sabre_layout"]
    robustness = {row["label"]: row for row in payload["trial_robustness"]}
    assert (
        robustness["stochastic/t1"]["equal_retained_case_count_to_stochastic_t20"]
        < _BASIS_CASE_COUNT
    ), "the basis cannot distinguish one trial from twenty, so it proves nothing"


def test_substituting_the_search_into_the_layout_search_is_still_a_regression(
    payload: dict,
) -> None:
    substituted = _row(payload, "layout/stochastic/t20")
    assert substituted["failed_case_count"] == 0
    # The closest the port comes is inside the layout search, and it is still
    # further from the shipped strategy than the shipped strategy is from itself.
    assert substituted["ratio_to_sabre_layout"] > 1.2


def test_measurement_reports_no_off_device_two_wire_instruction(payload: dict) -> None:
    for row in payload["configurations"]:
        assert row["off_device_two_wire_instruction_count"] == 0, row["label"]


def test_plan_consistency_check_rejects_a_plan_that_was_altered() -> None:
    # The replay anchor above is only evidence if the check it reports can fail.
    program = CircuitIR(
        5,
        (
            Instruction("cz", (0, 4)),
            Instruction("cz", (0, 2)),
            Instruction("cz", (1, 3)),
        ),
        dtype="complex128",
    )
    device = CouplingMap.line(5)
    plan = plan_stochastic_swaps(program, device, trials=20)
    consistent = check_plan(program, plan, device)
    assert consistent.placements_replay is True
    assert consistent.two_wire_placements_on_device is True
    # ``cz(0, 4)`` is two hops apart on a five-wire line, so routing it takes at
    # least one SWAP for dropping the plan's SWAPs to be observable.
    assert plan.swaps, "the plan must insert a SWAP before it can be corrupted"
    assert (
        check_plan(
            program, dataclasses.replace(plan, swaps=()), device
        ).placements_replay
        is False
    )
    placements = list(plan.placements)
    assert device.has_edge(0, 3) is False
    placements[0] = (0, 3)
    moved = check_plan(
        program, dataclasses.replace(plan, placements=tuple(placements)), device
    )
    assert moved.two_wire_placements_on_device is False


def test_default_configurations_include_the_shipped_baselines_and_the_trials() -> None:
    labels = {configuration.label for configuration in DEFAULT_CONFIGURATIONS}
    assert {"sabre", "sabre_layout"} <= labels
    assert {"stochastic/t1", "stochastic/t5", "stochastic/t20", "stochastic/t100"} <= (
        labels
    )
    assert "layout/stochastic/t20" in labels
