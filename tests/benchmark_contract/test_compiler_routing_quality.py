"""Contract for the routing quality W7-16 publishes.

The benchmark module answers one question: what does each routing entry point the
Compiler accepts actually cost on a fixed basis? These tests hold the recorded
answer in place. They fail if an entry point stops routing the whole basis, if an
entry point emits an operation the device cannot host, if the published ordering
stops being total, or if the automatic selection stops being a selection -- either
by becoming an alias of one candidate, or by silently widening its scope.

Only a small seeded basis is measured here, so this stays a fast
``benchmark_contract`` test. The basis is checked to contain the behaviour the
recorded result turns on -- both resolutions of the automatic selection, and at
least one program where its cost estimate picks the more expensive candidate --
so it cannot pass on a basis that never exercises them.
"""

from dataclasses import fields

import pytest

from benchmarks.compiler_routing_quality import (
    AUTOMATIC_SELECTION,
    MEASURED_STRATEGIES,
    run_benchmark,
)
from flagquantum.compiler.routing import ROUTING_STRATEGIES, RoutingStrategySelection

pytestmark = pytest.mark.benchmark_contract

# ``line4`` and ``ring6`` are where the cost estimate picks the wrong candidate;
# ``grid3x3`` keeps a two-dimensional device in the basis, where the two SABRE
# strategies separate furthest from the two estimate-driven ones.
_BASIS_TOPOLOGIES = ("line4", "ring6", "grid3x3")
_BASIS_SEEDS = (0, 1, 2)
_BASIS_CASE_COUNT = 36

# The strategies the automatic selection may resolve to. Read from the selection
# type rather than restated, because that dataclass is what decides them.
_CANDIDATES = tuple(
    field.name
    for field in fields(RoutingStrategySelection)
    if field.name != "selected_strategy"
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark(topologies=_BASIS_TOPOLOGIES, seeds=_BASIS_SEEDS)


def _row(payload: dict, label: str) -> dict:
    rows = [row for row in payload["strategies"] if row["label"] == label]
    assert len(rows) == 1, f"expected exactly one row for {label}"
    return rows[0]


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == "flagquantum_compiler_routing_quality_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["entry_point"] == "flagquantum.compiler.compile"


def test_measurement_covers_every_entry_point_the_compiler_accepts(
    payload: dict,
) -> None:
    # A fifth strategy cannot ship unmeasured: the measured set is derived from
    # ROUTING_STRATEGIES, and this fails when the benchmark stops deriving it.
    assert payload["measured_strategies"] == [
        *ROUTING_STRATEGIES,
        AUTOMATIC_SELECTION,
    ]
    assert payload["case_count"] == _BASIS_CASE_COUNT
    assert payload["topologies"] == list(_BASIS_TOPOLOGIES)
    assert payload["seeds"] == list(_BASIS_SEEDS)
    # Every label is measured on every case, and every compiled program is
    # compared with its source state, so no label can quietly drop out of the
    # basis. A label that failed closed produces no program to compare.
    assert payload["verified_case_count"] == _BASIS_CASE_COUNT * len(
        MEASURED_STRATEGIES
    )
    assert len(payload["case_records"]) == _BASIS_CASE_COUNT * len(MEASURED_STRATEGIES)
    assert payload["max_state_difference"] < 1e-12


def test_every_entry_point_routes_the_whole_basis_legally(payload: dict) -> None:
    for row in payload["strategies"]:
        label = row["label"]
        assert row["failed_case_count"] == 0, row["failed_cases"]
        assert row["planned_case_count"] == _BASIS_CASE_COUNT, label
        assert row["off_device_two_wire_instruction_count"] == 0, label
        assert row["retained_inserted_swap_count"] > 0, label
    for record in payload["case_records"]:
        assert record["off_device_two_wire_instruction_count"] == 0, record["case"]


def test_published_order_is_total_and_ranks_the_two_search_strategies_first(
    payload: dict,
) -> None:
    order = payload["quality_order"]
    assert sorted(order) == sorted(
        MEASURED_STRATEGIES
    ), "the published order must rank every measured entry point exactly once"
    by_label = {
        row["label"]: row["retained_inserted_swap_count"]
        for row in payload["strategies"]
    }
    values = [by_label[label] for label in order]
    assert values == sorted(values), f"quality_order is not the measured order: {order}"
    # The two search strategies are the reason the ordering is worth publishing:
    # they separate from the two estimate-driven strategies by more than a factor
    # of two, and the automatic selection sits between the two groups.
    assert order[0] == "sabre_layout"
    assert by_label["sabre_layout"] < by_label["sabre"]
    assert by_label["sabre"] < by_label[AUTOMATIC_SELECTION]
    assert by_label[AUTOMATIC_SELECTION] < min(
        by_label["restore_after_each_gate"], by_label["persistent_layout"]
    )
    assert _row(payload, "sabre")["ratio_to_best"] > 1.0
    assert _row(payload, "sabre_layout")["ratio_to_best"] == 1.0


def test_the_automatic_selection_resolves_to_the_candidates_its_estimate_ranks(
    payload: dict,
) -> None:
    selection = payload["automatic_selection"]
    assert selection["ranked_strategy_count"] == len(_CANDIDATES) == 2
    resolved = selection["resolved_strategy_counts"]
    assert set(resolved) <= set(_CANDIDATES), (
        f"the automatic selection resolved to {sorted(resolved)}, which its cost "
        f"estimate cannot rank ({list(_CANDIDATES)})"
    )
    assert sum(resolved.values()) == _BASIS_CASE_COUNT
    # Non-vacuity: a basis on which the estimate always makes the same call would
    # record a resolution without ever exercising the choice.
    assert all(
        resolved.get(label, 0) > 0 for label in _CANDIDATES
    ), f"this basis never exercised both resolutions: {resolved}"
    for row in payload["strategies"]:
        expected = (
            resolved
            if row["label"] == AUTOMATIC_SELECTION
            else {row["label"]: _BASIS_CASE_COUNT}
        )
        assert row["resolved_strategy_counts"] == expected, row["label"]


def test_the_automatic_selection_returns_the_route_of_the_candidate_it_names(
    payload: dict,
) -> None:
    # ``auto`` is a selection, not a third router: the program it returns has to
    # be the program the candidate it resolved to returns, or the published cost
    # would belong to an algorithm no caller can name.
    retained: dict[str, dict[str, int]] = {}
    resolved: dict[str, str] = {}
    for record in payload["case_records"]:
        retained.setdefault(record["case"], {})[record["strategy"]] = record[
            "retained_inserted_swap_count"
        ]
        if record["strategy"] == AUTOMATIC_SELECTION:
            resolved[record["case"]] = record["resolved_strategy"]
    assert len(resolved) == _BASIS_CASE_COUNT
    for case, strategy in resolved.items():
        assert retained[case][AUTOMATIC_SELECTION] == retained[case][strategy], (
            f"{case}: the automatic selection resolved to {strategy} but retained "
            f"{retained[case][AUTOMATIC_SELECTION]} SWAPs against {strategy}'s "
            f"{retained[case][strategy]}"
        )
    selection = payload["automatic_selection"]
    assert selection["better_than_both_candidate_case_count"] == 0


def test_the_cost_estimate_picks_the_cheaper_candidate_on_most_programs(
    payload: dict,
) -> None:
    selection = payload["automatic_selection"]
    decided = (
        selection["equal_to_best_candidate_case_count"]
        + selection["worse_than_best_candidate_case_count"]
    )
    assert decided == _BASIS_CASE_COUNT
    assert selection["equal_to_best_candidate_case_count"] > (
        selection["worse_than_best_candidate_case_count"]
    )
    # Non-vacuity: the estimate is measured to be wrong sometimes, so a basis
    # that happened to contain only its good calls could not support the claim.
    assert selection["worse_than_best_candidate_case_count"] > 0
    assert selection["ratio_to_best_candidate"] < 1.05


def test_the_scope_of_the_cost_estimate_costs_twice_the_best_available_strategy(
    payload: dict,
) -> None:
    selection = payload["automatic_selection"]
    # The number this measurement exists to publish. The scope of the automatic
    # selection is already stated in the capability manifest; its price was not.
    assert selection["ratio_to_best_available"] > 2.0
    assert (
        selection["best_available_retained_inserted_swap_count"]
        <= _row(payload, "sabre_layout")["retained_inserted_swap_count"]
    ), "the best available strategy cannot retain more than sabre_layout does"
