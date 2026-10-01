"""Contract for the lookahead SWAP measurement W7-04 records.

The benchmark module answers one question: should FlagQuantum expose the SWAP
search Qiskit ships, Jandura's lookahead beam search, as another routing
strategy? The recorded answer is no, and these tests hold the evidence for that
answer in place. They fail if the reference search stops being a faithful
replacement for the shipped planner, if the published objective starts or stops
being partial, or if any objective starts beating ``sabre_layout``.

Only a small seeded basis is measured here, so this stays a fast
``benchmark_contract`` test. The basis is chosen to contain the failure mode the
recorded result turns on, and the test asserts that it does, so it cannot pass on
a basis that never exercises it.
"""

import pytest

from benchmarks.compiler_lookahead_swap import (
    DEFAULT_CONFIGURATIONS,
    Configuration,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

# ``line6`` and ``line9`` are where the published objective strands an operation;
# ``line4`` keeps a connected small device in the basis.
_BASIS_TOPOLOGIES = ("line4", "line6", "line9")
_BASIS_SEEDS = (0, 1)
_BASIS_CASE_COUNT = 24

_CONFIGURATIONS = (
    Configuration("sabre", "sabre"),
    Configuration("sabre_layout", "sabre_layout"),
    Configuration("sabre/d1w1", "sabre", "sabre", 1, 1),
    Configuration("published/d4w4", "sabre", "published", 4, 4),
    Configuration("published/d4w4+repair", "sabre", "published", 4, 4, repair=True),
    Configuration("front/d4w4", "sabre", "front", 4, 4),
    Configuration("sabre/d3w2", "sabre", "sabre", 3, 2),
    Configuration("layout/sabre/d4w2", "sabre_layout", "sabre", 4, 2),
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
    assert payload["schema"] == "flagquantum_compiler_lookahead_swap_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["reference_algorithm"] == "qiskit_lookahead_swap_beam_search"


def test_measurement_covers_the_declared_basis(payload: dict) -> None:
    assert payload["case_count"] == _BASIS_CASE_COUNT
    assert payload["topologies"] == list(_BASIS_TOPOLOGIES)
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


def test_reference_search_reproduces_the_shipped_planner_at_depth_and_width_one(
    payload: dict,
) -> None:
    anchor = payload["fidelity_anchor"]
    assert anchor["configuration"] == "sabre/d1w1"
    assert anchor["case_count"] == _BASIS_CASE_COUNT
    # Non-vacuity: the anchor compares two planners, so a basis the search cannot
    # plan at all would show up as a shortfall here rather than as agreement.
    assert anchor["equal_planned_swap_count_case_count"] == _BASIS_CASE_COUNT
    assert anchor["identical_inserted_swap_sequence_case_count"] == _BASIS_CASE_COUNT
    assert (
        _row(payload, "sabre/d1w1")["failed_case_count"] == 0
    ), "the anchor must plan every case it is compared on"


def test_published_objective_fails_closed_where_a_single_operation_is_stranded(
    payload: dict,
) -> None:
    published = _row(payload, "published/d4w4")
    assert published["failed_case_count"] > 0, (
        "the recorded negative result depends on the published objective being "
        "partial; a basis where it terminates cannot support the conclusion"
    )
    assert published["planned_case_count"] + published["failed_case_count"] == (
        _BASIS_CASE_COUNT
    )
    for failure in published["failed_cases"]:
        assert "returns the same state it already held" in failure
        assert "operations still blocked" in failure


def test_the_two_total_objectives_plan_the_cases_the_published_one_cannot(
    payload: dict,
) -> None:
    published = _row(payload, "published/d4w4")
    for label in ("front/d4w4", "sabre/d3w2"):
        row = _row(payload, label)
        assert row["failed_case_count"] == 0
        assert row["planned_case_count"] == _BASIS_CASE_COUNT
    # The failure belongs to the published ranking, not to the beam search: on
    # every case it strands, both other objectives produce a route.
    stranded = {
        record["case"]
        for record in payload["case_records"]
        if record["configuration"] == "published/d4w4" and record["failed"]
    }
    assert len(stranded) == published["failed_case_count"]
    for label in ("front/d4w4", "sabre/d3w2"):
        planned = {
            record["case"]
            for record in payload["case_records"]
            if record["configuration"] == label and not record["failed"]
        }
        assert stranded <= planned


def test_repairing_the_published_objective_costs_more_than_the_shipped_strategy(
    payload: dict,
) -> None:
    repaired = _row(payload, "published/d4w4+repair")
    assert repaired["failed_case_count"] == 0
    assert repaired["ratio_to_sabre_layout"] > 1.2


def test_no_objective_beats_the_shipped_layout_search(payload: dict) -> None:
    # Searching SABRE's own cost does improve on the greedy SWAP SABRE takes, and
    # still loses to the layout search the product selects.
    searched = _row(payload, "sabre/d3w2")
    assert searched["ratio_to_sabre"] <= 1.0
    assert searched["ratio_to_sabre_layout"] > 1.05
    # The closest the port comes is inside the layout search, and the margin is
    # bounded well below the shipped strategy.
    substituted = _row(payload, "layout/sabre/d4w2")
    assert substituted["failed_case_count"] == 0
    assert substituted["ratio_to_sabre_layout"] < 1.03


def test_measurement_reports_no_off_device_two_wire_instruction(payload: dict) -> None:
    for row in payload["configurations"]:
        assert row["off_device_two_wire_instruction_count"] == 0, row["label"]


def test_default_configurations_include_the_shipped_baselines() -> None:
    labels = {configuration.label for configuration in DEFAULT_CONFIGURATIONS}
    assert {"sabre", "sabre_layout"} <= labels
    assert "sabre/d1w1" in labels
