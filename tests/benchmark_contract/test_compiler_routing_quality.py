"""Contract for the routing quality W7-16 publishes.

The benchmark module answers one question: what does each routing entry point the
Compiler accepts actually cost on a fixed basis? These tests hold the recorded
answer in place. They fail if an entry point stops routing the whole basis, if an
entry point emits an operation the device cannot host, if the published ordering
stops being total, or if the automatic selection stops being a selection -- either
by becoming an alias of one candidate, or by silently widening its scope.

Only a small seeded basis is measured here, so this stays a fast
``benchmark_contract`` test. The basis is checked to contain the behaviour the
recorded result turns on -- every resolution of the automatic selection, and at
least one program on which its cost estimate answers differently for the source
program and for the optimized one ``compile`` actually routes -- so it cannot
pass on a basis that never exercises them.

What this contract no longer holds is the check that the estimate sometimes picks
a *worse* candidate than one of its alternatives. That check was meaningful while
the candidate set was the two estimators, and it is now structurally
unsatisfiable: ``select_routing_strategy`` minimizes over
``DEPLOYABLE_ROUTING_STRATEGIES``, which includes ``sabre``, and pricing a
planner is planning it, so the price of ``sabre`` is the SWAP count its plan
actually inserts rather than an estimate of one. With an exactly-priced
candidate in the set and that candidate retaining least on most programs, the
choice lands on a best candidate on all 140 programs of the full basis as well as
on this one. The estimate's error is still measured -- it is what
``ratio_to_best_candidate`` would report above 1.0 -- but no basis in this
benchmark reaches it, so asserting it here would be asserting a property the
merged candidate set removed.
"""

import pytest

from benchmarks.compiler_lookahead_swap import case_basis
from benchmarks.compiler_routing_quality import (
    AUTOMATIC_SELECTION,
    MEASURED_STRATEGIES,
    OPTIMIZATION_LEVEL,
    retained_swap_count,
    run_benchmark,
)
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler import optimize
from flagquantum.compiler.optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    optimization_level_stages,
)
from flagquantum.compiler.routing import (
    DEPLOYABLE_ROUTING_STRATEGIES,
    ROUTING_STRATEGIES,
    select_routing_strategy,
)

pytestmark = pytest.mark.benchmark_contract

# Three topologies, chosen because each carries behaviour the recorded result
# turns on. ``grid3x3`` keeps a two-dimensional device in the basis, where the two
# SABRE strategies separate furthest from the two estimate-driven ones and where
# the largest program widths live. All three carry a program on which the cost
# estimate answers differently for the source program and for the routed one, and
# all three together exercise every resolution of the automatic selection.
_BASIS_TOPOLOGIES = ("line4", "ring6", "grid3x3")
_BASIS_SEEDS = (0, 1, 2, 3, 4)
_BASIS_CASE_COUNT = 60

# The strategies the automatic selection may resolve to. Read from the constant
# the selection itself minimizes over, because that constant is also what
# ``flagquantum/deployment/routing_evidence.py`` accepts, so a name cannot reach
# the choice without reaching the deployment contract.
_CANDIDATES = tuple(DEPLOYABLE_ROUTING_STRATEGIES)


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
    # of two, and the automatic selection lands on the cheaper of them rather than
    # between the two groups. It lands there because its candidate set contains
    # ``sabre`` and ``sabre`` is priced by planning it, so the price it is ranked
    # on is the SWAP count its plan inserts. A candidate set of only the two
    # estimators would put ``auto`` between the groups, which is the scope this
    # measurement exists to report.
    assert order[0] == "sabre_layout"
    assert by_label["sabre_layout"] < by_label["sabre"]
    # A fixed strategy's total is the sum of its own per-program counts, while the
    # automatic selection's total is the sum of the cheapest candidate's count on
    # each program, so the selection can be strictly below every candidate it is
    # allowed to pick. It is bounded by the cheapest candidate's total and not
    # equal to it, and the count of programs on which it retains more than a
    # candidate it could have picked is what would report the difference.
    assert by_label[AUTOMATIC_SELECTION] <= min(
        by_label[name] for name in _CANDIDATES
    ), "the automatic selection retained more than its cheapest candidate"
    assert by_label[AUTOMATIC_SELECTION] < min(
        by_label["restore_after_each_gate"], by_label["persistent_layout"]
    )
    assert _row(payload, "sabre")["ratio_to_best"] > 1.0
    assert _row(payload, "sabre_layout")["ratio_to_best"] == 1.0


def test_the_automatic_selection_resolves_to_the_candidates_its_estimate_ranks(
    payload: dict,
) -> None:
    selection = payload["automatic_selection"]
    assert selection["ranked_strategy_count"] == len(_CANDIDATES)
    assert len(_CANDIDATES) == len(DEPLOYABLE_ROUTING_STRATEGIES)
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
    ), f"this basis never exercised every resolution: {resolved}"
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


def test_the_cost_estimate_picks_a_best_candidate_on_every_program(
    payload: dict,
) -> None:
    selection = payload["automatic_selection"]
    decided = (
        selection["equal_to_best_candidate_case_count"]
        + selection["worse_than_best_candidate_case_count"]
    )
    assert decided == _BASIS_CASE_COUNT
    assert selection["equal_to_best_candidate_case_count"] == _BASIS_CASE_COUNT
    # This is the assertion the merged candidate set makes structural rather than
    # measured. ``worse_than_best_candidate_case_count`` is the count of programs
    # on which the estimate's choice retained more SWAPs than a candidate it could
    # have chosen instead; it is zero here and on the full 140-program basis,
    # because ``sabre`` is in the candidate set and its price is its plan. The
    # bound is asserted rather than the count, so a basis that did reach a wrong
    # call would still pass, and the ratio is what would report it.
    assert selection["worse_than_best_candidate_case_count"] == 0
    assert selection["better_than_both_candidate_case_count"] == 0
    assert selection["ratio_to_best_candidate"] == 1.0


def test_the_scope_of_the_cost_estimate_excludes_the_strategy_that_retains_least(
    payload: dict,
) -> None:
    selection = payload["automatic_selection"]
    # The number this measurement exists to publish. The scope of the automatic
    # selection is already stated in the capability manifest; its price was not.
    # The price is the one strategy the deployment contract does not accept:
    # ``auto`` may reach ``restore_after_each_gate``, ``persistent_layout`` and
    # ``sabre``, and ``sabre_layout`` is outside the set structurally, because it
    # starts from a layout the search chose. The ratio is therefore above 1.0 and
    # bounded by how much that excluded strategy would have saved, not above 2.0:
    # the 2x figure belongs to the candidate set this one replaced, which was the
    # two estimators alone.
    assert selection["ratio_to_best_available"] > 1.0
    assert (
        selection["best_available_retained_inserted_swap_count"]
        <= _row(payload, "sabre_layout")["retained_inserted_swap_count"]
    ), "the best available strategy cannot retain more than sabre_layout does"


def test_the_measured_automatic_selection_is_the_one_compile_actually_makes(
    payload: dict,
) -> None:
    """The published ``auto`` cost must be the cost of the selection ``compile`` makes.

    ``compile`` optimizes a program and then selects a strategy for the optimized
    program, so the estimate never sees the caller's source program. A measurement
    that selected on the source program instead would publish the cost of a route
    no caller receives -- and this basis is chosen to contain that difference
    rather than to assume it away.
    """

    cases = case_basis(topologies=_BASIS_TOPOLOGIES, seeds=_BASIS_SEEDS)
    recorded = {
        record["case"]: record["resolved_strategy"]
        for record in payload["case_records"]
        if record["strategy"] == AUTOMATIC_SELECTION
    }
    assert len(recorded) == _BASIS_CASE_COUNT == len(cases)

    differs_on_the_source_program = 0
    for case in cases:
        program = case.program()
        device = case.device()
        selected = select_routing_strategy(optimize(program), device).selected_strategy
        assert selected == recorded[case.label], (
            f"{case.label}: the benchmark recorded {recorded[case.label]} but "
            f"compile selects {selected} for the program it routes"
        )
        if (
            select_routing_strategy(program, case.device()).selected_strategy
            != selected
        ):
            differs_on_the_source_program += 1

    # Non-vacuity: the two places the estimate could be run apart on this basis, so
    # the assertion above is not trivially satisfied by them always agreeing.
    assert differs_on_the_source_program > 0, (
        "this basis cannot distinguish selecting on the optimized program from "
        "selecting on the source program, so it cannot hold the recorded choice"
    )


def test_the_recorded_counts_are_measured_at_the_declared_optimization_level(
    payload: dict,
) -> None:
    """A retained count depends on how much optimization ran after routing.

    The recorded number is what is left *after* the passes that remove inserted
    SWAPs, so a run at a level that runs fewer of them publishes a larger number
    under the same schema. The level is therefore part of the reading, not an
    implementation detail, and it is asserted here rather than left to whatever
    the default happened to be when the number was taken.
    """

    assert payload["optimization_level"] == DEFAULT_OPTIMIZATION_LEVEL

    # Non-vacuity: the basis must be able to tell the levels apart, or the field
    # above would pin a reading that no level can change. The lower levels that
    # run strictly fewer passes are replayed over the whole basis and their total
    # has to come out different, in the direction the assertion claims: fewer
    # passes remove fewer inserted SWAPs, so a lower level retains more.
    case = case_basis(topologies=_BASIS_TOPOLOGIES, seeds=_BASIS_SEEDS)
    records = {
        record["case"]: record["retained_inserted_swap_count"]
        for record in payload["case_records"]
        if record["strategy"] == "restore_after_each_gate"
    }
    recorded_total = sum(records.values())
    lower_levels = [
        level
        for level in range(OPTIMIZATION_LEVEL)
        if optimization_level_stages(level)
        != optimization_level_stages(OPTIMIZATION_LEVEL)
    ]
    assert lower_levels, "the declared level runs the same passes as every lower one"

    totals = {}
    for level in lower_levels:
        totals[level] = sum(
            retained_swap_count(
                compile_program(
                    program,
                    coupling_map=device,
                    routing_strategy="restore_after_each_gate",
                    optimization_level=level,
                )
            )
            for program, device in ((item.program(), item.device()) for item in case)
        )
    assert totals[lower_levels[-1]] != recorded_total, (
        f"this basis retains {recorded_total} SWAPs at level {OPTIMIZATION_LEVEL} "
        f"and {totals[lower_levels[-1]]} at level {lower_levels[-1]}, so it does "
        f"not exercise the level it claims to record"
    )
    assert min(totals.values()) >= recorded_total, (
        f"a lower optimization level removed SWAPs level {OPTIMIZATION_LEVEL} kept: "
        f"{totals} against {recorded_total}"
    )
