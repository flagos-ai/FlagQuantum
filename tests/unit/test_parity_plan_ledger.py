"""Guard the machine-readable row ledger of the CUDA-Q parity plan.

The plan itself is prose and it is not committed here. Its own blocking table
abbreviates the accelerator rows as ``W2-01``…``W2-08``, so no parser can read an
exact set of unfinished rows out of it, and a grep for a landing marker cannot
tell a row that is blocked from a row that is merely unannotated. Both of those
have produced wrong counts in the plan. The ledger is the enumerated record, and
these tests fail closed on it being wrong in either direction: a row that quietly
disappears, and a row that quietly becomes done.
"""

from __future__ import annotations

import copy

import pytest

from tools.ci_tier import CI_TIERS
from tools.parity_plan_ledger import (
    BLOCKER_VOCABULARY,
    KIND_VOCABULARY,
    LEDGER_PATH,
    SIZE_VOCABULARY,
    STATUS_VOCABULARY,
    _load,
    summarize,
    validate,
)

pytestmark = pytest.mark.unit

# The plan's own row total, which its count paragraph reports. This is the one
# number restated here rather than derived: everything else about the ledger's
# shape is checked as an internal property, but a total that shrinks by one is
# invisible to any internal check, so it is pinned against the plan.
DECLARED_ROW_TOTAL = 102

# The rows the plan names as waiting on an owner ruling, and the rows it names as
# waiting on hardware this machine does not have. The plan writes the second
# group as an ellipsis, so it is spelled out in the ledger and pinned here.
DECLARED_OWNER_BLOCKED = {
    "W0-07",
    "W1-06",
    "W4-05",
    "W5-01",
    "W5-02",
    "W5-10",
    "W5-11",
    "W5-13",
}
DECLARED_ACCELERATOR_BLOCKED = {
    "W0-10",
    "W2-01",
    "W2-02",
    "W2-03",
    "W2-04",
    "W2-05",
    "W2-06",
    "W2-07",
    "W2-08",
    "W3-02",
    "W3-03",
    "W3-04",
    "W3-05",
    "W4-04",
    "W4-07",
}

# A landed row whose own text carries no landing note. The plan reports two
# different landed totals depending on whether it counts these, and both are
# honest answers to different questions, so the distinction is data here.
DECLARED_SILENT_LANDED = {"W0-05", "W1-07a", "W3-01"}


@pytest.fixture()
def document() -> dict:
    return _load(LEDGER_PATH)


def test_the_ledger_is_sound(document: dict) -> None:
    assert validate(document) == []


def test_the_ledger_covers_every_row_the_plan_lists(document: dict) -> None:
    assert document["row_count"] == len(document["rows"]) == DECLARED_ROW_TOTAL


def test_no_row_was_dropped_from_the_middle_of_a_wave(document: dict) -> None:
    # Row indices within a wave are gapless, and an index is used once -- or
    # twice, and then by ``a`` and ``b``. A deleted row leaves a hole, so this
    # holds without restating how wide each wave is.
    waves: dict[str, list[tuple[int, str]]] = {}
    for row in document["rows"]:
        wave, number = row["id"].split("-")
        waves.setdefault(wave, []).append((int(number[:2]), number[2:]))
    for wave, entries in waves.items():
        indices = sorted({index for index, _ in entries})
        assert indices == list(range(indices[0], indices[0] + len(indices))), wave
        for index in indices:
            splits = sorted(split for candidate, split in entries if candidate == index)
            assert splits in ([""], ["a", "b"]), (wave, index, splits)


def test_every_row_id_is_unique_and_names_its_wave(document: dict) -> None:
    identifiers = [row["id"] for row in document["rows"]]
    assert identifiers == sorted(set(identifiers), key=identifiers.index)
    assert len(set(identifiers)) == len(identifiers)


def test_the_blocked_rows_are_the_ones_the_plan_names(document: dict) -> None:
    blocked = {
        row["id"]: row["blocker"]
        for row in document["rows"]
        if row["status"] == "blocked"
    }
    assert set(blocked) == DECLARED_OWNER_BLOCKED | DECLARED_ACCELERATOR_BLOCKED
    assert {
        name for name, why in blocked.items() if why == "owner_decision_required"
    } == (DECLARED_OWNER_BLOCKED)
    assert {
        name
        for name, why in blocked.items()
        if why == "accelerator_environment_unavailable"
    } == DECLARED_ACCELERATOR_BLOCKED
    assert not DECLARED_OWNER_BLOCKED & DECLARED_ACCELERATOR_BLOCKED


def test_the_unfinished_total_is_the_sum_of_its_two_conditions(document: dict) -> None:
    statuses = [row["status"] for row in document["rows"]]
    unfinished = [status for status in statuses if status != "landed"]
    blocked = DECLARED_OWNER_BLOCKED | DECLARED_ACCELERATOR_BLOCKED
    assert len(unfinished) == len(blocked)
    assert len(statuses) - len(unfinished) == DECLARED_ROW_TOTAL - len(blocked)


def test_the_two_landed_totals_are_both_derivable(document: dict) -> None:
    landed = [row for row in document["rows"] if row["status"] == "landed"]
    silent = {row["id"] for row in landed if not row["annotated"]}
    assert silent == DECLARED_SILENT_LANDED
    # The plan quotes both a strict and a complementary landed total, and the
    # difference between them is exactly these rows, so neither number needs to
    # be restated: each is a function of the other and of this set.
    assert len(landed) == DECLARED_ROW_TOTAL - len(
        DECLARED_OWNER_BLOCKED | DECLARED_ACCELERATOR_BLOCKED
    )
    assert sum(1 for row in landed if row["annotated"]) == len(landed) - len(silent)


def test_no_pending_row_is_left_unclassified(document: dict) -> None:
    # A row that is neither landed nor blocked is a row the plan has not
    # scheduled. The ledger carries no such row, which is what makes the
    # blocking table an exhaustive statement about what is left.
    assert [row["id"] for row in document["rows"] if row["status"] == "pending"] == []


@pytest.mark.parametrize("kind", sorted(KIND_VOCABULARY))
def test_every_declared_kind_is_one_a_row_actually_carries(
    document: dict, kind: str
) -> None:
    carried = {entry for row in document["rows"] for entry in row["kind"]}
    assert carried <= KIND_VOCABULARY
    assert kind in carried


def test_a_row_kind_that_names_no_work_is_refused(document: dict) -> None:
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["kind"] = ["miscellaneous"]
    assert any("is not a work kind" in problem for problem in validate(mutated))


def test_a_gate_tier_must_be_a_tier_or_a_marker(document: dict) -> None:
    used = {tier for row in document["rows"] for tier in row["gate_tier"]}
    assert used
    # Every name a row uses is a tier a workflow runs or a marker a lane selects.
    assert used & set(CI_TIERS)
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["gate_tier"] = ["the long lane"]
    assert any(
        "is neither a tier name nor a pytest marker" in problem
        for problem in validate(mutated)
    )


def test_a_size_must_be_a_row_size(document: dict) -> None:
    used = {row["size"] for row in document["rows"]}
    assert used <= SIZE_VOCABULARY
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["size"] = "enormous"
    assert any("is not a row size" in problem for problem in validate(mutated))


def test_a_status_must_be_a_row_status(document: dict) -> None:
    used = {row["status"] for row in document["rows"]}
    assert used <= STATUS_VOCABULARY
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["status"] = "in-progress"
    assert any("is not a row status" in problem for problem in validate(mutated))


def test_a_blocked_row_must_say_why_and_a_landed_row_must_not(document: dict) -> None:
    used = {row["blocker"] for row in document["rows"] if "blocker" in row}
    assert used == BLOCKER_VOCABULARY

    silent = copy.deepcopy(document)
    blocked = next(row for row in silent["rows"] if row["status"] == "blocked")
    del blocked["blocker"]
    assert any(
        "a blocked row must name a blocker" in problem for problem in validate(silent)
    )

    overclaimed = copy.deepcopy(document)
    landed = next(row for row in overclaimed["rows"] if row["status"] == "landed")
    landed["blocker"] = "owner_decision_required"
    assert any(
        "must not name a blocker" in problem for problem in validate(overclaimed)
    )


def test_a_landing_note_on_a_row_with_nothing_to_record_is_refused(
    document: dict,
) -> None:
    # A row that is neither landed nor blocked records no landing note, so it
    # cannot be annotated. The ledger has no such row today; the rule is what
    # keeps one from appearing already annotated when the plan schedules it.
    mutated = copy.deepcopy(document)
    mutable = next(row for row in mutated["rows"] if row["annotated"])
    mutable["status"] = "pending"
    assert any("records no landing note" in problem for problem in validate(mutated))


def test_a_dependency_must_name_a_row_that_exists(document: dict) -> None:
    mutated = copy.deepcopy(document)
    mutated["rows"][-1]["depends_on"] = ["W9-99"]
    assert any("which no row declares" in problem for problem in validate(mutated))


def test_a_dependency_cannot_point_forward(document: dict) -> None:
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["depends_on"] = [mutated["rows"][-1]["id"]]
    assert any("cannot point forward" in problem for problem in validate(mutated))


def test_a_dependency_cycle_is_refused(document: dict) -> None:
    mutated = copy.deepcopy(document)
    first, second = mutated["rows"][0], mutated["rows"][1]
    first["depends_on"] = [second["id"]]
    second["depends_on"] = [first["id"]]
    assert any("dependency cycle" in problem for problem in validate(mutated))


def test_a_row_cannot_depend_on_itself(document: dict) -> None:
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["depends_on"] = [mutated["rows"][0]["id"]]
    assert any("depends on itself" in problem for problem in validate(mutated))


def test_a_missing_row_is_not_a_quietly_smaller_ledger(document: dict) -> None:
    mutated = copy.deepcopy(document)
    del mutated["rows"][5]
    assert any("row_count" in problem for problem in validate(mutated))


def test_a_duplicated_row_is_refused(document: dict) -> None:
    mutated = copy.deepcopy(document)
    mutated["rows"].append(copy.deepcopy(mutated["rows"][0]))
    mutated["row_count"] = len(mutated["rows"])
    assert any("appears more than once" in problem for problem in validate(mutated))


def test_the_ledger_records_no_second_copy_of_path_ownership(document: dict) -> None:
    # `team-ownership.toml` owns path ownership. A team column here would be a
    # second authority and would preserve, rather than surface, the plan's
    # mentions of teams the roster does not declare.
    for row in document["rows"]:
        assert "team" not in row
        assert "path" not in row
        assert "owner" not in row


def test_the_summary_reports_the_totals_the_plan_argues_about(document: dict) -> None:
    summary = summarize(document)
    landed = DECLARED_ROW_TOTAL - len(
        DECLARED_OWNER_BLOCKED | DECLARED_ACCELERATOR_BLOCKED
    )
    assert f"{DECLARED_ROW_TOTAL} rows across 8 waves" in summary
    assert (
        f"landed: {landed} ({landed - len(DECLARED_SILENT_LANDED)} carry their own "
        f"landing note, {len(DECLARED_SILENT_LANDED)} do not:"
    ) in summary
    assert "blocked: 23 (accelerator_environment_unavailable:15" in summary
    assert "pending: 0" in summary
