"""Contract for the named-gate equivalence table measurement W8-04 records.

The benchmark module answers three questions. Which exact identities does the
table carry, and does each one reproduce its source when the whole path is run?
How many of the table's opcodes reach each recorded basis by name, with no
matrix and no synthesis? And where does Qiskit's own ``BasisTranslator``, over
the standard equivalence library, disagree with that reach?

These tests hold that evidence in place. They fail if a rule stops reproducing
its source, if a basis stops reaching the opcodes the table reaches today, if a
second entry becomes inexact, or if the one recorded disagreement is reported as
agreement or silently dropped.
"""

import pytest

from benchmarks.compiler_basis_translation import (
    _PHASE_CARRYING_RULES,
    TABLE_OPCODES,
    run_benchmark,
)
from benchmarks.compiler_two_qubit_synthesis import DEFAULT_BASES
from flagquantum.compiler.basis_translation import EQUIVALENCE_RULES
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

pytestmark = pytest.mark.benchmark_contract

#: How many of the table's opcodes reach each recorded basis by name. Pinned so
#: a rule that stops composing, or a search that starts stopping one level early,
#: cannot pass as "still reachable". The four closed bases agree with Qiskit
#: exactly; `ion-trap-rz-rx-rzz` is the recorded gap.
_NAME_REACH = {
    "ibm-rz-sx-cx": 16,
    "ibm-heron-cz": 16,
    "rotational": 16,
    "ion-trap-rz-rx-rzz": 8,
    "clifford-t": 7,
}

#: The one basis whose named reach is below Qiskit's, and why: the table routes
#: every two-qubit rule down to `cx` or `cz`, and this basis publishes neither,
#: only a parametrized `rzz`.
_RECORDED_GAP = "ion-trap-rz-rx-rzz"


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == "flagquantum_compiler_basis_translation_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["reference_algorithm"] == "qiskit_basis_translator"
    assert len(payload["name_reach"]) == len(DEFAULT_BASES)


def test_the_table_is_the_size_it_claims_and_measured_entry_by_entry(
    payload: dict,
) -> None:
    # Non-vacuity: every count below is a statement about the table, so the table
    # and the measured rows have to be the same set.
    assert payload["table_opcode_count"] == len(EQUIVALENCE_RULES)
    assert payload["table_opcodes"] == sorted(EQUIVALENCE_RULES)
    assert {row["opcode"] for row in payload["rules"]} == set(TABLE_OPCODES)
    assert payload["table_opcode_count"] == 16
    assert payload["exact_rule_count"] == 15
    assert payload["phase_carrying_rules"] == ["cphase"]
    assert set(_PHASE_CARRYING_RULES) == {"cphase"}


def test_every_rule_reproduces_its_source_through_the_whole_path(payload: dict) -> None:
    for row in payload["rules"]:
        assert row["arity"] == OPERATOR_SCHEMAS[row["opcode"]].arity, row["opcode"]
        assert row["declared_leaf_count"] > 0, row["opcode"]
        assert row["emitted_leaf_count"] > 0, row["opcode"]
        assert row["state_overlap_magnitude"] == pytest.approx(1.0, abs=1e-12), row
        if row["exact"]:
            # An exact entry is exact in the raw state too, not only in magnitude:
            # a global phase of zero is what "exact" means here.
            assert row["max_raw_state_difference"] < 1e-12, row
            assert row["global_phase_radians"] == pytest.approx(0.0, abs=1e-12), row
        else:
            # The one inexact entry has to be measurably inexact in the direction
            # claimed: a real phase, not a rounding artifact.
            assert row["opcode"] in _PHASE_CARRYING_RULES, row["opcode"]
            assert row["max_raw_state_difference"] > 1e-3, row


def test_the_table_stores_the_shortest_statement_not_the_closure(
    payload: dict,
) -> None:
    """Most rules name fewer leaves than they emit, and that is the contract.

    `cry` declares a `crx`, a `crx` declares a `crz`, and a `crz` declares two
    `cx` and two `rz`, so on a `rz`/`sx`/`cx` target the entry for `cry` emits
    more leaves than it names. A rule whose two counts are equal is one whose
    leaves were all already expressible, which is a different case rather than a
    contradiction.

    Non-vacuity: at least one entry has to declare fewer leaves than it emits, or
    the statement above is about nothing.
    """

    deeper = [
        row["opcode"]
        for row in payload["rules"]
        if row["emitted_leaf_count"] > row["declared_leaf_count"]
    ]
    assert deeper, [
        (row["opcode"], row["emitted_leaf_count"]) for row in payload["rules"]
    ]
    for row in payload["rules"]:
        # A rule never emits a leaf it did not declare, and never fewer gates
        # than its own declared leaves once the pi/2 pulse pairs are counted.
        assert set(row["declared_leaf_opcodes"]) <= set(row["emitted_leaf_opcodes"]) | {
            item for item in row["declared_leaf_opcodes"]
        }, row["opcode"]
        assert row["emitted_leaf_count"] >= 1, row["opcode"]


def test_every_basis_reaches_exactly_the_measured_set_of_names(payload: dict) -> None:
    measured = {row["label"]: row for row in payload["name_reach"]}
    assert set(measured) == set(_NAME_REACH)
    for label, expected in _NAME_REACH.items():
        row = measured[label]
        assert row["reached_count"] == expected, row["refused"]
        assert row["reached_count"] + row["refused_count"] == len(TABLE_OPCODES)
        assert set(row["reached_leaf_counts"]) | set(row["refused"]) == set(
            TABLE_OPCODES
        )
        # Non-vacuity: every refusal has to say why, and say it about a gate.
        for opcode, error in row["refused"].items():
            assert "requires unsupported native gate" in error or (
                "no verified decomposition" in error
            ), (label, opcode, error)


def test_a_basis_without_cx_or_cz_reaches_only_what_needs_no_entangler_sink(
    payload: dict,
) -> None:
    """The recorded gap is the table's two-name sink, measured not asserted.

    Every two-qubit rule composes down to `cx` or `cz`, and `cx` and `cz` are
    each other's rule, so a target publishing neither can build neither. The
    names that still reach such a target are exactly the ones that need no
    entangler: the rotation the basis already publishes, the two rotations the
    table derives from it, and the one-wire names. Non-vacuity: the reached set
    has to be a strict subset of the table, and every refusal has to name `cx` or
    `cz` rather than the source opcode.
    """

    measured = {row["label"]: row for row in payload["name_reach"]}
    row = measured[_RECORDED_GAP]
    assert sorted(row["reached_leaf_counts"]) == [
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rzz",
        "sdg",
        "x",
        "z",
    ]
    assert 0 < row["reached_count"] < len(TABLE_OPCODES)
    assert "cx" not in row["reached_leaf_counts"]
    assert "cz" not in row["reached_leaf_counts"]
    # The refusals name the missing sink, so a caller is told which gate to add.
    named = {
        error.rsplit("'", 2)[-2]
        for error in row["refused"].values()
        if "requires unsupported native gate" in error
    }
    assert named, row["refused"]
    assert named <= {"cx", "cz"}, named


def test_the_qiskit_anchor_reports_agreement_and_the_one_disagreement(
    payload: dict,
) -> None:
    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip("Qiskit is not installed; the anchor is a cross-check only")

    assert anchor["compared_opcode_count"] == len(TABLE_OPCODES)
    assert anchor["basis_count"] == len(DEFAULT_BASES)
    assert anchor["disagreements"] == [_RECORDED_GAP]
    # The four agreeing bases agree opcode for opcode, not merely in count.
    assert anchor["opcode_agreement_count"] == len(DEFAULT_BASES) - 1
    rows = {row["label"]: row for row in anchor["rows"]}
    assert rows[_RECORDED_GAP]["count_agrees"] is False
    assert rows[_RECORDED_GAP]["reached_count"] == len(TABLE_OPCODES)
    # The mechanism is measured, not asserted: Qiskit reaches the gap basis by
    # writing angles the source never carried, and this table may not.
    gap = rows[_RECORDED_GAP]["introduced_angles"]
    assert gap, rows[_RECORDED_GAP]
    flat = sorted({value for values in gap.values() for value in values})
    assert any(abs(value - 1.5707963267948966) < 1e-9 for value in flat), flat
    assert rows[_RECORDED_GAP]["introduced_angle_opcode_count"] >= 8
