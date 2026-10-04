"""Contract for the declared-basis re-spelling measurement W9-06 records.

The benchmark module answers one question: can `Optimize1qGatesDecomposition` --
re-synthesizing each single-qubit run straight into the target's own basis -- be
ported here? The recorded answer is no, not until the program can carry a global
phase, and the reason is arithmetic rather than a missing routine.

These tests hold that evidence in place. Two of them are the ones that must not be
weakened:

* the determinant test, which is the obstruction itself. A word's determinant is
  the product of its factors' determinants, so a basis whose arity-1 gates carry
  determinants from a finite subgroup cannot spell an operator outside that
  subgroup at any length. A failure here means either the measurement or the
  obstruction is wrong, and both are load-bearing.
* the legalization-phase test, which is the obstruction where it is already
  shipped. `native_gate_legalization` reaches a target basis through
  `one_qubit_synthesis`, which documents its result as equal to its source up to
  one global phase; the test asserts that the whole statevector is multiplied by a
  phase no field of `CircuitIR` records, and that the phase is not merely a sign,
  so it cannot be mistaken for the `-1` a determinant already accounts for.
"""

import math

import pytest

from benchmarks.compiler_one_qubit_decomposition import (
    RUN_TRIAL_COUNT,
    SINGLE_QUBIT_OPCODES,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

#: The determinant argument, in degrees, that each shipped target basis's arity-1
#: gates carry, and whether that argument moves with the gate's parameter. Pinned
#: because the whole obstruction reads off this table: `rz` contributes `0` at
#: every angle, so `{rz, sx, x}` generates only the multiples of 90 degrees, while
#: a z-rotation whose determinant *is* its parameter would generate the circle.
_PINNED_DETERMINANTS = {
    "ibm-rz-sx-cx": {"rz": (0.0, False), "sx": (90.0, False), "x": (180.0, False)},
    "ibm-heron-cz": {"rz": (0.0, False), "sx": (90.0, False), "x": (180.0, False)},
    "rotational": {"rz": (0.0, False), "rx": (0.0, False)},
    "ion-trap-rz-rx-rzz": {"rz": (0.0, False), "rx": (0.0, False)},
    "clifford-t": {"h": (180.0, False), "s": (90.0, False), "t": (45.0, False)},
}

#: The subgroup each basis's words can carry, the declared arity-1 opcodes whose
#: determinant falls outside it at the probed angle, and the opcodes that fall
#: inside it without being declared. An opcode in the middle list is one whose
#: presence in a run puts that run beyond every word over the basis at any length,
#: up to a global phase; an opcode in the third list is one a run may contain and
#: still be exactly spellable, which is why the blocked list is not simply "every
#: opcode the basis does not declare".
_PINNED_REACHABILITY = {
    "ibm-rz-sx-cx": (
        [0.0, 90.0, 180.0, 270.0],
        ["phase", "t", "tdg", "u1", "u2", "u3"],
        ["h", "i", "rx", "ry", "s", "sdg", "sxdg", "y", "z"],
    ),
    "ibm-heron-cz": (
        [0.0, 90.0, 180.0, 270.0],
        ["phase", "t", "tdg", "u1", "u2", "u3"],
        ["h", "i", "rx", "ry", "s", "sdg", "sxdg", "y", "z"],
    ),
    "rotational": (
        [0.0],
        [
            "h",
            "phase",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "u1",
            "u2",
            "u3",
            "x",
            "y",
            "z",
        ],
        ["i", "ry"],
    ),
    "ion-trap-rz-rx-rzz": (
        [0.0],
        [
            "h",
            "phase",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "u1",
            "u2",
            "u3",
            "x",
            "y",
            "z",
        ],
        ["i", "ry"],
    ),
    "clifford-t": (
        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        ["phase", "u1", "u2", "u3"],
        ["i", "rx", "ry", "rz", "sdg", "sx", "sxdg", "tdg", "x", "y", "z"],
    ),
}

#: The measured columns of the re-spelling question, per declared pair. The first
#: two pairs are a shipped target's z-rotation and pulse, where the obstruction
#: applies and the entry-for-entry column is bounded. The second two swap in a
#: z-rotation whose determinant is its parameter, where the obstruction does not
#: apply and the same column instead measures this repository's own five-gate
#: template.
#:
#: The up-to-a-phase column is pinned as an equality: every run has a word, and
#: that is arithmetic. The entry-for-entry column is pinned as a **reference with a
#: margin**, and the distinction is the point of this contract rather than a
#: concession to it. Which runs this repository's template hits exactly is decided
#: by the exact-equality branch tests inside `one_qubit_synthesis._leaves` and
#: `._pulse_leaf` (`polar == _HALF_PI`, `_is_zero(angle)`) together with `_emit`'s
#: `abs(anchor) > 1.0e-15`, all of them reading `cmath`/`math` results a different C
#: library may return one unit in the last place away. The same tree measures
#: `247`/`188` here, `241` on the x86-64 CI runner, and moves by one run under a
#: one-ulp change to every run angle: it is a **platform quantity**. What is not a
#: platform quantity is `beyond_the_subgroup`, asserted below as a hard bound,
#: because the two cases it separates are 45 degrees apart.
_PINNED_REACH = {
    ("rz", "sx"): (True, 247, 188),
    ("rz", "rx"): (True, 230, 193),
    ("phase", "sx"): (False, 267, 489),
    ("u1", "rx"): (False, 425, 529),
}

#: The measured number of runs per pair whose *own* product determinant lies
#: outside the subgroup a word over that pair can carry. No exact word can exist
#: for those runs at any length, so this bounds the entry-for-entry column from
#: above, and unlike that column the bound is not a rounding question.
_PINNED_BOUND = {("rz", "sx"): 2429, ("rz", "rx"): 3770}

#: The allowance on the entry-for-entry reference, in runs. Measured spread across
#: the two platforms available is 6 runs of 4000 and 24 saved gates; a quarter of
#: the reference is several times that and is still far below the fourfold gap to
#: the phase-blind column that separates the two claims.
_PLATFORM_MARGIN = 12

#: The same allowance for the 80-circuit legalization population, where 12 would
#: be a sixth of the population. Three circuits is the same few-percent band.
_PLATFORM_MARGIN_SMALL = 3


def _reference_margin(reference: int) -> int:
    """How far a platform quantity may sit from its reference and be the same."""

    return max(_PLATFORM_MARGIN, reference // 4)


#: The phase each shipped basis multiplies the statevector by when it legalizes a
#: program, over the seeded entangled population. ``phase_is_not_a_sign`` is the
#: count that the determinant cannot explain, which is why the obstruction is a
#: field and not an accounting detail.
_PINNED_LEGALIZATION_PHASE = {
    "ibm-rz-sx-cx": (80, 2, 78, 77),
    "ibm-heron-cz": (80, 2, 78, 77),
    "rotational": (80, 3, 77, 76),
    "ion-trap-rz-rx-rzz": (80, 3, 77, 76),
    # `clifford-t` publishes neither a z-rotation nor a pulse and refuses every
    # circuit whose opcodes it cannot spell, so it contributes no phase datum.
    # It is pinned as a refusal rather than dropped, because a basis that refuses
    # is not a basis that preserved the phase.
    "clifford-t": (0, 0, 0, 0),
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_schema_and_population(payload: dict) -> None:
    """The payload is the declared schema and covers every arity-1 opcode."""

    assert payload["schema"] == (
        "flagquantum_compiler_one_qubit_decomposition_benchmark_v1"
    )
    assert payload["single_qubit_opcodes"] == sorted(SINGLE_QUBIT_OPCODES)
    assert len(SINGLE_QUBIT_OPCODES) == 18
    assert RUN_TRIAL_COUNT == 4000


def test_determinant_table_is_pinned(payload: dict) -> None:
    """Every declared arity-1 gate carries the determinant this analysis assumes."""

    measured = {
        row["label"]: {
            gate["opcode"]: (
                gate["determinant_argument_degrees"][0],
                gate["moves_with_parameter"],
            )
            for gate in row["arity_one_gates"]
        }
        for row in payload["determinants"]
    }
    assert measured == _PINNED_DETERMINANTS
    # The three bases that publish only fixed-determinant gates are exactly the
    # ones the obstruction binds; no basis here has a free determinant, which is
    # why no shipped target can escape it today.
    assert not any(row["determinant_is_free"] for row in payload["determinants"])


def test_reachability_is_the_subgroup_the_basis_generates(payload: dict) -> None:
    """The blocked opcodes are the ones outside the basis's own subgroup.

    This is the obstruction, and it is arithmetic: the sets below are recomputed
    from the pinned determinant table by the same greatest-common-divisor rule the
    module uses, so a failure means the measurement and the claim have parted.
    """

    for row in payload["reachability"]:
        reachable, blocked, reachable_only = _PINNED_REACHABILITY[row["label"]]
        assert row["reachable_arguments_degrees"] == reachable
        assert row["blocked_opcodes"] == blocked
        assert row["determinant_is_free"] is False
        # Nothing in the basis is ever blocked by its own subgroup, and the three
        # groups together are every declared arity-1 opcode exactly once: an
        # opcode is either declared, or carries a determinant the basis can reach,
        # or is the obstruction.
        declared = {gate["opcode"] for gate in row["arity_one_gates"]}
        assert not declared & set(blocked)
        assert not declared & set(reachable_only)
        assert set(blocked) | declared | set(reachable_only) == set(
            SINGLE_QUBIT_OPCODES
        )
        assert len(declared) + len(blocked) + len(reachable_only) == len(
            SINGLE_QUBIT_OPCODES
        )


def test_every_run_has_a_word_but_almost_none_has_an_exact_one(payload: dict) -> None:
    """The two columns, and the gap between them, are the phase the IR lacks.

    A word that agrees only up to a global phase exists for every run, and it is
    three gates shorter on mean; a word that agrees entry for entry exists for
    fewer than one run in ten. The gap is not an implementation shortfall where
    the basis is phase-free -- the reachability test above bounds it -- so a
    regression here would be a claim that the port is cheaper than it is.

    Three assertions have three different strengths on purpose. The phase-blind
    column is an equality, because 4000 of 4000 is arithmetic and every platform
    measures it. The entry-for-entry column is a reference within
    `_reference_margin`, because it counts a floating-point coincidence. And for
    the pairs where the obstruction applies, that column is additionally bounded
    by `_PINNED_BOUND`, which is recomputed here from the run set by an
    independent route -- each run's own product determinant against the pair's
    subgroup -- and which no platform can move. If the two routes ever disagree,
    the measurement is wrong rather than the platform.
    """

    for row in payload["declared_basis_reach"]:
        pair = (row["z_rotation"], row["pulse_opcode"])
        obstruction, exact_words, exact_saved = _PINNED_REACH[pair]
        assert row["determinant_obstruction_applies"] is obstruction
        assert row["trial_count"] == RUN_TRIAL_COUNT
        phase_blind = row["up_to_a_global_phase"]
        assert phase_blind["words_found"] == RUN_TRIAL_COUNT
        assert phase_blind["words_shorter"] > 0
        assert phase_blind["gates_saved"] > 0
        exact = row["entry_for_entry"]
        assert exact["words_found"] > 0
        assert exact["gates_saved"] > 0
        assert abs(exact["words_found"] - exact_words) <= _reference_margin(exact_words)
        assert abs(exact["gates_saved"] - exact_saved) <= _reference_margin(exact_saved)
        # The two columns must stay far apart, and in this direction. An exact
        # word is also a phase-blind one, so this can never invert.
        assert exact["words_found"] < phase_blind["words_found"] // 4
        assert exact["gates_saved"] < phase_blind["gates_saved"] // 4
        bound = row["beyond_the_subgroup"]
        if not obstruction:
            # A free determinant makes the generated subgroup dense, so there is
            # nothing left to bound the column with, and saying so is the honest
            # report: this pair measures the template rather than an obstruction.
            assert bound is None
            continue
        assert bound is not None
        assert bound["run_count"] == _PINNED_BOUND[pair]
        assert exact["words_found"] <= bound["run_count"]
        # The bound has to be a real bound, not a restatement of the column.
        assert bound["run_count"] > exact["words_found"]


def test_legalization_already_drops_a_phase_that_is_not_a_sign(payload: dict) -> None:
    """The obstruction where it is shipped, on the program's own statevector.

    `native_gate_legalization` is the consumer that reaches a target basis, and
    `one_qubit_synthesis` documents its result as equal to its source up to one
    global phase. The statevector is multiplied by that phase, so the phase is
    real and unrecorded -- `CircuitIR` has no field for it -- and it is not the
    `-1` a determinant already accounts for on all but one of the phased programs.
    """

    rows = {row["label"]: row for row in payload["legalization_phase"]}
    for label, (
        circuits,
        unchanged,
        phased,
        not_a_sign,
    ) in _PINNED_LEGALIZATION_PHASE.items():
        row = rows[label]
        assert row["circuit_count"] == circuits
        # Three counts of one 80-circuit population, and the two that are not
        # structural are decided by an equality test on a statevector entry that a
        # different C library may round on the other side. They are held to their
        # references within the margin below rather than exactly; the claim is that
        # almost every program is phased, not that exactly 76 are.
        assert abs(row["statevector_unchanged"] - unchanged) <= _PLATFORM_MARGIN_SMALL
        assert abs(row["statevector_phased"] - phased) <= _PLATFORM_MARGIN_SMALL
        assert abs(row["phase_is_not_a_sign"] - not_a_sign) <= _PLATFORM_MARGIN_SMALL
        assert row["statevector_unchanged"] + row["statevector_phased"] == circuits
        assert row["phase_is_not_a_sign"] >= circuits - 2 * _PLATFORM_MARGIN_SMALL
        # No program is ever non-proportional: the phase is global, so the
        # legalized program computes the same state up to that one scalar.
        assert row["worst_non_proportional_gap"] < 1e-9
        if phased:
            assert 0.0 < row["phase_degrees_min"] <= row["phase_degrees_max"] < 360.0


def test_the_witness_is_two_instructions_and_still_loses_the_phase(
    payload: dict,
) -> None:
    """The minimal reproducer, so the phase cannot be blamed on a fold.

    `h` followed by the basis's own entangler is two instructions. On every basis
    that publishes a z-rotation the statevector is still multiplied by a phase of
    minus 45 or minus 90 degrees, which no sequence of gates could have folded
    away; on `clifford-t`, which publishes `h`, it is unchanged. A basis that has
    the gate keeps the phase, and one that does not loses it -- which is the whole
    point.

    The sign is part of the assertion on purpose. The phase is the one legalization
    *introduced*, so it is the reciprocal of the ratio the other way round, and a
    measurement that reported the wrong direction would look just as plausible: a
    convention with no sign pinned in it flips silently when the implementation
    changes.
    """

    assert len(payload["phase_witness"]) == len(payload["legalization_phase"])
    for row in payload["phase_witness"]:
        assert row["source_opcodes"][0] == "h"
        assert len(row["source_opcodes"]) == 2
        assert row["amplitude_ratios_are_constant"] is True
        degrees = row["global_phase_degrees"]
        assert degrees is not None
        if row["label"] == "clifford-t":
            assert degrees == 0.0
            assert row["max_amplitude_gap"] == 0.0
            assert row["legalized_opcodes"] == row["source_opcodes"]
            continue
        assert math.isclose(degrees % 45.0, 0.0, abs_tol=1e-9)
        assert degrees % 360.0 != 0.0
        assert -180.0 < degrees < 0.0
        # The gap is a real amplitude difference, not a rounding residue: a phase
        # of a quarter turn moves a `1/sqrt(2)` amplitude by more than `0.5`.
        assert row["max_amplitude_gap"] > 0.5


def test_declared_and_measured_agree_on_every_basis(payload: dict) -> None:
    """The three tables cover the same bases, so no basis escapes a measurement."""

    labels = {row["label"] for row in payload["determinants"]}
    assert labels == {row["label"] for row in payload["reachability"]}
    assert labels == {row["label"] for row in payload["legalization_phase"]}
    assert len(labels) == 5
