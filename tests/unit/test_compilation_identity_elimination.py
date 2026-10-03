"""Check the identity rule against the runtime gate matrices and the shipped loop.

`pipeline.remove_identity_gates` used to decide a single-qubit instruction from a
hand-written set of opcode names -- `{"i", "id"}` -- plus a per-parameter zero test
that only ever applied to the two-wire rotations. So a `u3` whose polar angle had
come back to zero was never removed however its phases read, and the pass never
consulted a matrix at all. It now asks
`one_qubit_synthesis.is_identity_one_qubit`, which reads the angle triple the
operator schema declares and folds every angle modulo `4*pi`.

Four claims hold that rule up, and this file tests them separately because they
fail differently.

The **classification** claim is not asserted from a list. For each declared
unitary the tests *measure* the runtime gate matrix at five angle points and ask
whether that matrix is the identity, then assert the pass never claims an identity
the matrix denies. The reverse gap -- identities the pass declines -- is compared
with the refusals this round names rather than left as a hope, so a fold that was
widened without a measurement behind it fails here.

The **period** claim is the one that decides the design, and it is measured rather
than derived. `U3(2*pi, 0, 0)` is minus the identity and `U3(4*pi, 0, 0)` is the
identity; `CircuitIR` has no field in which to record a global phase; and
`optimize` is required to leave the statevector identical, which
`tests/unit/test_compilation_inverse_cancellation.py` asserts over random
programs. So the fold is `4*pi`, a half turn stays, and the price is a named set of
refusals rather than a silent sign.

The **band** claim is that the tolerance is the shipped `1e-12` applied to the
angle's residue modulo `4*pi`, and nothing looser: a target's error rate is not
consulted, so an angle merely near a multiple stays.

The **loop** claim is that the verdicts survive `optimize` end to end, which is
where a removal can still be undone by a later pass. The fold limit is pinned too,
because above it the fold is arithmetically meaningless and the pass has to refuse
rather than guess.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.compiler import optimize
from flagquantum.compiler.one_qubit_synthesis import (
    _MODULAR_FOLD_LIMIT,
    is_identity_one_qubit,
)
from flagquantum.compiler.pipeline import _ROTATION_PARAM, remove_identity_gates
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import (
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
)
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: The runtime gate matrices are complex128, so this is rounding rather than a
#: convention, and it is deliberately the same magnitude the pass folds with.
_ATOL = 1.0e-12

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

#: Angle points the classification is measured at. `_GENERIC` is a generic point, at
#: which nothing but a parameter-free identity can be removable. The others drive the
#: polar angle onto a multiple of `pi`, which is the only way a generally
#: non-diagonal opcode approaches the identity, and they are chosen so that the
#: questions the design turns on can be read apart:
#:
#: * the two zero-polar points differ only in whether `phi + lam` cancels, which is a
#:   discrimination no parameter-at-a-time rule can make at all;
#: * `_HALF_TURN` is `theta = 2*pi`, where `U3` is minus the identity -- the identity
#:   up to a global phase and not the identity, which is the whole reason the fold is
#:   `4*pi`;
#: * `_FULL_TURN` is `theta = 4*pi`, where `U3` is the identity exactly, so the fold's
#:   reach is visible and not only its refusals.
_GENERIC = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}
_ZERO_POLAR_SUMMING = {"theta": 0.0, "phi": 0.3, "lbd": -0.3}
_ZERO_POLAR_NOT_SUMMING = {"theta": 0.0, "phi": -0.4211, "lbd": 1.9073}
_HALF_TURN = {"theta": _TWO_PI, "phi": 0.3, "lbd": -0.3}
_FULL_TURN = {"theta": _FOUR_PI, "phi": 0.3, "lbd": -0.3}

_POINTS = {
    "generic": _GENERIC,
    "zero_polar_summing": _ZERO_POLAR_SUMMING,
    "zero_polar_not_summing": _ZERO_POLAR_NOT_SUMMING,
    "half_turn": _HALF_TURN,
    "full_turn": _FULL_TURN,
}

_SINGLE_QUBIT_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)
_ALL_UNITARIES = tuple(
    sorted(name for name, schema in OPERATOR_SCHEMAS.items() if schema.unitary)
)
_MULTI_WIRE_ROTATIONS = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity >= 2 and name in _ROTATION_PARAM
    )
)

#: The rows of the *whole* declared unitary grid the shipped rule removes at each
#: point, measured. `generic` is `i` alone -- nothing else has an angle to be generic
#: in -- and the two zero-polar points differ by exactly one entry, `u3`, which is the
#: discrimination between a rule that sums `phi + lam` and one that cannot. `half_turn`
#: is `i` alone again, which is the sharpest row here: the matrix says `phase(2*pi)`
#: and `u1(2*pi)` are the identity too, and the fold declines both.
_MEASURED_REMOVALS = {
    "generic": ["i"],
    "zero_polar_summing": [
        "cphase",
        "crx",
        "cry",
        "crz",
        "i",
        "phase",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rz",
        "rzz",
        "u1",
        "u3",
    ],
    "zero_polar_not_summing": [
        "cphase",
        "crx",
        "cry",
        "crz",
        "i",
        "phase",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rz",
        "rzz",
        "u1",
    ],
    "half_turn": ["i"],
    "full_turn": ["i", "phase", "rx", "ry", "rz", "u1", "u3"],
}

#: The rows the shipped rule declines although the runtime matrix *does* make them the
#: identity, measured per point. Two mechanisms produce them and both are named in the
#: pass's docstring. At a half turn `phase` and `u1` are the identity while `rz` is
#: minus it, so the one declared triple `(0, 0, theta)` all three share cannot separate
#: them and all three are declined. And the two-wire branch reads the raw angle
#: parameter without folding it, so every two-wire rotation is declined at a full turn.
_EXPECTED_REFUSALS = {
    "generic": [],
    "zero_polar_summing": [],
    "zero_polar_not_summing": [],
    "half_turn": ["cphase", "phase", "u1"],
    "full_turn": ["cphase", "crx", "cry", "crz", "rxx", "ryy", "rzz"],
}

#: The rows where the runtime matrix is a global phase times the identity and *not*
#: the identity -- every one of them minus the identity, measured. The pass declines
#: all of them and the Qiskit anchor removes all of them, so this set is the whole of
#: the disagreement between the two, and it is the sign FlagQuantum IR cannot record.
_MINUS_THE_IDENTITY = {
    "half_turn": ["rx", "rxx", "ry", "ryy", "rz", "rzz", "u3"],
}


def _angles_for(name: str, point: dict[str, float]) -> dict[str, float]:
    schema = get_operator_schema(name)
    assert schema is not None
    return {key: point[key] for key in schema.parameters}


def _wires_for(name: str) -> tuple[int, ...]:
    schema = get_operator_schema(name)
    assert schema is not None
    return tuple(range(schema.arity))


def _matrix(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> torch.Tensor:
    width = 2 ** len(wires)
    return gate_matrix(
        Instruction(name, wires, params=params),
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
    ).reshape(width, width)


def _measured_identity_up_to_phase(
    name: str, wires: tuple[int, ...], params: dict[str, float]
):
    """The matrix as `c * I`, or None when no single `c` fits.

    A unitary is the identity up to a global phase exactly when it is a scalar
    multiple of the identity, and the scalar is then its own `[0, 0]` entry. Reading
    the candidate off that entry rather than solving for it keeps this an independent
    measurement of the gate matrix, which is the whole point: the pass under test
    never reads a matrix at all.
    """

    matrix = _matrix(name, wires, params)
    width = matrix.shape[0]
    candidate = matrix[0, 0]
    if abs(complex(candidate)) < _ATOL:
        return None
    candidate = candidate / abs(candidate)
    residual = float(
        (matrix - candidate * torch.eye(width, dtype=matrix.dtype)).abs().max()
    )
    if residual > _ATOL:
        return None
    return complex(candidate)


def _measured_exact_identity(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> bool:
    """Whether the runtime matrix is the identity matrix itself, not minus it."""

    matrix = _matrix(name, wires, params)
    residual = matrix - torch.eye(matrix.shape[0], dtype=matrix.dtype)
    return float(residual.abs().max()) <= _ATOL


def _pass_says_identity(name: str, wires: tuple[int, ...], **params: float) -> bool:
    instruction = Instruction(name, wires, params=params)
    ir = CircuitIR(max(wires) + 1, (instruction,))
    return len(remove_identity_gates(ir)) == 0


def _gate(name: str, wires: tuple[int, ...], **params: float) -> Instruction:
    return Instruction(name, wires, params=params)


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _statevector(ir: CircuitIR) -> torch.Tensor:
    """The statevector of the program's gates, with the measurements dropped.

    Dropped rather than skipped in the simulator, because what every test below
    compares is the amplitude vector a removal could move -- and a global-phase
    removal moves it by a factor of minus one, which is exactly the change this
    design exists to refuse.
    """

    from flagquantum.simulation.statevector.local import run_local_statevector

    return run_local_statevector(
        CircuitIR(
            ir.n_wires,
            tuple(
                instruction
                for instruction in ir.instructions
                if instruction.name != "measure"
            ),
            dtype="complex128",
        ),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )[0]


def _measured_rows(point: str) -> dict[str, bool]:
    values = _POINTS[point]
    return {
        name: _measured_exact_identity(
            name, _wires_for(name), _angles_for(name, values)
        )
        for name in _ALL_UNITARIES
    }


def _removed_rows(point: str) -> list[str]:
    values = _POINTS[point]
    return sorted(
        name
        for name in _ALL_UNITARIES
        if _pass_says_identity(name, _wires_for(name), **_angles_for(name, values))
    )


@pytest.mark.parametrize("point", sorted(_POINTS))
def test_single_qubit_verdict_never_claims_an_identity_the_matrix_denies(
    point: str,
) -> None:
    """The measured matrix is the truth, and the pass is never looser than it."""

    measured = _measured_rows(point)
    assert set(measured) == set(_ALL_UNITARIES)
    overclaims = sorted(
        name
        for name in _ALL_UNITARIES
        if _pass_says_identity(
            name, _wires_for(name), **_angles_for(name, _POINTS[point])
        )
        and not measured[name]
    )
    assert overclaims == [], point


@pytest.mark.parametrize("point", sorted(_POINTS))
def test_the_removals_are_exactly_the_measured_identities_minus_the_named_refusals(
    point: str,
) -> None:
    """A census comparing nothing with nothing would pass the test above."""

    measured = sorted(name for name, answer in _measured_rows(point).items() if answer)
    removed = _removed_rows(point)
    assert removed == _MEASURED_REMOVALS[point], point
    assert sorted(set(measured) - set(removed)) == _EXPECTED_REFUSALS[point], point
    assert set(removed) <= set(measured), point


@pytest.mark.parametrize("point", sorted(_POINTS))
def test_the_rows_the_matrix_calls_minus_the_identity_are_declined_by_design(
    point: str,
) -> None:
    """The phase-blind column, which is what a lossy optimizer would call removable."""

    values = _POINTS[point]
    minus = sorted(
        name
        for name in _ALL_UNITARIES
        if _measured_identity_up_to_phase(
            name, _wires_for(name), _angles_for(name, values)
        )
        is not None
        and not _measured_exact_identity(
            name, _wires_for(name), _angles_for(name, values)
        )
    )
    assert minus == _MINUS_THE_IDENTITY.get(point, []), point
    for name in minus:
        candidate = _measured_identity_up_to_phase(
            name, _wires_for(name), _angles_for(name, values)
        )
        assert candidate is not None
        assert abs(candidate - (-1 + 0j)) <= _ATOL, (name, candidate)
        assert not _pass_says_identity(
            name, _wires_for(name), **_angles_for(name, values)
        ), name


def test_the_measured_census_is_neither_empty_nor_total() -> None:
    """Over the five points both answers occur, and both occur for real reasons."""

    removed, kept = [], []
    for point in _POINTS:
        for name in _SINGLE_QUBIT_UNITARIES:
            row = f"{point}:{name}"
            if _measured_exact_identity(name, (0,), _angles_for(name, _POINTS[point])):
                removed.append(row)
            else:
                kept.append(row)

    assert len(removed) == 24
    assert len(kept) == 66
    # The opcode the old rule could not classify at all: a `u3` whose polar angle
    # vanishes and whose phases cancel. And `u2`, which is the identity at no angle
    # because its polar angle is the constant `pi/2`.
    assert "zero_polar_summing:u3" in removed
    assert "zero_polar_not_summing:u3" in kept
    assert "zero_polar_summing:u2" in kept
    for point in _POINTS:
        assert f"{point}:u2" in kept, point
        assert f"{point}:i" in removed, point


def test_the_zero_polar_points_differ_in_exactly_one_opcode() -> None:
    """What makes the census a measurement of `phi + lam` and not of `theta`."""

    summing = {
        name
        for name in _SINGLE_QUBIT_UNITARIES
        if _measured_exact_identity(name, (0,), _angles_for(name, _ZERO_POLAR_SUMMING))
    }
    not_summing = {
        name
        for name in _SINGLE_QUBIT_UNITARIES
        if _measured_exact_identity(
            name, (0,), _angles_for(name, _ZERO_POLAR_NOT_SUMMING)
        )
    }
    assert summing - not_summing == {"u3"}
    assert not_summing - summing == set()
    # And the equal-and-opposite sum is not the only shape that removes: `u3` at a
    # zero polar angle with both phases zero is a bare phase, and it is a phase only
    # when the sum vanishes.
    assert _pass_says_identity("u3", (0,), theta=0.0, phi=0.0, lbd=0.0)
    assert not _pass_says_identity("u3", (0,), theta=0.0, phi=0.3, lbd=0.0)


def test_a_half_turn_is_not_a_full_turn() -> None:
    """The decision the round turns on, measured from the matrices rather than derived.

    A half turn of any of these rotations is minus the identity, so removing it would
    multiply the program's statevector by minus one -- a change `CircuitIR` cannot
    record and `optimize` is required not to make. A full turn is the identity, and
    that one is removed.
    """

    for name in ("rx", "ry", "rz", "u3"):
        half = _angles_for(name, _HALF_TURN)
        full = _angles_for(name, _FULL_TURN)
        candidate = _measured_identity_up_to_phase(name, (0,), half)
        assert candidate is not None and abs(candidate - (-1 + 0j)) <= _ATOL, name
        assert not _measured_exact_identity(name, (0,), half), name
        assert _measured_exact_identity(name, (0,), full), name
        assert not _pass_says_identity(name, (0,), **half), name
        assert _pass_says_identity(name, (0,), **full), name
    # `phase` and `u1` are the other way round at a half turn -- they *are* the
    # identity there -- and they are declined anyway, because their declared triple
    # is `rz`'s. That refusal is the price of the shared triple, not an oversight.
    for name in ("phase", "u1"):
        assert _measured_exact_identity(name, (0,), _angles_for(name, _HALF_TURN)), name
        assert not _pass_says_identity(
            name, (0,), **_angles_for(name, _HALF_TURN)
        ), name
        assert _pass_says_identity(name, (0,), **_angles_for(name, _FULL_TURN)), name


def test_a_half_turn_survives_the_loop_and_leaves_the_statevector_alone() -> None:
    """The end-to-end form of the same claim, through all nine shipped passes."""

    program = CircuitIR(2, (_gate("h", (0,)), _gate("rz", (0,), theta=_TWO_PI)))
    optimized = optimize(program)
    assert _names(optimized) == ["h", "rz"]
    assert _statevector(optimized) == pytest.approx(_statevector(program), abs=_ATOL)
    # A full turn is removed by the same loop, and the statevector is still equal.
    full = CircuitIR(2, (_gate("h", (0,)), _gate("rz", (0,), theta=_FOUR_PI)))
    assert _names(optimize(full)) == ["h"]
    assert _statevector(optimize(full)) == pytest.approx(_statevector(full), abs=_ATOL)


def test_every_removal_leaves_the_statevector_identical() -> None:
    """The claim the fold exists to protect: no removal moves the statevector.

    Not "up to a global phase" -- identical. A rule that folded modulo `2*pi` would
    pass every classification test above and fail this one, which is why the two are
    separate tests. The candidate is placed between Hadamards on its own wires so
    that a wrong removal has somewhere to show.
    """

    rows = sorted(
        {
            (name, key, value)
            for point in _POINTS
            for name in _SINGLE_QUBIT_UNITARIES
            if _pass_says_identity(name, (0,), **_angles_for(name, _POINTS[point]))
            for key, value in _angles_for(name, _POINTS[point]).items()
        }
    )
    # Every removing row of the grid, with its parameters. `phase(0)`, `rx(0)` and the
    # rest carry no parameter at all, so they contribute no key and are covered by the
    # explicit programs below.
    assert len(rows) > 0
    checked = 0
    for name in _SINGLE_QUBIT_UNITARIES:
        for point in _POINTS:
            params = _angles_for(name, _POINTS[point])
            if not _pass_says_identity(name, (0,), **params):
                continue
            program = CircuitIR(
                2,
                (
                    _gate("h", (0,)),
                    Instruction(name, (0,), params=params),
                    _gate("h", (0,)),
                    _gate("h", (1,)),
                ),
            )
            optimized = optimize(program)
            assert name not in _names(optimized), (point, name)
            assert _statevector(optimized) == pytest.approx(
                _statevector(program), abs=_ATOL
            ), (point, name)
            checked += 1
    assert checked == sum(
        1
        for names in _MEASURED_REMOVALS.values()
        for name in names
        if name in _SINGLE_QUBIT_UNITARIES
    )
    # The two-wire rotations at a zero angle are the older rule's reach, and the
    # statevector claim holds for them too.
    for name in _MULTI_WIRE_ROTATIONS:
        program = CircuitIR(
            2,
            (
                _gate("h", (0,)),
                _gate("h", (1,)),
                _gate(name, (0, 1), theta=0.0),
                _gate("h", (0,)),
                _gate("h", (1,)),
            ),
        )
        optimized = optimize(program)
        assert name not in _names(optimized), name
        assert _statevector(optimized) == pytest.approx(
            _statevector(program), abs=_ATOL
        ), name


def test_the_rule_reaches_rows_the_opcode_set_could_not() -> None:
    """What the new predicate earns over the literal set and the zero test it replaced.

    The deleted branch was two rules, not one, and both are stated here so that the
    comparison is against what shipped rather than against a caricature of it. The
    single-qubit half removed `{"i", "id"}` and nothing else. The two-wire half
    removed any declared rotation whose parameters were all zero, which over this grid
    is the seven two-wire rotations at the two zero-polar points -- and that half is
    *retained*, so the new rule is a strict widening and not a replacement.

    Seven of the seventeen rows earned are the single-qubit rotations at a zero polar
    angle, which is the whole of the new predicate's reach on this grid; the rest are
    the ones a full turn adds.
    """

    old_reach = {f"{point}:i" for point in _POINTS} | {
        f"{point}:{name}"
        for point in ("zero_polar_summing", "zero_polar_not_summing")
        for name in _MULTI_WIRE_ROTATIONS
    }
    new_reach = {
        f"{point}:{name}" for point in _POINTS for name in _removed_rows(point)
    }
    assert len(old_reach) == 19
    assert len(new_reach) == 36
    assert old_reach <= new_reach
    earned = new_reach - old_reach
    assert len(earned) == 17
    # Every earned row is single-wire: the retained branch already had the others.
    assert all(row.split(":")[1] in _SINGLE_QUBIT_UNITARIES for row in earned)
    assert canonical_opcode("id") == "i"


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("i", {}),
        ("u3", {"theta": 0.0, "phi": 0.0, "lbd": 0.0}),
        ("u3", {"theta": 0.0, "phi": 0.3, "lbd": -0.3}),
        ("u3", {"theta": _FOUR_PI, "phi": 0.3, "lbd": -0.3}),
        ("u3", {"theta": _FOUR_PI, "phi": _TWO_PI, "lbd": -_TWO_PI}),
        ("rx", {"theta": 0.0}),
        ("rx", {"theta": _FOUR_PI}),
        ("rx", {"theta": -_FOUR_PI}),
        ("ry", {"theta": _FOUR_PI}),
        ("rz", {"theta": 0.0}),
        ("rz", {"theta": _FOUR_PI}),
        ("phase", {"theta": _FOUR_PI}),
        ("u1", {"theta": _FOUR_PI}),
    ],
)
def test_declared_removals(name: str, params: dict[str, float]) -> None:
    assert _pass_says_identity(name, (0,), **params)
    assert _measured_exact_identity(name, (0,), params)
    # And the statevector is not merely equal up to a phase, which is the reason the
    # half-turn rows are absent from this list rather than a fourth parameter.
    program = CircuitIR(
        1, (_gate("h", (0,)), Instruction(name, (0,), params=params), _gate("h", (0,)))
    )
    assert _statevector(optimize(program)) == pytest.approx(
        _statevector(program), abs=_ATOL
    )


@pytest.mark.parametrize(
    ("name", "params"),
    [
        # A half turn: minus the identity, not the identity.
        ("rx", {"theta": _TWO_PI}),
        ("ry", {"theta": _TWO_PI}),
        ("rz", {"theta": _TWO_PI}),
        ("u3", {"theta": _TWO_PI, "phi": 0.0, "lbd": 0.0}),
        # The identity, declined because `phase` and `rz` share one declared triple.
        ("phase", {"theta": _TWO_PI}),
        ("u1", {"theta": _TWO_PI}),
        # The identity, declined because `u3`'s phase sum meets the same fold: `2*pi`
        # is a multiple of `2*pi` and not of `4*pi`.
        ("u3", {"theta": 0.0, "phi": 0.3, "lbd": _TWO_PI - 0.3}),
        ("u3", {"theta": 0.0, "phi": -0.3, "lbd": -_TWO_PI + 0.3}),
        ("u3", {"theta": _FOUR_PI, "phi": 0.3, "lbd": _TWO_PI - 0.3}),
        # `u2`'s polar angle is the constant `pi/2`, so it is never the identity.
        ("u2", {"phi": 0.0, "lbd": 0.0}),
        ("u2", {"phi": 0.0, "lbd": _FOUR_PI}),
        # Near a multiple, and a phase that does not cancel.
        ("u3", {"theta": 0.0, "phi": 0.3, "lbd": -0.3 + 1e-4}),
        ("u3", {"theta": 0.4, "phi": 0.0, "lbd": 0.0}),
        ("x", {}),
        ("z", {}),
        ("h", {}),
        ("s", {}),
        ("t", {}),
        ("sx", {}),
    ],
)
def test_declared_refusals(name: str, params: dict[str, float]) -> None:
    assert not _pass_says_identity(name, (0,), **params)


def test_the_tolerance_band_is_the_shipped_one_and_both_of_its_edges_hold() -> None:
    """`1e-12` on the residue modulo `4*pi`, checked from inside and outside the band."""

    inside = (0.0, 1e-13, -1e-13, 1e-12, _FOUR_PI - 1e-13, -_FOUR_PI + 1e-13)
    for residue in inside:
        assert _pass_says_identity("rz", (0,), theta=residue), residue
        assert _pass_says_identity("u3", (0,), theta=0.0, phi=residue, lbd=0.0), residue
    # `2*pi` and `2*pi + 1e-13` are *outside* this band: they are exact multiples of
    # `2*pi` and not of `4*pi`, and that is the whole point of the fold.
    outside = (
        1e-11,
        -1e-11,
        1e-8,
        1e-6,
        1e-4,
        math.pi,
        _TWO_PI,
        _TWO_PI + 1e-13,
        -_TWO_PI,
    )
    for residue in outside:
        assert not _pass_says_identity("rz", (0,), theta=residue), residue
        assert not _pass_says_identity(
            "u3", (0,), theta=0.0, phi=residue, lbd=0.0
        ), residue


def test_a_target_error_rate_is_not_consulted_so_a_near_multiple_stays() -> None:
    """The deliberate narrowness against the Qiskit anchor this round is measured on.

    `RemoveIdentityEquivalent` in Qiskit 2.0.3 removes any gate whose average gate
    fidelity with the identity clears a cutoff, which at its default removes an `rz`
    of `1e-6`. This pass consults no such cutoff and removes neither that nor `1e-4`,
    because an angle merely near a multiple of `4*pi` is a rotation a target may well
    distinguish.
    """

    for residue in (1e-6, 1e-4, 1e-3):
        assert not _measured_exact_identity("rz", (0,), {"theta": residue})
        assert not _pass_says_identity("rz", (0,), theta=residue)


def test_the_fold_limit_refuses_rather_than_inventing_a_removal() -> None:
    """Above the limit the fold would compare rounded integers, so the pass refuses."""

    assert 2048.0 * math.pi == _MODULAR_FOLD_LIMIT
    # `2048*pi` is a multiple of `4*pi`, and exactly on the limit still folds because a
    # float64 there is still finer than the tolerance.
    assert _MODULAR_FOLD_LIMIT % _FOUR_PI == 0.0
    assert _pass_says_identity("rz", (0,), theta=_MODULAR_FOLD_LIMIT)
    assert _pass_says_identity("rz", (0,), theta=-_MODULAR_FOLD_LIMIT)
    # Past it, an angle that is an exact float multiple of `4*pi` is an arbitrary
    # rotation, and folding would call it the identity.
    assert not _pass_says_identity("rz", (0,), theta=4096.0 * math.pi)
    assert not _pass_says_identity("rz", (0,), theta=1e18)
    assert not is_identity_one_qubit(_gate("rz", (0,), theta=4096.0 * math.pi))


def test_a_trainable_angle_is_never_removed() -> None:
    """The refusal that keeps a rotation initialized at zero in the autograd graph."""

    for name in ("rx", "ry", "rz", "phase", "u1"):
        trainable = torch.zeros(1, requires_grad=True)
        assert not _pass_says_identity(name, (0,), theta=trainable), name
    for key in ("theta", "phi", "lbd"):
        angles = {"theta": 0.0, "phi": 0.0, "lbd": 0.0}
        angles[key] = torch.zeros(1, requires_grad=True)
        assert not _pass_says_identity("u3", (0,), **angles), key
    # `rz` is diagonal for every value of its parameter, so the refusal here is about
    # the autograd graph and not about the angle: a full turn is exactly the value the
    # untrainable path removes.
    assert _pass_says_identity("rz", (0,), theta=_FOUR_PI)
    trainable = torch.full((1,), _FOUR_PI, requires_grad=True)
    assert not _pass_says_identity("rz", (0,), theta=trainable)


def test_a_multi_wire_rotation_keeps_the_older_rule() -> None:
    """The fold reaches the single-qubit group only, and that boundary is pinned."""

    assert set(_MULTI_WIRE_ROTATIONS) == {
        "cphase",
        "crx",
        "cry",
        "crz",
        "rxx",
        "ryy",
        "rzz",
    }
    for name in _MULTI_WIRE_ROTATIONS:
        assert _pass_says_identity(name, (0, 1), theta=0.0), name
        assert not _pass_says_identity(name, (0, 1), theta=0.4), name
    # A full turn on a two-wire rotation is measured to be the identity -- all seven of
    # them -- and the pass still keeps all seven: `canonical_euler_angles` describes
    # the single-qubit group only, so there is no declared triple to fold, and the
    # branch decides from the raw parameter. This is a named reach of the round rather
    # than an accident, and the assertion fails if it is widened without a measurement.
    for name in _MULTI_WIRE_ROTATIONS:
        assert _measured_exact_identity(name, (0, 1), {"theta": _FOUR_PI}), name
        assert not _pass_says_identity(name, (0, 1), theta=_FOUR_PI), name
    # ... and a half turn is the identity for only one of them, which is why the
    # two-wire group cannot be folded by the same period as the single-wire group.
    for name in ("crx", "cry", "crz", "rxx", "ryy", "rzz"):
        assert not _measured_exact_identity(name, (0, 1), {"theta": _TWO_PI}), name
    assert _measured_exact_identity("cphase", (0, 1), {"theta": _TWO_PI})


def test_a_gate_carrying_a_caller_supplied_matrix_is_left_alone() -> None:
    """A caller who attached a matrix has said the opcode's name is not the operator.

    The declared triple of `i` is `(0, 0, 0)`, so a rule that read only the triple
    would remove this instruction and change the program. The refusal is the same
    one `diagonal_before_measure._is_diagonal` makes, for the same reason.
    """

    attached = Instruction("i", (0,), matrix=torch.tensor([[0, 1], [1, 0]]))
    assert not is_identity_one_qubit(attached)
    assert _names(remove_identity_gates(CircuitIR(1, (attached,)))) == ["i"]
    # And the same gate without a matrix *is* removed, so what the row above measures
    # is the guard rather than the opcode.
    assert is_identity_one_qubit(Instruction("i", (0,)))
    assert (
        remove_identity_gates(CircuitIR(1, (Instruction("i", (0,)),))).instructions
        == ()
    )


def test_an_operation_outside_the_schema_is_never_removed() -> None:
    """A gate the schema does not describe has no declared triple to read."""

    matrix = torch.eye(2, dtype=torch.complex128)
    instruction = Instruction("custom_identity", (0,), matrix=matrix)
    assert not is_identity_one_qubit(instruction)
    assert _names(remove_identity_gates(CircuitIR(1, (instruction,)))) == [
        "custom_identity"
    ]
    # The four non-unitary channels are not declared unitaries either, whatever their
    # parameters are.
    for name in ("bit_flip", "phase_flip", "depolarizing", "amplitude_damping"):
        schema = get_operator_schema(name)
        assert schema is not None
        assert schema.unitary is False
        zeros = dict.fromkeys(schema.parameters, 0.0)
        assert not is_identity_one_qubit(_gate(name, (0,), **zeros)), name


def test_the_removal_survives_the_shipped_loop() -> None:
    """End to end, including a fold reached only through a merge."""

    def program(*instructions: Instruction) -> list[str]:
        return _names(optimize(CircuitIR(3, instructions)))

    assert program(
        _gate("h", (0,)), _gate("u3", (0,), theta=0.0, phi=0.3, lbd=-0.3)
    ) == ["h"]
    # A fold no single parameter holds: the two rotations merge into `rz(4*pi)`, which
    # is the identity, where they would have merged into `rz(2*pi)` under a `2*pi`
    # fold. So this row is a reach of the new rule and not of the old one.
    assert (
        program(
            _gate("rz", (0,), theta=0.25),
            _gate("rz", (0,), theta=_FOUR_PI - 0.25),
        )
        == []
    )
    assert program(_gate("phase", (0,), theta=_FOUR_PI)) == []
    # The same merge to a half turn stays, and it has to: it is minus the identity.
    assert program(
        _gate("rz", (0,), theta=0.25),
        _gate("rz", (0,), theta=_TWO_PI - 0.25),
    ) == ["rz"]
    # A `u3` at a zero polar angle whose phases do not cancel is still a phase.
    surviving = program(
        _gate("h", (0,)), _gate("u3", (0,), theta=0.0, phi=0.3, lbd=0.0)
    )
    assert "u3" in surviving
    # And a rotation the target can distinguish survives the whole loop.
    assert program(_gate("rz", (0,), theta=1e-4)) == ["rz"]
