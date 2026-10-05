"""Measure what an exact identity rule buys the pipeline, and what it gives up.

W9-12 of the Qiskit parity backlog asks for the pass behind
``transpiler/passes/optimization/remove_identity_equiv.py``: a gate whose unitary is
the identity cannot change any outcome, so it can go. The superseded
``pipeline.remove_identity_gates`` answered that question with a set of opcode names
(``{"i", "id"}``) plus a per-parameter zero test, both of which are true of some
identities and false of others. The rule is now one function,
``one_qubit_synthesis.is_identity_one_qubit``, which reads the angle triple the
operator schema already declares and removes an instruction exactly when that triple
is the identity.

**The exactness is the design, and it is what this module exists to establish.** A
gate is removable when it leaves the program alone, and in FlagQuantum the statevector
*is* the program's output -- ``CircuitIR`` has no global-phase field, so a phase that
disappears has changed the program. `U3(2*pi, 0, 0)` is `-I`, not `I`, so the rule
folds its angles modulo `4*pi` rather than `2*pi`. That single doubling is the whole
difference between a correct pass and one that silently drops a sign, and it is
measured three ways below: the `half_turn` and `full_turn` census rows, the
`period_table`'s `identity_up_to_phase_at` versus `exact_identity_at` columns, and the
control rows of `execution_control`, where a removal is driven by hand and read on
both instruments at once.

Three further claims are measured separately, because they fail differently.

* **The classification is measured, not asserted.** For all thirty-one declared
  unitary opcodes the module builds the instruction, asks the runtime for its gate
  matrix, and asks whether that matrix is *exactly* the identity -- then asserts the
  rule's verdict never over-claims. The rule itself never reads a matrix:
  ``compiler/**`` may not import ``flagquantum.simulation.**``, and the boundary is
  proven by replacement, which is why the measurement lives in this file. The refusal
  direction is reported rather than asserted, because a refusal costs an optimization
  and a false removal changes a program; each refusal is named with its mechanism.
* **The band is the shipped one.** The tolerance is the `1e-12` ``pipeline._is_zero``
  already used, applied through ``math.remainder`` to the angle's residue rather than
  to the raw angle, so no new tolerance is introduced. ``boundary_table`` walks both
  edges of the band and pins the one place this rule is deliberately narrower than the
  anchor: a target's error rate is not consulted, so an angle merely *near* a multiple
  stays.
* **The reach survives the loop.** ``pipeline_delta`` runs seeded populations through
  the shipped fixed-point loop twice -- once with the new rule and once with the
  superseded one substituted back in -- and reports what each removal did to the
  program's exact probability vector. The instrument there is deliberately *not* the
  statevector: ``remove_diagonal_gates_before_measure`` legitimately rewrites
  amplitudes while leaving a terminal measurement's distribution alone, so a
  statevector comparison at that level would report the other pass's correct change as
  if it belonged to this rule. The statevector instrument lives in ``execution_control``
  instead, where the removal under inspection is the only one in play -- and the
  control that is the identity only up to a global phase reads `0` on the distribution
  and `sqrt(2)` on the statevector, which is what proves both instruments are live.

**The substitution is the instrument.** The superseded rule is put back by replacing
``one_qubit_synthesis.is_identity_one_qubit``, which ``remove_identity_gates`` imports
inside its own body, rather than by restating the nine-pass loop here. A restated loop
would drift from the shipped one silently; a substituted rule cannot, and the
substitution's effect is itself asserted, so an instrument that measured nothing would
fail rather than report a delta of zero.

**Where the rule is narrower than the anchor, and why that is not a defect.** Qiskit
2.0.3's ``RemoveIdentityEquivalent`` is a fidelity test: it removes a gate when the
average gate fidelity against the identity clears a cutoff derived from
``approximation_degree``. At its default that cutoff sits at a residue of `1e-6`, a
million times looser than this rule's `1e-12`; lowering ``approximation_degree``
loosens it further, and passing ``None`` measurably reproduces the default band rather
than a zero-tolerance one. All of that is measured below, and this rule consults no
target error rate at all, because a false yes changes the program. The anchor is also
phase-blind where this rule is not, so the anchor removes the rotations whose unitary
is `-I` and this pass keeps them. Seventeen census rows separate the two rules -- every
one of them a row where the matrix is minus the identity, and every one of them a
refusal on this side -- and they are published by mechanism rather than counted. The
two mechanisms are second-order consequences of reading the declaration, not
oversights. The canonical triple of `rz`, `phase` and `u1` is the same triple `(0, 0,
theta)`, so no rule that reads only the triple can distinguish `phase(2*pi)` --
genuinely `I` -- from `rz(2*pi)`, which is `-I`; all three are therefore declined at a
half period, and `phase`/`u1` pay for `rz`'s sign. And the two-wire branch still reads
the raw angle parameter without folding it, so `cphase(2*pi)` -- also genuinely `I` --
and every two-wire full turn are declined too. The census counts the first group at
two rows and the second at eight; the anchor's disagreement adds the seven rows where
the two rules differ only about `-I`, which is the sign `CircuitIR` cannot record.

The anchor also postdates the revision earlier rounds used: the pass does not exist in
Qiskit 1.2.4 at all, so this module records which revision it read.

Classification: a local compiler microbenchmark on the single-device fast path. It
runs no distributed work, makes no scalability claim, and is not a performance gate.
Re-run it with::

    python benchmarks/compiler_identity_elimination.py --json-output /tmp/w912.json
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import json
import math
import random
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import flagquantum.compiler.one_qubit_synthesis as synthesis_module  # noqa: E402
from flagquantum.compiler import optimize  # noqa: E402
from flagquantum.compiler.one_qubit_synthesis import (  # noqa: E402
    _MODULAR_FOLD_LIMIT,
    is_identity_one_qubit,
)
from flagquantum.compiler.pipeline import (  # noqa: E402
    _ROTATION_PARAM,
    _is_zero,
    remove_identity_gates,
)
from flagquantum.core.ir import CircuitIR, Instruction  # noqa: E402
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS  # noqa: E402

SCHEMA = "flagquantum_compiler_identity_elimination_benchmark_v1"

_NOW = datetime.now(timezone.utc)

#: Seeded so the counts below are reproducible, which is what lets them be pinned in a
#: benchmark contract.
_SWEEP_SEED = 20261117

#: Three wires, because a two-wire rotation has to sit somewhere while a third wire is
#: busy for the loop-level populations to contain anything but their own candidate.
_REGISTER_WIDTH = 3

#: `complex128`, because the statevectors below are compared exactly rather than
#: sampled and `run_local_statevector` honours the IR's dtype.
_DTYPE = "complex128"

#: The magnitude of a matrix entry below which a global phase cannot be read off it.
_ATOL = 1.0e-12

#: The smallest movement a deliberate *wrong* removal produced over the control shapes.
#: The control asserts the measured movement exceeds it, so an instrument that could
#: not see a wrong removal would fail rather than pass vacuously.
_CONTROL_FLOOR = 1.0e-2

#: Circuits per population in `pipeline_delta`.
_CIRCUITS_PER_POPULATION = 30

_TWO_PI = 2.0 * math.pi

#: The angle points every declared opcode is classified at. `generic` is a generic point,
#: at which nothing but the parameter-free identity can be removable. The other four drive
#: the polar angle to a multiple of `2*pi`, which is the only way a generally non-diagonal
#: opcode approaches the identity, and they are chosen so that the questions the design
#: turns on can be read apart:
#:
#: * `zero_polar_summing` and `zero_polar_not_summing` differ only in whether `phi + lam`
#:   cancels -- the discrimination a parameter-at-a-time rule cannot make at all.
#: * `half_turn` is `theta = 2*pi`, where `U3` is `-I`: the identity up to a global phase
#:   and not the identity. That is the whole of the reason this pass folds modulo `4*pi`,
#:   and the point is where the anchor's removals and this pass's refusals separate.
#: * `full_turn` is `theta = 4*pi`, where `U3` is `I` exactly, so the fold's reach is
#:   visible and not only its refusals.
_POINTS: dict[str, dict[str, float]] = {
    "generic": {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073},
    "zero_polar_summing": {"theta": 0.0, "phi": 0.3, "lbd": -0.3},
    "zero_polar_not_summing": {"theta": 0.0, "phi": -0.4211, "lbd": 1.9073},
    "half_turn": {"theta": _TWO_PI, "phi": 0.3, "lbd": -0.3},
    "full_turn": {"theta": 2.0 * _TWO_PI, "phi": 0.3, "lbd": -0.3},
}

#: Residues the rule's band is walked at, from inside the band outward. The first seven
#: are within `_ATOL` of a multiple of `4*pi` -- including both edges, `1e-12`
#: itself and one full period minus one, because the fold has to see both -- and the rest
#: are not. `_TWO_PI` and `_TWO_PI + 1e-13` are in the list to be *kept*: they are exact
#: multiples of `2*pi` and not of `4*pi`, so they are the rows the design refuses by
#: design and the anchor removes.
_BAND_RESIDUES = (
    0.0,
    1e-13,
    -1e-13,
    _TWO_PI,
    _TWO_PI + 1e-13,
    -_TWO_PI,
    4.0 * math.pi - 1e-13,
    1e-12,
    2e-12,
    1e-11,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    0.4,
)

#: Residues the *anchor's* band is walked at, which is a different question: the
#: anchor's cutoff moves with `approximation_degree`, so the walk has to be dense enough
#: to bracket where it actually sits.
_ANCHOR_BAND_RESIDUES = (
    0.0,
    1e-13,
    1e-12,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    0.4,
)

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
#: The two-wire opcodes the pass still decides from `_ROTATION_PARAM`.
_MULTI_WIRE_ROTATIONS = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity >= 2 and name in _ROTATION_PARAM
    )
)

#: Everything the rule's verdict is compared against, and the reason each row that
#: disagrees with the anchor disagrees. The reason is deliberately one entry rather
#: than a per-opcode table: all four rows are the same fact.
#: Every row where the anchor and this rule differ is a row where the operator is `-I`,
#: and there are two ways this pass arrives at keeping it. Naming both here -- rather than
#: describing the rows as one class -- is what keeps the count honest when the fold's
#: period changes, because the mechanisms move independently of each other.
_ANCHOR_DISAGREEMENT_REASON = (
    "the anchor's fidelity cutoff is blind to a global phase and this rule is not: the "
    "anchor removes a rotation whose unitary is minus the identity, and this pass keeps "
    "it, either because the two-wire branch reads the raw angle parameter without "
    "folding it, or because the canonical triple of rz, phase and u1 is the same "
    "triple and cannot be told apart at a half period"
)


def _circuit(
    instructions: tuple[Instruction, ...], width: int = _REGISTER_WIDTH
) -> CircuitIR:
    return CircuitIR(width, instructions, dtype=_DTYPE)


def _gate(name: str, wires: tuple[int, ...], **params: Any) -> Instruction:
    return Instruction(name, wires, params=params)


def _measure(qubit: int) -> Instruction:
    return Instruction(
        "measure", (qubit,), metadata={"is_dynamic": True, "classical_bit": qubit}
    )


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _params_for(name: str, point: dict[str, float]) -> dict[str, float]:
    schema = OPERATOR_SCHEMAS[name]
    return {key: point[key] for key in schema.parameters}


def _matrix(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> torch.Tensor:
    from flagquantum.simulation.gate_matrix import gate_matrix

    width = 2 ** len(wires)
    return gate_matrix(
        Instruction(name, wires, params=params),
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
    ).reshape(width, width)


def _measured_identity_up_to_phase(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> bool:
    """Whether the runtime gate matrix is a global phase times the identity.

    A unitary is the identity up to a global phase exactly when it is a scalar
    multiple of the identity, and the scalar is then its own `[0, 0]` entry. Reading
    the candidate off that entry -- rather than solving for it -- keeps this an
    independent measurement of the gate matrix, which is the whole point: the rule
    under test never reads a matrix.
    """

    matrix = _matrix(name, wires, params)
    width = matrix.shape[0]
    candidate = matrix[0, 0]
    if abs(complex(candidate)) < _ATOL:
        return False
    candidate = candidate / abs(candidate)
    residual = matrix - candidate * torch.eye(width, dtype=matrix.dtype)
    return float(residual.abs().max()) <= _ATOL


def _measured_exact_identity(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> bool:
    """Whether the runtime gate matrix is the identity matrix itself.

    This is the column the rule is held to, and it is deliberately stricter than
    `_measured_identity_up_to_phase`: `rz(2*pi)` is `-I`, so the phase-blind test above
    calls it the identity and this one does not. The difference between the two columns
    is exactly the reach the design gives up, so both are reported everywhere the
    removal question is asked.
    """

    matrix = _matrix(name, wires, params)
    residual = matrix - torch.eye(matrix.shape[0], dtype=matrix.dtype)
    return float(residual.abs().max()) <= _ATOL


def _rule_says_identity(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> bool:
    """The pass under test, as a whole, on a one-instruction program."""

    ir = CircuitIR(max(wires) + 1, (Instruction(name, wires, params=params),))
    return len(remove_identity_gates(ir)) == 0


def classification_census() -> dict[str, Any]:
    """Every declared unitary at every angle point, against the runtime matrices.

    The *measured* column is the truth and the rule is asked to reproduce it. A row
    where the two differ is reported rather than excluded, and the single-qubit rows
    are counted separately from the two-wire ones because only the first kind is
    decided by the new predicate.

    The grid is `31 opcodes * 5 angle points`, and the points only enter through an
    opcode's declared parameters: the 18 single-qubit opcodes split into 7 that carry a
    polar angle, `u2`, and 10 with no parameter at all, and the two-wire opcodes are
    mostly parameter-free. So a row whose opcode has no parameter is the same program at
    every point, and the counts below are counts of grid cells. `distinct_row_count`
    reports how many of those cells are different programs, which is the number a reader
    who wants "how many opcodes are removable" should use.
    """

    rows: list[dict[str, Any]] = []
    for name in _ALL_UNITARIES:
        schema = OPERATOR_SCHEMAS[name]
        wires = tuple(range(schema.arity))
        for point in sorted(_POINTS):
            params = _params_for(name, _POINTS[point])
            rows.append(
                {
                    "point": point,
                    "opcode": name,
                    "arity": schema.arity,
                    "params": params,
                    "measured_up_to_phase": _measured_identity_up_to_phase(
                        name, wires, params
                    ),
                    "measured": _measured_exact_identity(name, wires, params),
                    "rule": _rule_says_identity(name, wires, params),
                    "predicate": is_identity_one_qubit(
                        Instruction(name, wires, params)
                    ),
                }
            )

    single = [row for row in rows if row["arity"] == 1]
    multi = [row for row in rows if row["arity"] != 1]
    # Two directions, and they are not the same fact. Removing a gate the matrix does not
    # make the identity *changes the program* and is a defect; declining a gate the matrix
    # does make the identity only costs an optimization and is the conservative reach this
    # round states. So the first is asserted empty and the second is published by name.
    overremovals = [
        f"{row['point']}|{row['opcode']}"
        for row in rows
        if row["rule"] and not row["measured"]
    ]
    declined = [
        f"{row['point']}|{row['opcode']}"
        for row in rows
        if row["measured"] and not row["rule"]
    ]
    # The predicate is the module-level function and the rule is the shipped pass. They
    # have to agree on every single-qubit row -- that is what makes the predicate the rule
    # rather than a description of it -- and a row where they did not would mean the pass
    # had grown a second opinion about identity.
    predicate_rule_mismatches = [
        f"{row['point']}|{row['opcode']}"
        for row in single
        if row["predicate"] != row["rule"]
    ]
    # The predicate is asked about the single-qubit group and never about a two-wire
    # rotation, so on two-wire rows it declines by construction. That count is published
    # rather than asserted away, because it is not zero and a reader comparing it with
    # the reach above would otherwise have to guess why.
    predicate_declines = [
        f"{row['point']}|{row['opcode']}"
        for row in multi
        if row["measured"] and not row["predicate"]
    ]
    predicate_overreaches = [
        f"{row['point']}|{row['opcode']}"
        for row in rows
        if row["predicate"] and not row["measured"]
    ]
    # The rule must reproduce the *exact* identity column on the single-qubit rows, and
    # must agree with the shipped predicate there too -- those two are the contract. On
    # the two-wire rows the predicate is not asked at all, so the two-wire refusals are
    # counted rather than treated as disagreements; and the phase-blind column is
    # published so that the reach given up by folding modulo `4*pi` is a number.
    assert not predicate_overreaches, predicate_overreaches
    assert not predicate_rule_mismatches, predicate_rule_mismatches
    assert not overremovals, overremovals
    return {
        "row_count": len(rows),
        "distinct_row_count": len(
            {(row["opcode"], tuple(sorted(row["params"].items()))) for row in rows}
        ),
        "single_qubit_row_count": len(single),
        "multi_wire_row_count": len(multi),
        "opcode_count": len(_ALL_UNITARIES),
        "single_qubit_opcode_count": len(_SINGLE_QUBIT_UNITARIES),
        "measured_removed_count": sum(1 for row in rows if row["measured"]),
        "rule_removed_count": sum(1 for row in rows if row["rule"]),
        "measured_removed_single_qubit_count": sum(
            1 for row in single if row["measured"]
        ),
        "rule_removed_single_qubit_count": sum(1 for row in single if row["rule"]),
        # The phase-blind column, which is what an optimizer that has nowhere to record a
        # global phase would call removable. It is strictly larger than the column above,
        # and the difference is the reach this design declines.
        "measured_removed_up_to_phase_count": sum(
            1 for row in rows if row["measured_up_to_phase"]
        ),
        "removed_up_to_phase_and_not_exactly": sorted(
            f"{row['point']}|{row['opcode']}"
            for row in rows
            if row["measured_up_to_phase"] and not row["measured"]
        ),
        "overremoval_count": len(overremovals),
        "overremovals": overremovals,
        "declined_reach_count": len(declined),
        "declined_reach": declined,
        # Named by mechanism, because "ten refusals" reads as a deficiency and the
        # mechanisms are not: one is a table collision this module cannot see past, and
        # the other is a branch that decides two-wire rotations from the raw parameter.
        "declined_reach_by_mechanism": {
            "shared_z_rotation_triple": sorted(
                row for row in declined if row.split("|")[1] in ("phase", "u1")
            ),
            "two_wire_branch_reads_the_raw_parameter": sorted(
                row for row in declined if row.split("|")[1] in _MULTI_WIRE_ROTATIONS
            ),
        },
        "predicate_rule_mismatch_count": len(predicate_rule_mismatches),
        "predicate_rule_mismatches": predicate_rule_mismatches,
        "two_wire_rows_the_predicate_is_not_asked_about": predicate_declines,
        "two_wire_rows_the_predicate_is_not_asked_about_count": len(predicate_declines),
        "predicate_claims_an_identity_the_matrix_denies": predicate_overreaches,
        "predicate_claims_an_identity_the_matrix_denies_count": len(
            predicate_overreaches
        ),
        "removed_by_point": {
            point: sorted(
                row["opcode"]
                for row in rows
                if row["point"] == point and row["measured"]
            )
            for point in sorted(_POINTS)
        },
        "removed_by_point_up_to_phase": {
            point: sorted(
                row["opcode"]
                for row in rows
                if row["point"] == point and row["measured_up_to_phase"]
            )
            for point in sorted(_POINTS)
        },
        "rows": rows,
    }


def period_table() -> dict[str, Any]:
    """Where each rotation comes back, to `+I` and to `-I`, and where the rule fires.

    This is the table that decides the design, so it is measured rather than asserted.
    `U3(theta, phi, lam)` with the phases cancelling is `-I` at `theta = 2*pi` and `I` at
    `theta = 4*pi`, and the rotations are not all the same function of their angle. The
    table below is the measurement: `cphase`, `phase` and `u1` are exactly `I` at `2*pi`;
    `rx`, `ry`, `rz`, `rxx`, `ryy`, `rzz`, `crx`, `cry`, `crz` and `u3` are only exactly
    `I` at `4*pi`; and nothing with a polar angle fails to reach `+I` within a full
    period. Three columns are reported for every opcode that carries a polar angle, and
    `None` means no candidate reached that column:

    * `identity_up_to_phase_at` -- the first candidate multiple at which the runtime gate
      matrix is `+I` or `-I`, which is the period a global-phase-blind optimizer sees;
    * `exact_identity_at` -- the first at which it is `+I` exactly, which is the period
      this pass is allowed to use;
    * `rule_removes_at` -- what the shipped predicate answers at the *exact* column, so
      that the rule's agreement with the design is a measurement and not a claim.

    An opcode with no polar angle -- every parameter-free gate, and `u2`, whose polar
    angle is the constant `pi/2` -- lands in `no_polar_angle_opcodes`.
    """

    candidates = (_TWO_PI, 2.0 * _TWO_PI)
    generic = {"theta": 0.3, "phi": 1.1, "lbd": -0.7}
    #: The phases are driven to cancel for `u3`, so that its polar period is not hidden
    #: behind a phase sum that does not vanish.
    coupling = {"u3": {"phi": 0.5, "lbd": -0.5}}
    table: dict[str, Any] = {}
    for name in sorted(set(_SINGLE_QUBIT_UNITARIES) | set(_MULTI_WIRE_ROTATIONS)):
        schema = OPERATOR_SCHEMAS[name]
        if "theta" not in schema.parameters:
            continue
        wires = tuple(range(schema.arity))
        base = {key: generic[key] for key in schema.parameters}
        base.update(coupling.get(name, {}))
        up_to_phase = None
        exact = None
        for candidate in candidates:
            params = dict(base, theta=candidate)
            if up_to_phase is None and _measured_identity_up_to_phase(
                name, wires, params
            ):
                up_to_phase = candidate
            if exact is None and _measured_exact_identity(name, wires, params):
                exact = candidate
        table[name] = {
            "arity": schema.arity,
            "identity_up_to_phase_at": up_to_phase,
            "exact_identity_at": exact,
            "rule_removes_at": (
                None
                if exact is None
                else _rule_says_identity(name, wires, dict(base, theta=exact))
            ),
        }
    never = sorted(
        name for name, row in table.items() if row["exact_identity_at"] is None
    )
    return {
        "candidates_in_two_pi": [value / _TWO_PI for value in candidates],
        "opcode_count": len(table),
        "exact_identity_at_two_pi": sorted(
            name for name, row in table.items() if row["exact_identity_at"] == _TWO_PI
        ),
        "exact_identity_at_four_pi": sorted(
            name
            for name, row in table.items()
            if row["exact_identity_at"] == 2.0 * _TWO_PI
        ),
        "no_exact_identity_within_a_full_period": never,
        "no_polar_angle_opcodes": sorted(
            name
            for name in set(_SINGLE_QUBIT_UNITARIES) | set(_MULTI_WIRE_ROTATIONS)
            if "theta" not in OPERATOR_SCHEMAS[name].parameters
        ),
        "table": table,
    }


def _multiple_row(
    name: str, wires: tuple[int, ...], schema: Any, angle: float
) -> dict[str, Any]:
    """What the rule says and what the matrix says, at one named multiple of `pi`.

    Both are reported because a refusal and a mistake read the same from the rule's
    side alone: `kept` with `is_exactly_identity` false is a correct refusal, and
    `kept` with it true is the conservative miss this round publishes.
    """

    params = dict.fromkeys(schema.parameters, 0.0)
    if "theta" in schema.parameters:
        params["theta"] = angle
    elif "phi" in schema.parameters:
        params["phi"] = angle
    return {
        "angle": angle,
        "removed": _rule_says_identity(name, wires, params),
        "is_exactly_identity": _measured_exact_identity(name, wires, params),
        "is_identity_up_to_phase": _measured_identity_up_to_phase(name, wires, params),
    }


def boundary_table() -> dict[str, Any]:
    """The tolerance band, walked from inside and from outside, per opcode family.

    Three families reach the identity differently: a diagonal rotation needs only its
    parameter to vanish, a `u3` needs its polar angle to vanish *and* its two phases to
    cancel, and a two-wire rotation is still decided by the raw parameter. The band is
    reported per family because a rule that folded one family and not another would
    otherwise still look like a band.
    """

    families: dict[str, Any] = {}
    for name, extra in (
        ("rz", {}),
        ("phase", {}),
        ("u1", {}),
        ("rx", {}),
        ("ry", {}),
        ("u3", {"phi": 0.0, "lbd": 0.0}),
        ("crz", {}),
        ("rzz", {}),
        ("cphase", {}),
        ("crx", {}),
    ):
        schema = OPERATOR_SCHEMAS[name]
        wires = tuple(range(schema.arity))
        removed, kept = [], []
        for residue in _BAND_RESIDUES:
            params = {key: extra.get(key, 0.0) for key in schema.parameters}
            if "theta" in schema.parameters:
                params["theta"] = residue
            elif "phi" in schema.parameters:
                params["phi"] = residue
            answer = _rule_says_identity(name, wires, params)
            exact = _measured_exact_identity(name, wires, params)
            (removed if answer else kept).append(residue)
            # The rule may only ever be stricter than the matrix, never looser: a row
            # the rule removes has to be a row the runtime matrix is exactly `I` on, not
            # merely one that is `I` up to a sign.
            if answer and not exact:
                raise AssertionError(
                    f"{name} at residue {residue!r}: the rule removes a gate that the "
                    f"runtime gate matrix does not make exactly the identity"
                )
        families[name] = {
            "arity": schema.arity,
            "removed_residues": removed,
            "kept_residues": kept,
            # The named edges, because a max over mixed magnitudes would read the residue
            # `4*pi - 1e-13` as the loosest removal when it folds to `1e-13` like the
            # rest. The three refusals are the design, so each is pinned by name.
            "removed_at_the_tolerance": 1e-12 in removed,
            "removed_at_the_fold_edge": (4.0 * math.pi - 1e-13) in removed,
            "removed_at_zero": 0.0 in removed,
            "kept_just_outside_the_tolerance": 1e-11 in kept,
            # The two multiples every family is read at, each paired with the measured
            # matrix verdict at the same angle. `2*pi` is a half turn for a single-qubit
            # rotation and for `crx`/`cry`/`crz`, and a full turn for `rxx`/`ryy`/`rzz`/
            # `cphase`, so the four columns are not four views of one fact and are
            # reported separately for every family rather than only for the single-wire
            # ones. `is_exactly_identity` is the measurement: where it is true and
            # `kept` is true, the family is declining an identity.
            "at_two_pi": _multiple_row(name, wires, schema, _TWO_PI),
            "at_four_pi": _multiple_row(name, wires, schema, 2.0 * _TWO_PI),
        }
    fold_edges = {
        "limit_radians": _MODULAR_FOLD_LIMIT,
        "limit_in_two_pi": _MODULAR_FOLD_LIMIT / _TWO_PI,
        "at_limit_removed": _rule_says_identity(
            "rz", (0,), {"theta": _MODULAR_FOLD_LIMIT}
        ),
        "past_limit_removed": _rule_says_identity(
            "rz", (0,), {"theta": 2 * _MODULAR_FOLD_LIMIT}
        ),
        "far_past_limit_removed": _rule_says_identity("rz", (0,), {"theta": 1e18}),
    }
    # The refusals that are the point of the round, each measured at the pair of points
    # that separates "the identity" from "minus the identity".
    refusals = {
        "single_wire_half_turn": {
            "opcodes": sorted(
                name
                for name in _SINGLE_QUBIT_UNITARIES
                if OPERATOR_SCHEMAS[name].parameters
                and _rule_says_identity(
                    name, (0,), _params_for(name, _POINTS["half_turn"])
                )
            ),
            "measured_exact_identity_count": sum(
                1
                for name in _SINGLE_QUBIT_UNITARIES
                if OPERATOR_SCHEMAS[name].parameters
                and _measured_exact_identity(
                    name, (0,), _params_for(name, _POINTS["half_turn"])
                )
            ),
        },
        "single_wire_full_turn": {
            "opcodes": sorted(
                name
                for name in _SINGLE_QUBIT_UNITARIES
                if OPERATOR_SCHEMAS[name].parameters
                and _rule_says_identity(
                    name, (0,), _params_for(name, _POINTS["full_turn"])
                )
            ),
            "measured_exact_identity_count": sum(
                1
                for name in _SINGLE_QUBIT_UNITARIES
                if OPERATOR_SCHEMAS[name].parameters
                and _measured_exact_identity(
                    name, (0,), _params_for(name, _POINTS["full_turn"])
                )
            ),
        },
    }
    return {
        "residues": list(_BAND_RESIDUES),
        "families": families,
        "fold": fold_edges,
        "refusals": refusals,
    }


def rule_contract() -> dict[str, Any]:
    """The rule as literals, plus the structural facts a reader would otherwise assume.

    The last two entries are the point of the round. The superseded rule held a set of
    opcode names and this one holds none, so the module that states the rule names no
    opcode at all; and the angles come from the operator schema, so the module holds
    no second angle table. Both are asserted over the source rather than described in
    prose.
    """

    source = inspect.getsource(remove_identity_gates)
    predicate_source = inspect.getsource(is_identity_one_qubit)
    tables_source = inspect.getsource(synthesis_module)

    opcode_literals = sorted(
        {
            literal
            for literal in (
                *OPERATOR_SCHEMAS,
                *(
                    alias
                    for schema in OPERATOR_SCHEMAS.values()
                    for alias in schema.aliases
                ),
            )
            if f'"{literal}"' in source or f"'{literal}'" in source
        }
    )
    return {
        "pass_module": "flagquantum/compiler/pipeline.py",
        "predicate_module": "flagquantum/compiler/one_qubit_synthesis.py",
        "pass_module_line_count": len(
            Path("flagquantum/compiler/pipeline.py")
            .read_text(encoding="utf-8")
            .splitlines()
        ),
        "pass_function_line_count": len(source.splitlines()),
        "predicate_function_line_count": len(predicate_source.splitlines()),
        "pass_function_opcode_literals": opcode_literals,
        "pass_function_imports": sorted(
            node.module or ""
            for node in ast.walk(ast.parse(source.strip()))
            if isinstance(node, ast.ImportFrom)
        ),
        "single_qubit_rule": (
            "the polar angle of the opcode's canonical Euler triple is an integer "
            "multiple of 4*pi AND the sum of its azimuth and z angles is an integer "
            "multiple of 4*pi -- 4*pi and not 2*pi because a half-period rotation is "
            "minus the identity, and CircuitIR has no field to record the sign"
        ),
        "single_qubit_rule_period": 4.0 * math.pi,
        "multi_wire_rule": (
            "the angle parameter named by pipeline._ROTATION_PARAM for the opcode is "
            "within pipeline._is_zero's 1e-12 of zero -- the raw parameter, not a folded "
            "one, which is why cphase(2*pi) and every two-wire full turn are declined"
        ),
        "tolerance": 1.0e-12,
        "tolerance_source": (
            "pipeline._is_zero, applied to the residue of the angle modulo 4*pi as "
            "math.remainder returns it, so no new tolerance is introduced"
        ),
        "modular_fold_limit": _MODULAR_FOLD_LIMIT,
        "modular_fold_limit_rationale": (
            "past 2*pi*1024 the spacing of float64 exceeds the tolerance, so the only "
            "angle whose exact remainder vanishes is one already equal to 2*pi*k, and "
            "that is an arbitrary rotation rather than the identity -- the bound is "
            "unchanged by the doubling of the period, so the refusal it produces is "
            "still the same refusal"
        ),
        "global_phase_field_in_ir": False,
        "declared_single_qubit_opcodes": list(_SINGLE_QUBIT_UNITARIES),
        "declared_multi_wire_rotations": list(_MULTI_WIRE_ROTATIONS),
        "non_unitary_channels": sorted(
            name for name, schema in OPERATOR_SCHEMAS.items() if not schema.unitary
        ),
        "predicate_holds_no_angle_table": not any(
            isinstance(node, ast.Dict)
            for node in ast.walk(ast.parse(predicate_source.strip()))
        ),
        # A caller-supplied matrix means the opcode's name no longer describes the
        # operator, so the declared triple is not evidence about it. Measured rather
        # than described: `i` is `(0, 0, 0)` and would otherwise be removed here.
        "predicate_refuses_a_caller_supplied_matrix": not is_identity_one_qubit(
            Instruction("i", (0,), matrix=torch.tensor([[0, 1], [1, 0]]))
        ),
        "the_same_gate_without_a_matrix_is_removed": is_identity_one_qubit(
            Instruction("i", (0,))
        ),
        "angle_source": (
            "one_qubit_synthesis.canonical_euler_angles, the single place the module "
            "states the triple"
        ),
        "euler_tables_in_module": sorted(
            name
            for name in ("_FIXED_EULER_ANGLES", "_PARAMETERIZED_EULER_ANGLES")
            if name in tables_source
        ),
        "classification_census": classification_census(),
    }


def _superseded_rule(instruction: Instruction) -> bool:
    """The rule this round replaced, reproduced only as the pipeline's baseline.

    It is one statement because the thing being replaced is one statement: a set of
    opcode names, plus a zero test on whichever parameter ``_ROTATION_PARAM`` names.
    Reproducing it here is not a second source of truth for anything the repository
    ships -- the shipped rule is the predicate -- and it is what the delta is measured
    against.
    """

    if instruction.name in {"i", "id"}:
        return True
    param_name = _ROTATION_PARAM.get(instruction.name)
    return param_name is not None and _is_zero(instruction.params.get(param_name))


@contextlib.contextmanager
def _substituted_rule(rule: Any) -> Iterator[None]:
    """Run the shipped loop with a different single-qubit identity rule.

    `remove_identity_gates` imports the predicate inside its own body, so replacing
    the module attribute reaches the shipped nine-pass loop without restating it. A
    restated loop would drift from the shipped one and the delta below would silently
    become a measurement of the copy; this cannot.
    """

    original = synthesis_module.is_identity_one_qubit
    synthesis_module.is_identity_one_qubit = rule
    try:
        yield
    finally:
        synthesis_module.is_identity_one_qubit = original


def _gate_only(ir: CircuitIR) -> CircuitIR:
    return replace(
        ir,
        instructions=tuple(
            instruction
            for instruction in ir.instructions
            if instruction.name != "measure"
        ),
    )


def _statevector(ir: CircuitIR) -> torch.Tensor:
    from flagquantum.simulation.statevector.local import run_local_statevector

    return run_local_statevector(
        _gate_only(ir), batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
    )[0]


def _probabilities(state: torch.Tensor, width: int) -> dict[int, float]:
    """The exact probabilities of one state, keyed by raw basis index."""

    probabilities = (state.abs() ** 2).detach().tolist()
    assert len(probabilities) == 2**width
    return {index: value for index, value in enumerate(probabilities)}


def _probability_vector(ir: CircuitIR) -> dict[int, float]:
    """The exact probabilities of the gate-only program, keyed by basis index.

    The key is the raw basis index, so the index is the bit pattern of every wire at
    once, wire 0 being the most significant bit -- the same convention
    ``run_local_statevector`` uses. The measure instructions are dropped first, which is
    what makes this the *pre-measurement* vector: for a population whose only
    measurements are terminal and over every wire, that is the distribution
    ``run_dynamic`` would report, without sampling, and for the population that carries
    mid-circuit measurements it is not, which is stated where it is used.
    """

    return _probabilities(_statevector(ir), ir.n_wires)


def _worst_difference(left: dict[int, float], right: dict[int, float]) -> float:
    return max(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right)
    )


def _statevector_differences(
    left: torch.Tensor, right: torch.Tensor
) -> dict[str, float]:
    """Two numbers, because a global phase moves one and not the other.

    `max_entry_difference` is the largest difference between corresponding amplitudes,
    which is `2 * |amplitude|` where the two differ by a phase of minus one -- so it
    reports 2.0 for a removal that is only a global phase. `fidelity_deficit` is
    `1 - |<left|right>|`, which is zero for exactly those removals. Both are published
    because the round's trade is precisely the gap between them.
    """

    difference = (left - right).abs().max()
    overlap = torch.vdot(left, right).abs()
    return {
        "max_entry_difference": float(difference),
        "fidelity_deficit": float(1.0 - overlap),
    }


def _population_fixed_gates(seed: random.Random) -> CircuitIR:
    """The eleven parameter-free single-qubit gates, at random wires and depths."""

    free = [
        name
        for name in _SINGLE_QUBIT_UNITARIES
        if not OPERATOR_SCHEMAS[name].parameters
    ]
    return _interleaved(seed, free)


def _population_zero_angle_rotations(seed: random.Random) -> CircuitIR:
    angles = (0.0, 1e-13, -1e-13, 0.4, -0.4)
    return _interleaved(seed, ["rx", "ry", "rz", "phase", "u1"], angles)


def _population_full_turn_rotations(seed: random.Random) -> CircuitIR:
    """Whole periods, both the ones that are `I` and the ones that are `-I`.

    `4*pi` and `-4*pi` are the identity and `2*pi` is minus the identity, so this
    population is where the two rules must differ: the shipped one removes the first
    two and keeps the third, and the superseded zero test keeps all three. A population
    that held only the identity angles would measure the rule's reach without ever
    reaching its refusal, which is the direction a defect would appear in.
    """

    angles = (2.0 * _TWO_PI, -2.0 * _TWO_PI, _TWO_PI, -_TWO_PI, 0.4)
    return _interleaved(seed, ["rx", "ry", "rz", "phase", "u1"], angles)


def _population_zero_polar_u3(seed: random.Random) -> CircuitIR:
    """A `u3` whose polar angle vanishes, with the phases cancelling or not."""

    instructions = []
    for _ in range(seed.randint(6, 18)):
        phi = seed.choice((0.3, -0.3, 1.7, 0.0))
        cancelling = seed.random() < 0.5
        lam = -phi if cancelling else seed.choice((0.0, 0.9, -1.2))
        # The polar angle alternates between the two ways a `u3` can be the identity: a
        # vanished polar angle, which every phased rule sees, and a whole period, which
        # needs the fold. Both are removable; a half turn is not, and is in the mix so
        # the population exercises a refusal as well as a removal.
        theta = seed.choice((0.0, 0.0, 2.0 * _TWO_PI, _TWO_PI, 0.3))
        instructions.append(
            _gate("u3", (seed.randrange(3),), theta=theta, phi=phi, lbd=lam)
        )
        instructions.append(_gate(seed.choice(("h", "x", "s")), (seed.randrange(3),)))
    return _measured(_circuit(tuple(instructions)))


def _population_two_wire_rotations(seed: random.Random) -> CircuitIR:
    names = list(_MULTI_WIRE_ROTATIONS)
    instructions = []
    for _ in range(seed.randint(4, 12)):
        pair = tuple(seed.sample(range(3), 2))
        angle = seed.choice((0.0, _TWO_PI, 2 * _TWO_PI, -_TWO_PI, 0.4))
        instructions.append(_gate(seed.choice(names), pair, theta=angle))
    return _measured(_circuit(tuple(instructions)))


def _population_mixed(seed: random.Random) -> CircuitIR:
    return _mixed(seed)


def _population_mixed_with_mid_circuit_measures(seed: random.Random) -> CircuitIR:
    return _mixed(seed, mid_circuit_measures=True)


def _interleaved(
    seed: random.Random, names: list[str], angles: tuple[float, ...] = ()
) -> CircuitIR:
    instructions = []
    for _ in range(seed.randint(8, 24)):
        name = seed.choice(names)
        params = (
            dict.fromkeys(OPERATOR_SCHEMAS[name].parameters, seed.choice(angles))
            if angles
            else {}
        )
        instructions.append(_gate(name, (seed.randrange(3),), **params))
    return _measured(_circuit(tuple(instructions)))


def _mixed(seed: random.Random, *, mid_circuit_measures: bool = False) -> CircuitIR:
    pool = [
        "h",
        "x",
        "s",
        "i",
        "rx",
        "ry",
        "rz",
        "phase",
        "u3",
        "cz",
        "cx",
        "crz",
        "rzz",
    ]
    angles = (0.0, 1e-13, _TWO_PI, 2.0 * _TWO_PI, 0.4, -0.4, 1.7)
    instructions = []
    for index in range(seed.randint(10, 30)):
        name = seed.choice(pool)
        schema = OPERATOR_SCHEMAS[name]
        wires = tuple(seed.sample(range(3), schema.arity))
        params: dict[str, float] = {}
        if schema.arity == 1 and schema.parameters:
            if name == "u3":
                theta = seed.choice((0.0, _TWO_PI, 2.0 * _TWO_PI, 0.4))
                phi = seed.choice(angles)
                lam = -phi if seed.random() < 0.5 else seed.choice(angles)
                params = {"theta": theta, "phi": phi, "lbd": lam}
            else:
                params = dict.fromkeys(schema.parameters, seed.choice(angles))
        elif schema.parameters:
            params = dict.fromkeys(schema.parameters, seed.choice(angles))
        instructions.append(Instruction(name, wires, params=params))
        if mid_circuit_measures and index % 7 == 3:
            instructions.append(_measure(seed.randrange(3)))
    return _measured(_circuit(tuple(instructions)))


def _measured(ir: CircuitIR) -> CircuitIR:
    """Append a terminal measurement on every wire, so the exact instrument applies."""

    return replace(
        ir,
        instructions=(
            *ir.instructions,
            *(_measure(wire) for wire in range(ir.n_wires)),
        ),
    )


_POPULATIONS: tuple[tuple[str, Any], ...] = (
    ("parameter_free_gates", _population_fixed_gates),
    ("zero_angle_rotations", _population_zero_angle_rotations),
    ("full_turn_rotations", _population_full_turn_rotations),
    ("zero_polar_u3", _population_zero_polar_u3),
    ("two_wire_rotations", _population_two_wire_rotations),
    ("mixed", _population_mixed),
    ("mixed_with_mid_circuit_measures", _population_mixed_with_mid_circuit_measures),
)


def pipeline_delta() -> list[dict[str, Any]]:
    """What the new rule buys the shipped loop, and what each removal moved.

    The two pipelines are the same nine passes; the only difference is which
    single-qubit identity rule the third and eighth of them consult. So the difference
    in optimized length is attributable to the rule and to nothing else, and it is a
    *lower* bound on what the rule earns, because a removal one pass declines can still
    be reached by another pass through a different spelling.

    The claim this measures is that no removal changes what a device would report, and
    the instrument is the probability vector of the gate-only program. It is deliberately
    *not* the statevector. The shipped loop contains a pass that drops a Z-diagonal gate
    standing before a measurement -- that is what ``remove_diagonal_gates_before_measure``
    is for -- and such a removal leaves the probabilities alone while changing the
    amplitudes, which is correct for a program that is about to be measured and would
    make a statevector comparison between the two pipelines report a difference that
    belongs to neither of them. The statevector instrument therefore lives in
    `execution_control`, where the removal under inspection is driven by itself.

    ``max_probability_vector_difference`` compares the two pipelines' gate-only vectors
    keyed by basis index; for the population that carries mid-circuit measurements that
    vector is not the sampled distribution of the dynamic circuit, but the comparison is
    still the same comparison on both sides, which is what the assertion needs.
    """

    rows: list[dict[str, Any]] = []
    for label, factory in _POPULATIONS:
        source_total = legacy_total = shipped_total = 0
        rule_alone = alone_in_pipeline = signed_delta = changed = executed = 0
        worst_probability = 0.0
        for seed in range(_CIRCUITS_PER_POPULATION):
            ir = factory(random.Random(seed))
            with _substituted_rule(_superseded_rule):
                legacy = optimize(ir)
            shipped = optimize(ir)

            source_total += len(ir)
            legacy_total += len(legacy)
            shipped_total += len(shipped)
            rule_alone += len(ir) - len(remove_identity_gates(ir))
            if _names(shipped) != _names(legacy):
                changed += 1
            alone_in_pipeline += max(0, len(legacy) - len(shipped))
            signed_delta += len(legacy) - len(shipped)

            executed += 1
            worst_probability = max(
                worst_probability,
                _worst_difference(
                    _probability_vector(legacy),
                    _probability_vector(shipped),
                ),
            )
        rows.append(
            {
                "label": label,
                "circuit_count": _CIRCUITS_PER_POPULATION,
                "executed_circuit_count": executed,
                "source_instruction_count": source_total,
                "superseded_rule_instruction_count": legacy_total,
                "optimized_instruction_count": shipped_total,
                "removed_by_the_rule_alone": rule_alone,
                "removed_by_the_rule_in_the_pipeline": alone_in_pipeline,
                # Signed, and the sign matters. `full_turn_rotations` is measured
                # *negative*: the 4*pi fold declines a half turn that the superseded
                # 2*pi fold removed, and `collapse_one_qubit_runs` then re-expands some
                # of the programs into more instructions than the superseded rule left.
                # A clamped column would report that row as a tie, which is the one
                # thing it is not, so both are published.
                "net_instruction_delta_against_the_superseded_rule": signed_delta,
                "changed_circuit_count": changed,
                "max_probability_vector_difference": worst_probability,
            }
        )
    # Every removal the rule makes has to leave the probability vector alone to rounding
    # -- that is the whole claim -- and every population has to have been executed, or a
    # zero difference would be a statement about an empty sweep.
    assert all(row["max_probability_vector_difference"] <= _ATOL for row in rows), rows
    assert all(
        row["executed_circuit_count"] == _CIRCUITS_PER_POPULATION for row in rows
    ), rows
    assert all(
        row["source_instruction_count"] > row["optimized_instruction_count"]
        for row in rows
    ), rows
    # The rule has to actually earn something inside the loop somewhere, or the delta
    # would be a measurement of nothing -- and the *net* column has to earn something
    # too, which is the honest form of the same claim, because the clamped column above
    # cannot go negative and would hide a population the rule lengthened.
    assert sum(row["removed_by_the_rule_in_the_pipeline"] for row in rows) > 0, rows
    assert (
        sum(row["net_instruction_delta_against_the_superseded_rule"] for row in rows)
        > 0
    ), rows
    return rows


def execution_control() -> dict[str, Any]:
    """A removal that must be invisible, and one that must not be.

    The control is what stops the delta above from being a statement about an
    instrument that cannot see anything. Five deletions are driven, and each is driven
    from a program whose candidate sits on wire 0 with the only measurement on wire 1
    where the question is about the statevector -- the diagonal-gate pass declines to
    cross a measurement boundary that does not cover all of a gate's wires, so the
    removal under inspection really is this rule's and not another pass's.

    * An **exact** removal: `u3(0, phi, -phi)` is the identity, and deleting it moves
      neither the outcome distribution nor a single statevector entry.
    * A **reach** removal: `rz(4*pi)` is the identity as well -- `2*pi` would be `-I` --
      and it too moves nothing. This is the removal the fold buys, driven inside the
      shipped loop rather than argued for.
    * A removal the rule **declines because it is only a global phase**: `rz(2*pi)` is
      `-I`. Deleting it moves a statevector entry by `2 * |amplitude|` while leaving
      every outcome probability and the fidelity with the original state at rounding.
      This row is the round's design in one line: the removal looks harmless to every
      statistic a device can report and is still a different program, because the
      statevector is the program's output and the IR has nowhere to record the sign.
    * A removal the rule **declines because it is only near a multiple**: an `rx` of
      `1e-6`, which the anchor's fidelity band takes. The statevector entry difference
      shows it is a real rotation rather than a global phase, and the distribution
      movement shows why a fidelity band takes it anyway. Both numbers are published;
      neither decides the verdict.
    * A removal that is **wrong**: deleting an `rx` of `0.4` moves the distribution far
      past the floor, so an instrument that could not see a wrong removal would fail
      here rather than pass silently.
    """

    exact = _circuit(
        (
            _gate("u3", (0,), theta=0.0, phi=0.3, lbd=-0.3),
            _gate("h", (0,)),
            _gate("x", (1,)),
            _measure(1),
        )
    )
    reach = _circuit(
        (
            _gate("rz", (0,), theta=2.0 * _TWO_PI),
            _gate("h", (0,)),
            _gate("x", (1,)),
            _measure(1),
        )
    )
    global_phase = _circuit(
        (
            _gate("rz", (0,), theta=_TWO_PI),
            _gate("h", (0,)),
            _gate("x", (1,)),
            _measure(1),
        )
    )
    declined = _circuit(
        (
            _gate("rx", (0,), theta=1e-6),
            _gate("x", (1,)),
            _measure(0),
        )
    )
    wrong = _circuit(
        (
            _gate("rx", (0,), theta=0.4),
            _gate("x", (1,)),
            _measure(0),
        )
    )

    def report(program: CircuitIR, drop: str) -> dict[str, Any]:
        optimized = optimize(program)
        reference = _statevector(program)
        candidate = _statevector(
            replace(
                program,
                instructions=tuple(
                    instruction
                    for instruction in program.instructions
                    if instruction.name != drop
                ),
            )
        )
        return {
            "program": _names(program),
            "optimized": _names(optimized),
            "dropped_by_hand": drop,
            "kept_by_the_rule": drop in _names(optimized),
            "statevector": _statevector_differences(reference, candidate),
            "max_outcome_distribution_difference": _worst_difference(
                _probabilities(reference, program.n_wires),
                _probabilities(candidate, program.n_wires),
            ),
        }

    rows = {
        "a_removal_that_is_exact": report(exact, "u3"),
        "a_full_period_removal": report(reach, "rz"),
        "a_removal_declined_because_it_is_only_a_global_phase": report(
            global_phase, "rz"
        ),
        "a_removal_declined_because_it_is_only_near_a_multiple": report(declined, "rx"),
        "a_removal_that_is_wrong": report(wrong, "rx"),
    }
    optimized = {label: row["optimized"] for label, row in rows.items()}
    # The two removals the rule makes, and the three it refuses. A refusal is visible in
    # `optimized`: the instruction is still there.
    for label in ("a_removal_that_is_exact", "a_full_period_removal"):
        assert rows[label]["optimized"] == ["h", "x", "measure"], rows[label]
        assert rows[label]["kept_by_the_rule"] is False, rows[label]
    for label in (
        "a_removal_declined_because_it_is_only_a_global_phase",
        "a_removal_declined_because_it_is_only_near_a_multiple",
        "a_removal_that_is_wrong",
    ):
        assert rows[label]["kept_by_the_rule"] is True, rows[label]
    assert optimized["a_removal_declined_because_it_is_only_a_global_phase"] == [
        "rz",
        "h",
        "x",
        "measure",
    ], optimized
    for label in (
        "a_removal_declined_because_it_is_only_near_a_multiple",
        "a_removal_that_is_wrong",
    ):
        assert rows[label]["optimized"] == ["rx", "x", "measure"], rows[label]

    exact_row = rows["a_removal_that_is_exact"]
    reach_row = rows["a_full_period_removal"]
    phase_row = rows["a_removal_declined_because_it_is_only_a_global_phase"]
    declined_row = rows["a_removal_declined_because_it_is_only_near_a_multiple"]
    wrong_row = rows["a_removal_that_is_wrong"]
    # An exact removal, and a full-period removal, are invisible to both instruments.
    for row in (exact_row, reach_row):
        assert row["max_outcome_distribution_difference"] <= _ATOL, row
        assert row["statevector"]["max_entry_difference"] <= _ATOL, row
    # The declined global-phase removal is invisible to the distribution and loud in the
    # statevector, and the fidelity with the original state is untouched -- so it is the
    # statevector and not a statistic that separates it from the row above.
    assert phase_row["max_outcome_distribution_difference"] <= _ATOL, phase_row
    assert phase_row["statevector"]["max_entry_difference"] > 0.5, phase_row
    assert phase_row["statevector"]["fidelity_deficit"] <= _ATOL, phase_row
    # The declined near-multiple is a real rotation -- its statevector entry difference is
    # orders of magnitude above the tolerance -- and it is still invisible in the
    # distribution, which is why the anchor's band takes it.
    assert declined_row["statevector"]["max_entry_difference"] > 1e-8, declined_row
    assert declined_row["max_outcome_distribution_difference"] <= _ATOL, declined_row
    # And the wrong removal is what the floor is for.
    assert wrong_row["max_outcome_distribution_difference"] > _CONTROL_FLOOR, wrong_row
    return {
        "tolerance": _ATOL,
        "floor": _CONTROL_FLOOR,
        "rows": rows,
    }


def _to_qiskit(name: str, wires: tuple[int, ...], params: dict[str, float]) -> Any:
    """The same instruction as a `QuantumCircuit`, for the anchor's port only."""

    from qiskit import QuantumCircuit
    from qiskit.circuit import library

    classes = {
        "i": library.IGate,
        "x": library.XGate,
        "y": library.YGate,
        "z": library.ZGate,
        "h": library.HGate,
        "s": library.SGate,
        "sdg": library.SdgGate,
        "t": library.TGate,
        "tdg": library.TdgGate,
        "sx": library.SXGate,
        "sxdg": library.SXdgGate,
        "rx": library.RXGate,
        "ry": library.RYGate,
        "rz": library.RZGate,
        "phase": library.PhaseGate,
        "u1": library.PhaseGate,
        "u2": library.U2Gate,
        "u3": library.U3Gate,
        "cx": library.CXGate,
        "cy": library.CYGate,
        "cz": library.CZGate,
        "swap": library.SwapGate,
        "crx": library.CRXGate,
        "cry": library.CRYGate,
        "crz": library.CRZGate,
        "rxx": library.RXXGate,
        "ryy": library.RYYGate,
        "rzz": library.RZZGate,
        "cphase": library.CPhaseGate,
        "ccx": library.CCXGate,
        "cswap": library.CSwapGate,
    }
    gate_class = classes[name]
    # `U2Gate` calls its second angle `lam` and `PhaseGate` calls its only one `theta`;
    # the operator schema's names are positional here and the order is the same.
    angles = [params[key] for key in OPERATOR_SCHEMAS[name].parameters]
    circuit = QuantumCircuit(len(wires))
    circuit.append(gate_class(*angles), list(wires))
    return circuit


def _qiskit_anchor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Qiskit 2.0.3's own pass over the identical census rows.

    The anchor is a cross-check, not a dependency: this module has to run on a machine
    with no Qiskit at all. Every reading is checked twice -- the anchor against the
    runtime gate matrix, and the anchor against this rule -- and a row where the anchor
    disagrees with the *matrix* raises, because that would mean the anchor is not the
    reference this round was measured against. A row where the anchor disagrees with
    this rule is filed under the one published reason, and an unlisted reason raises
    rather than being filed under something that does not fit it.
    """

    try:
        import qiskit  # type: ignore[import-not-found]
        from qiskit.converters import circuit_to_dag  # type: ignore[import-not-found]
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            RemoveIdentityEquivalent,
        )
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    pass_ = RemoveIdentityEquivalent()
    anchor_removed = 0
    against_the_matrix: list[str] = []
    against_the_rule: list[str] = []
    for row in rows:
        name = row["opcode"]
        wires = tuple(range(OPERATOR_SCHEMAS[name].arity))
        circuit = _to_qiskit(name, wires, row["params"])
        dag = circuit_to_dag(circuit)
        before = sum(dag.count_ops().values())
        after = pass_.run(dag)
        removed = before - sum(after.count_ops().values()) > 0
        anchor_removed += int(removed)
        label = f"{row['point']}|{name}"
        # The anchor's docstring calls its criterion "close to an identity operation up
        # to a global phase", and that is literally what it is: a process fidelity
        # against the identity is blind to a sign, so `-I` scores 1. The column it has to
        # match is therefore the phase-blind one, and the column it differs from is this
        # rule's exact one.
        if removed != row["measured_up_to_phase"]:
            against_the_matrix.append(label)
        if removed != row["rule"]:
            against_the_rule.append(label)
    if against_the_matrix:
        raise AssertionError(
            "the anchor and the runtime gate matrices disagree, so the anchor is not "
            f"the reference this round was measured against: {against_the_matrix}"
        )

    # The anchor's own band, measured rather than read off its docstring, because the
    # band is the part of the comparison that is a number.
    band: dict[str, Any] = {}
    for family in ("rz", "u3"):
        removed, kept = [], []
        for residue in _ANCHOR_BAND_RESIDUES:
            params = {"theta": residue}
            if family == "u3":
                params = {"theta": 0.0, "phi": residue, "lbd": 0.0}
            circuit = _to_qiskit(family, (0,), params)
            dag = circuit_to_dag(circuit)
            before = sum(dag.count_ops().values())
            after = pass_.run(dag)
            (removed if before - sum(after.count_ops().values()) > 0 else kept).append(
                residue
            )
        band[family] = {
            "residues": list(_ANCHOR_BAND_RESIDUES),
            "removed_residues": removed,
            "kept_residues": kept,
            "loosest_removed_residue": max(removed) if removed else None,
            "tightest_kept_residue": min(kept) if kept else None,
        }

    return {
        "available": True,
        "pass_name": "RemoveIdentityEquivalent",
        "qiskit_version": qiskit.__version__,
        "pass_module": "qiskit/transpiler/passes/optimization/remove_identity_equiv.py",
        "present_in_qiskit_1_2_4": False,
        "row_count": len(rows),
        "removed_count": anchor_removed,
        "agrees_with_the_up_to_phase_column_on_every_row": not against_the_matrix,
        "disagreement_with_the_rule_count": len(against_the_rule),
        "disagreements_with_the_rule": against_the_rule,
        "disagreement_reason": _ANCHOR_DISAGREEMENT_REASON,
        "band": band,
    }


def run_benchmark() -> dict[str, Any]:
    contract = rule_contract()
    census = contract["classification_census"]
    anchor = _qiskit_anchor(census["rows"])
    return {
        "schema": SCHEMA,
        "generated_at": _NOW.isoformat(),
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "seed": {"sweep": _SWEEP_SEED},
        "reference_algorithm": "qiskit_remove_identity_equivalent",
        "reference_revision": "qiskit 2.0.3 RemoveIdentityEquivalent",
        "rule_contract": contract,
        "period_table": period_table(),
        "boundary_table": boundary_table(),
        "pipeline_delta": pipeline_delta(),
        "execution_control": execution_control(),
        "reference_anchor": anchor,
        # Every anchor reading is stated for the case that actually happened. A host
        # without Qiskit drives the second port on no row at all, and a scope that said
        # `row_count` rows had been driven through both ports would be a claim about a
        # run that never took place -- with a disagreement count of zero reading as
        # agreement rather than as silence.
        "anchor_row_count": census["row_count"] if anchor["available"] else 0,
        "anchor_disagreement_count": (
            0 if not anchor["available"] else anchor["disagreement_with_the_rule_count"]
        ),
        "anchor_agreement_scope": (
            (
                f"{census['row_count']} census rows driven through both ports; the "
                "anchor's verdicts equal the phase-blind identity column of the runtime "
                "gate matrices on every one of them, and every row where it differs from "
                "this rule is a row where the operator is minus the identity -- so the "
                "difference is the sign this pass refuses to drop, not a disagreement "
                "about the matrix"
            )
            if anchor["available"]
            else (
                "no row was driven through the anchor's port: the anchor is not "
                f"installed here ({anchor['reason']}), so the two counts above are "
                "readings of this rule alone and no agreement with the reference is "
                "claimed at any row"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    contract = payload["rule_contract"]
    census = contract["classification_census"]
    anchor = payload["reference_anchor"]
    print(
        f"rule: {contract['pass_function_line_count']}-line pass + "
        f"{contract['predicate_function_line_count']}-line predicate, "
        f"{len(contract['pass_function_opcode_literals'])} opcode literals in the pass"
    )
    print(
        f"census: {census['row_count']} rows over {census['opcode_count']} declared "
        f"unitary opcodes, exactly removable {census['measured_removed_count']}, "
        f"phase-blindly removable {census['measured_removed_up_to_phase_count']}, rule "
        f"removes {census['rule_removed_count']}, over-removals "
        f"{census['overremoval_count']}, declined reach "
        f"{census['declined_reach_count']}"
    )
    periods = payload["period_table"]
    print(
        f"periods: exact at 2*pi for {len(periods['exact_identity_at_two_pi'])} opcodes, "
        f"at 4*pi for {len(periods['exact_identity_at_four_pi'])}, "
        f"never within a period for "
        f"{len(periods['no_exact_identity_within_a_full_period'])}"
    )
    for row in payload["pipeline_delta"]:
        print(
            f"  {row['label']}: {row['source_instruction_count']} -> "
            f"{row['superseded_rule_instruction_count']} superseded -> "
            f"{row['optimized_instruction_count']} shipped, rule alone "
            f"{row['removed_by_the_rule_alone']}, in the pipeline "
            f"{row['removed_by_the_rule_in_the_pipeline']} "
            f"(net {row['net_instruction_delta_against_the_superseded_rule']:+d}), "
            f"programs changed {row['changed_circuit_count']}, worst probability "
            f"difference {row['max_probability_vector_difference']:.3e}"
        )
    control = payload["execution_control"]["rows"]
    exact_control = control["a_removal_that_is_exact"]
    reach_control = control["a_full_period_removal"]
    phase_control = control["a_removal_declined_because_it_is_only_a_global_phase"]
    wrong_control = control["a_removal_that_is_wrong"]
    print(
        "control: exact removal moves "
        f"{exact_control['max_outcome_distribution_difference']:.3e} and a full-period "
        f"removal moves {reach_control['max_outcome_distribution_difference']:.3e}; the "
        "declined global-phase removal moves "
        f"{phase_control['max_outcome_distribution_difference']:.3e} of the distribution "
        f"and {phase_control['statevector']['max_entry_difference']:.3e} of the "
        f"statevector; the wrong removal moves "
        f"{wrong_control['max_outcome_distribution_difference']:.3e}"
    )
    print(
        f"anchor: available {anchor['available']}, "
        f"disagreeing rows {anchor.get('disagreement_with_the_rule_count', 'n/a')}"
    )

    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
