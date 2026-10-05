"""Conformance of the optimization pass family against one shared contract.

`optimize` runs ten passes to a fixed point, and `compile` runs the same ten a
second time on a routed program. Each pass is a separate implementation of the
same task -- delete work that cannot change an observable -- so adding or
rewriting one is acceptable only while it satisfies the same postconditions. This
module states those postconditions once and drives every pass through all of them,
rather than asserting a different subset per pass.

The contract has five parts, and it holds for every pass:

* the program scaffolding is untouched: the wire count, dtype, IR version,
  measurements, observables and metadata survive;
* no pass grows the program, and no pass emits an operation on a wire the source
  never used;
* no pass mutates its input, and no pass is non-deterministic;
* each wire keeps the order it carried, so a conditional consumer of a mid-circuit
  outcome still reads the same history -- operations on *disjoint* wires may be
  reordered, because they commute and a pass that walks one wire's stack at a time
  does reorder them;
* the observable the pass is allowed to move is preserved.

**That last part is not one assertion, and the split is measured rather than
chosen.** Eight passes act on unitary opcodes only, and for them the observable is
the state vector itself: the assertion is an exact comparison, not an overlap,
because a pass that drops a global phase is exactly the defect this repository has
already paid for once (`one_qubit_synthesis` states that it drops one, and the W9-06
work found the loss reachable from every shipped target basis).
`remove_zero_state_resets` and `remove_diagonal_gates_before_measure` are the two
whose subject matter is a ``reset`` or a ``measure``, and neither opcode is declared
in `operator_schema`, so neither can be executed by the state-vector engine at all.
Their observable is the outcome distribution, and the instrument for it is a
sampled one. `test_the_dynamic_family_is_exactly_the_passes_that_own_an_undeclared_
opcode` derives that split from the operator schema instead of restating it.

**Two of the assertions are floors with a stated source, not counts.** A pass
decides whether to act partly by exact-equality branches over `math` and `cmath`
return values (`one_qubit_synthesis._is_zero`, the half-turn and full-turn
comparisons), so *how many* cases a pass changes is a platform quantity, while the
property being asserted -- that the matrix exercises every pass at all -- is not.
The same reasoning applies to the sampled share tolerance, and each floor below says
where its margin comes from.

The checks are themselves controlled, because a check that has never been seen to
fail is not evidence. Four controls measure the distance between a correct and an
incorrect answer rather than asserting one: a pass that lengthens a program by one
identity, a pass that advances one surviving rotation by a full turn -- a global
phase of pi, which the overlap between the two states cannot see and the vector
difference can -- and the two removals each dynamic pass refuses, taken by hand,
which swap one certain outcome for the other.

The case family is generated rather than enumerated. Half the operations it emits
are *partners* of the one before them -- a repeat, a declared inverse, an entangler
between two wires -- because those are the shapes the passes exist to fold, and a
suite driven over uniform random circuits would report that most of them do nothing.
`test_the_matrix_gives_every_pass_real_work` is what keeps that generator honest.

This is the optimization analogue of `test_routing_conformance.py`: the same
boundary-with-several-implementations problem, the same answer, and the reason
`tests/unit/test_compilation_*.py` cannot give it -- those files state each pass's
own rule deeply and none of them state what the ten share.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.compiler.commutation_cancellation as commutation_cancellation
import flagquantum.compiler.diagonal_before_measure as diagonal_before_measure
import flagquantum.compiler.inverse_cancellation as inverse_cancellation
import flagquantum.compiler.one_qubit_optimization as one_qubit_optimization
import flagquantum.compiler.pipeline as compiler_pipeline
import flagquantum.compiler.two_qubit_optimization as two_qubit_optimization
from flagquantum.compiler import optimize
from flagquantum.compiler.commutation_cancellation import (
    cancel_commuting_self_inverse,
    merge_commuting_rotations,
)
from flagquantum.compiler.diagonal_before_measure import (
    remove_diagonal_gates_before_measure,
)
from flagquantum.compiler.inverse_cancellation import inverse_pairs, merge_inverse_pairs
from flagquantum.compiler.one_qubit_optimization import collapse_one_qubit_runs
from flagquantum.compiler.two_qubit_optimization import collapse_two_qubit_blocks
from flagquantum.compiler.zero_state_reset import remove_zero_state_resets
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import get_operator_schema
from flagquantum.runtime.dynamic import DynamicCircuit, run_dynamic

pytestmark = pytest.mark.unit

# A pass takes a program and returns a program. Every check below is written against
# this signature so a rewritten pass and the shipped one are measured the same way.
_Pass = Callable[[CircuitIR], CircuitIR]

#: The passes whose subject matter is a unitary opcode, and which therefore must
#: leave the state vector itself alone.
_UNITARY_PASSES: tuple[tuple[str, _Pass], ...] = (
    ("remove_identity_gates", compiler_pipeline.remove_identity_gates),
    ("merge_self_inverse", compiler_pipeline.merge_self_inverse),
    ("merge_inverse_pairs", merge_inverse_pairs),
    ("merge_adjacent_rotations", compiler_pipeline.merge_adjacent_rotations),
    ("merge_commuting_rotations", merge_commuting_rotations),
    ("cancel_commuting_self_inverse", cancel_commuting_self_inverse),
    ("collapse_one_qubit_runs", collapse_one_qubit_runs),
    ("collapse_two_qubit_blocks", collapse_two_qubit_blocks),
)

#: The two passes that reason about an opcode `operator_schema` does not declare,
#: and which therefore must leave the outcome distribution alone instead.
_DYNAMIC_PASSES: tuple[tuple[str, _Pass], ...] = (
    ("remove_zero_state_resets", remove_zero_state_resets),
    ("remove_diagonal_gates_before_measure", remove_diagonal_gates_before_measure),
)

#: The passes `_optimize_to_fixed_point` calls, once per fixed-point round, in the
#: order it calls them. `remove_identity_gates` appears twice on purpose: it runs
#: immediately after the inversion passes and again after the commutation passes,
#: for the reason the loop's own comments give.
_ROUND_BODY: tuple[str, ...] = (
    "remove_zero_state_resets",
    "remove_diagonal_gates_before_measure",
    "remove_identity_gates",
    "merge_self_inverse",
    "merge_inverse_pairs",
    "merge_adjacent_rotations",
    "merge_commuting_rotations",
    "cancel_commuting_self_inverse",
    "remove_identity_gates",
    "collapse_one_qubit_runs",
    "collapse_two_qubit_blocks",
)

#: Every pass named once, which is the roster the shared checks are driven over.
#: `test_the_roster_is_the_round_body_the_pipeline_runs` is what ties the two tuples
#: together, so a pass added to the loop and not to this file cannot pass silently.
_ROSTER: tuple[str, ...] = tuple(dict.fromkeys(_ROUND_BODY))

#: A program four wires wide and fourteen operations long, so that a two-wire block
#: has somewhere to sit and a run has something to fold.
_CASE_WIRES = 4
_CASE_GATES = 14
_UNITARY_SEEDS = range(60)
_BOUNDARY_SEEDS = range(60)
_RESET_SEEDS = range(40)
_DIAGONAL_SEEDS = range(25)

#: The raw state-vector difference two programs may show and still be the same
#: mathematical vector. Both sides are float64 computations of the same unitary, so
#: what survives is rounding: over every pass and every family the largest value
#: measured is 5.03e-16, about two float64 epsilons. The margin above that is a
#: thousandfold rather than a hundredfold on purpose, because the number was measured
#: on one libm and CI runs on another whose `sin`/`cos` differ in the last place.
#:
#: What the bound must not absorb is a moved global phase, and two controls measure
#: where that sits instead of asserting it: a full turn appended to a program is a
#: phase of pi and moves this difference by at least 0.638, while a full turn added to
#: an `rz` already inside a program moves it by at least the same 0.638. The floor is
#: not arbitrary -- a global phase of pi multiplies the state by -1, so the difference
#: it moves is `2 * max|amplitude|`, which for a four-wire state is at least
#: `2 / sqrt(16)` = 0.5. The bound sits twelve orders of magnitude below the signal.
_STATE_ATOL = 1.0e-12

#: The floor on how many cases one pass must change for its row to mean anything. A
#: pass driven only over programs it declines is measured on nothing. The smallest
#: count measured here is 17 of 120, so eight is a floor with better than twofold
#: margin. It is a floor rather than the count itself because a pass decides whether to
#: act partly by exact-equality branches over `math`/`cmath` return values, which
#: different C libraries round differently, and because the generator's own stream is
#: free to change: the property being asserted is that the matrix reaches every pass,
#: not that any particular count holds.
_REACH_FLOOR = 8

#: A reset draws a random number, so removing one shifts the generator stream: the
#: same seed no longer produces the same per-shot trajectory even though the
#: distribution is unchanged. Outcome shares are compared for that reason, and the
#: tolerance is the one `tests/unit/test_compilation_zero_state_reset.py` already
#: measured for the same instrument.
_SHOTS = 4000
_SHARE_SEED = 20261110
_SHARE_TOLERANCE = 0.05

#: What a control that really does change the program has to move the state vector by.
#: A full turn is a global phase of pi, so the vectors differ by `2 * max|amplitude|`,
#: at least `2 / sqrt(2**_CASE_WIRES)` = 0.5; the smallest value the control below
#: measured is 0.638. The floor is set at half the arithmetic minimum so the assertion
#: is about the signal being a signal, not about a particular state's amplitudes.
_CONTROL_STATE_GAP = 0.25

#: What a control removal -- one that really does change the program -- has to move a
#: share by. Both controls measured below move a share by 1.000, because each swaps one
#: certain outcome for the other with no sampling between them; the passes' own
#: legitimate removals move a share by at most 0.0197. The floor is set halfway between
#: those two, so the assertion is about the signal being a signal rather than about the
#: exact magnitude of a deliberate change.
_CONTROL_SHARE = 0.5

#: The angles a folded rotation can carry. The generic family draws freely in
#: radians; the boundary family draws from the values where the passes' exact-equality
#: branches over `math`/`cmath` return values make a decision, including the full
#: turns where `u3` is `-I` and a lost global phase would show up as a change of
#: state rather than as rounding.
_GENERIC_ANGLE = 3.0
_BOUNDARY_ANGLES = (
    0.0,
    1.0e-16,
    math.pi / 2,
    math.pi,
    -math.pi,
    2.0 * math.pi,
    -2.0 * math.pi,
    4.0 * math.pi,
)

_ROTATIONS = ("rx", "ry", "rz", "u1", "phase")
_GENERAL_ROTATIONS = (*_ROTATIONS, "u2", "u3")
_SINGLE_WIRE = ("h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "t", "tdg", "i")
_TWO_WIRE = (
    "cx",
    "cz",
    "cy",
    "swap",
    "crx",
    "cry",
    "crz",
    "cphase",
    "rxx",
    "ryy",
    "rzz",
)
_DIAGONAL_ONE_WIRE = ("z", "s", "sdg", "t", "tdg")


def _angle(rng: random.Random, boundary: bool) -> float:
    """Return an angle from the family's own population."""

    if boundary:
        return rng.choice(_BOUNDARY_ANGLES)
    return rng.uniform(-_GENERIC_ANGLE, _GENERIC_ANGLE)


def _parameters(opcode: str, rng: random.Random, boundary: bool) -> dict[str, float]:
    """Return a value for every parameter the schema declares for ``opcode``.

    The declared names are read from the schema rather than written here, so a program
    this file builds is always one the IR accepts. The opcode pools above are still
    written out: what this suite covers is a decision about the repository, and it
    should have to be restated when the vocabulary grows.
    """

    schema = get_operator_schema(opcode)
    if schema is None:
        return {}
    return {name: _angle(rng, boundary) for name in schema.parameters}


def _random_operation(rng: random.Random, n_wires: int, boundary: bool) -> Instruction:
    rotations = _GENERAL_ROTATIONS if boundary else _ROTATIONS
    roll = rng.random()
    if roll < 0.5:
        opcode = rng.choice(rotations)
        return Instruction(
            opcode,
            (rng.randrange(n_wires),),
            params=_parameters(opcode, rng, boundary),
        )
    if roll < 0.85:
        return Instruction(rng.choice(_SINGLE_WIRE), (rng.randrange(n_wires),))
    left, right = rng.sample(range(n_wires), 2)
    opcode = rng.choice(_TWO_WIRE)
    return Instruction(opcode, (left, right), params=_parameters(opcode, rng, boundary))


def _partner(
    previous: Instruction, rng: random.Random, n_wires: int
) -> Instruction | None:
    """Return an operation shaped to be folded with ``previous``, or ``None``.

    These are the shapes the passes exist to reach: a repeat for the self-inverse
    and commuting-rotation passes, a declared inverse for the inversion pass, a
    diagonal gate for the single-qubit run folders, and an entangler that gives a
    two-wire block something to be composed across.
    """

    roll = rng.random()
    if roll < 0.30:
        return Instruction(previous.name, previous.wires, params=dict(previous.params))
    if roll < 0.45:
        partner = inverse_pairs().get(previous.name)
        if partner is not None:
            return Instruction(partner, previous.wires, params=dict(previous.params))
    if roll < 0.70:
        left, right = rng.sample(range(n_wires), 2)
        return Instruction("cx", (left, right))
    if roll < 0.90 and len(previous.wires) == 1:
        return Instruction(rng.choice(_DIAGONAL_ONE_WIRE), previous.wires)
    return None


def _unitary_case(
    seed: int, n_wires: int = _CASE_WIRES, *, boundary: bool = False
) -> CircuitIR:
    """Return a deterministic unitary program with foldable structure in it."""

    rng = random.Random(seed)
    instructions: list[Instruction] = []
    while len(instructions) < _CASE_GATES:
        if instructions and rng.random() < 0.45:
            partner = _partner(instructions[-1], rng, n_wires)
            if partner is not None:
                instructions.append(partner)
                continue
        instructions.append(_random_operation(rng, n_wires, boundary))
    return CircuitIR(n_wires, tuple(instructions), dtype="complex128")


def _measure(wire: int, bit: int) -> Instruction:
    return Instruction(
        "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": bit}
    )


def _reset(wire: int) -> Instruction:
    return Instruction("reset", (wire,), metadata={"is_dynamic": True})


def _measurement_case(seed: int, n_wires: int = 3) -> CircuitIR:
    """Return a program ending in a diagonal run whose only readers are measurements."""

    rng = random.Random(seed)
    body = [_random_operation(rng, n_wires, False) for _ in range(10)]
    for wire in range(n_wires):
        for _ in range(rng.randrange(1, 3)):
            body.append(Instruction(rng.choice(_DIAGONAL_ONE_WIRE), (wire,)))
    body.extend(_measure(wire, wire) for wire in range(n_wires))
    return CircuitIR(n_wires, tuple(body), dtype="complex128")


def _reset_case(seed: int, n_wires: int = 4) -> CircuitIR:
    """Return a program with resets on wires nothing has touched.

    The last wire is deliberately left idle by the body, so the reset before the
    measurements is a mid-program candidate rather than only a leading one.
    """

    rng = random.Random(seed)
    body = [_random_operation(rng, n_wires - 1, False) for _ in range(9)]
    leading = [_reset(wire) for wire in range(n_wires) if rng.random() < 0.6]
    tail = [_reset(n_wires - 1)]
    measurements = [_measure(wire, wire) for wire in range(n_wires)]
    return CircuitIR(
        n_wires, tuple(leading + body + tail + measurements), dtype="complex128"
    )


def _unitary_cases() -> list[CircuitIR]:
    """The two unitary families the shared checks run over.

    Both are needed and neither is enough. The generic family is where folding
    happens; the boundary family is where the passes' exact-equality angle branches
    decide, and it is the population where a dropped global phase would be visible as
    a change of state rather than absorbed as rounding.
    """

    return [_unitary_case(seed) for seed in _UNITARY_SEEDS] + [
        _unitary_case(seed, boundary=True) for seed in _BOUNDARY_SEEDS
    ]


def _measurement_cases() -> list[CircuitIR]:
    return [_measurement_case(seed) for seed in _DIAGONAL_SEEDS]


def _reset_cases() -> list[CircuitIR]:
    return [_reset_case(seed) for seed in _RESET_SEEDS]


#: The passes every shared check runs over, paired with the case family that gives
#: each one real work. The two observer classes differ in what they must preserve,
#: not in what they may touch, so scaffolding, growth, mutability and order are
#: checked on all ten.
_ALL_PASSES: tuple[tuple[str, _Pass], ...] = _UNITARY_PASSES + _DYNAMIC_PASSES

_CASE_FAMILY: dict[str, Callable[[], list[CircuitIR]]] = {
    **{name: _unitary_cases for name, _ in _UNITARY_PASSES},
    "remove_zero_state_resets": _reset_cases,
    "remove_diagonal_gates_before_measure": _measurement_cases,
}


def _state(program: CircuitIR) -> torch.Tensor:
    return fq.Circuit.from_ir(program).state().reshape(-1)


def _shares(program: CircuitIR) -> dict[tuple[int, ...], float]:
    result = run_dynamic(
        DynamicCircuit.from_ir(program), shots=_SHOTS, seed=_SHARE_SEED
    )
    counts: dict[tuple[int, ...], int] = {}
    for row in result.classical_bits.tolist():
        outcome = tuple(row)
        counts[outcome] = counts.get(outcome, 0) + 1
    return {outcome: count / _SHOTS for outcome, count in counts.items()}


def _share_gap(
    left: dict[tuple[int, ...], float], right: dict[tuple[int, ...], float]
) -> float:
    return max(
        abs(left.get(outcome, 0.0) - right.get(outcome, 0.0))
        for outcome in set(left) | set(right)
    )


def _changed(pass_fn: _Pass, program: CircuitIR) -> bool:
    return pass_fn(program).instructions != program.instructions


# ---------------------------------------------------------------------------
# The four checks every pass shares.
# ---------------------------------------------------------------------------


def _scaffolding_failures(
    pass_name: str, pass_fn: _Pass, cases: list[CircuitIR]
) -> list[str]:
    """Report a program field or wire the pass had no business changing.

    A caller reads `measurements`, `observables`, `dtype` and `metadata` off the
    optimized program and expects them to describe the program it submitted. A wire
    outside the ones the source uses would be an operation the executor cannot
    address.
    """

    failures: list[str] = []
    for program in cases:
        out = pass_fn(program)
        for field in (
            "n_wires",
            "dtype",
            "version",
            "measurements",
            "observables",
            "metadata",
        ):
            if getattr(out, field) != getattr(program, field):
                failures.append(f"{pass_name}: {field} changed")
        used = {wire for instruction in program for wire in instruction.wires}
        for instruction in out:
            if any(wire not in used for wire in instruction.wires):
                failures.append(f"{pass_name}: emitted {instruction} on an unused wire")
                break
    return failures


def _growth_failures(
    pass_name: str, pass_fn: _Pass, cases: list[CircuitIR]
) -> list[str]:
    """Report a pass that made a program longer.

    This is the property that decides which passes belong in the fixed-point loop: a
    pass that can lengthen a program cannot sit in a loop that stops when the length
    stops falling. It is the reason a pass that trades one operation for several is
    reachable from a legalizer that has a target in mind and absent from `_ROUND_BODY`,
    and `test_the_growth_check_rejects_a_pass_that_lengthens_a_program` measures that
    this check catches such a pass rather than trusting the claim.
    """

    failures: list[str] = []
    for program in cases:
        out = pass_fn(program)
        if len(out) > len(program):
            failures.append(f"{pass_name}: {len(program)} operations became {len(out)}")
    return failures


def _mutability_failures(
    pass_name: str, pass_fn: _Pass, cases: list[CircuitIR]
) -> list[str]:
    """Report a pass that changed its input or answered differently the second time."""

    failures: list[str] = []
    for program in cases:
        before = program.to_json()
        first = pass_fn(program)
        if program.to_json() != before:
            failures.append(f"{pass_name}: the source program was modified")
        if pass_fn(program).instructions != first.instructions:
            failures.append(f"{pass_name}: two runs on one input disagreed")
    return failures


def _instruction_key(instruction: Instruction) -> Any:
    """Return what makes two instructions the same operation to a reader.

    The matrix is folded in by shape rather than by value: a caller-supplied matrix
    is a list of lists and not hashable, and two instructions carrying matrices of
    different sizes are different operations whatever the entries are.
    """

    return (
        instruction.name,
        instruction.wires,
        tuple(sorted(instruction.params.items())),
        None if instruction.matrix is None else len(instruction.matrix),
    )


def _subsequence_holds(source: list[Any], wanted: list[Any]) -> bool:
    """Whether ``wanted`` is a subsequence of ``source``, order included."""

    remaining = iter(source)
    return all(any(item == key for item in remaining) for key in wanted)


def _wire_survivors(
    program: CircuitIR, out: CircuitIR, wire: int
) -> tuple[list[Any], list[Any]]:
    """Return the source and output survivor sequences that share ``wire``.

    An operation the pass rewrote is not a survivor: its spelling changed, so it
    cannot be matched to a source operation by name and wires. What remains is the
    part of the program the pass left exactly as it found it, and that part has to
    hold the order the wire carried.
    """

    key = _instruction_key
    source_counts: dict[Any, int] = {}
    for instruction in program:
        source_counts[key(instruction)] = source_counts.get(key(instruction), 0) + 1
    emitted_counts: dict[Any, int] = {}
    for instruction in out:
        emitted_counts[key(instruction)] = emitted_counts.get(key(instruction), 0) + 1
    source = [key(i) for i in program if wire in i.wires]
    emitted = [key(i) for i in out if wire in i.wires]
    survivors = [k for k in emitted if emitted_counts[k] <= source_counts.get(k, 0)]
    return source, survivors


def _order_failures(
    pass_name: str, pass_fn: _Pass, cases: list[CircuitIR]
) -> list[str]:
    """Report a survivor that moved ahead of a survivor it used to follow on a wire.

    The order is per wire, and that scope is the whole content of the assertion.
    Operations on disjoint wires may be reordered: they commute, and a pass that folds
    one wire's stack at a time does reorder them. Stating the check over the whole
    program instead would reject `collapse_one_qubit_runs` for moving a rotation past an
    entangler that shares no wire with it -- the pass is licensed to do that, so a check
    that forbids it is wrong rather than strict. What may not move is the order two
    survivors share a wire in: a conditional consumer of a mid-circuit outcome reads the
    sequence its wire carries, and the passes that reason about commutation are the ones
    that could break this without noticing.
    """

    failures: list[str] = []
    for program in cases:
        out = pass_fn(program)
        for wire in range(program.n_wires):
            source, survivors = _wire_survivors(program, out, wire)
            if not _subsequence_holds(source, survivors):
                failures.append(
                    f"{pass_name}: wire {wire} reordered a surviving operation"
                )
    return failures


def _state_failures(pass_name: str, pass_fn: _Pass) -> list[str]:
    """Report a pass whose output is not the same state vector as its input.

    Compared as vectors rather than as `|<in|out>|`, because an overlap is blind to
    exactly the failure this file is here to catch: a fold that drops a global phase,
    which leaves the overlap at rounding and the vectors two apart.
    """

    failures: list[str] = []
    for program in _unitary_cases():
        out = pass_fn(program)
        difference = float(torch.max(torch.abs(_state(program) - _state(out))))
        if difference > _STATE_ATOL:
            failures.append(f"{pass_name}: state vector moved by {difference:.3e}")
    return failures


# ---------------------------------------------------------------------------
# The coverage of the case matrix, and the two instrument controls.
# ---------------------------------------------------------------------------


def test_the_matrix_gives_every_pass_real_work() -> None:
    """Guard the reach the shared checks depend on.

    A pass driven only over programs it declines is measured on nothing, and would
    satisfy every postcondition by doing nothing at all. Each pass below has to
    reach its own removal on at least `_REACH_FLOOR` of the cases in its family, and
    the family has to be non-empty, since a pass with no cases at all would otherwise
    satisfy the floor by making the loop body never run.
    """

    starved: list[str] = []
    counts: dict[str, str] = {}
    for name, pass_fn in _ALL_PASSES:
        cases = _CASE_FAMILY[name]()
        changed = sum(1 for program in cases if _changed(pass_fn, program))
        counts[name] = f"{changed}/{len(cases)}"
        if not cases or changed < _REACH_FLOOR:
            starved.append(f"{name} changed {changed} of {len(cases)} cases")
    assert not starved, "\n".join(starved + [f"counts: {counts}"])


def test_the_state_vector_instrument_can_see_a_dropped_global_phase() -> None:
    """Measure that the assertion above is tuned to the defect it names.

    A trailing `rz(2*pi)` is `-I`: a declared opcode, executable, and a global phase
    of pi and nothing else. The claim is per case, not about the worst case: on *every*
    program the phase has to move the raw difference above `_STATE_ATOL` while leaving
    `1 - |<in|out>|` at rounding. That is what makes the choice of a vector comparison
    over an overlap a load-bearing decision rather than a stylistic one, and it is why
    the bound is stated as a vector difference rather than as an overlap.

    How far the phase moves the vectors is not a constant: it is
    `2 * max|amplitude|`, at least `2 / sqrt(2**n_wires)`, so the smallest value here
    is 0.638 and the largest is 2.000. A check written as "the worst case exceeds
    one" would have passed on this family without the property holding anywhere in
    particular.
    """

    differences: list[float] = []
    worst_overlap_gap = 0.0
    for program in _unitary_cases():
        phased = replace(
            program,
            instructions=program.instructions
            + (Instruction("rz", (0,), params={"theta": 2.0 * math.pi}),),
        )
        left, right = _state(program), _state(phased)
        differences.append(float(torch.max(torch.abs(left - right))))
        overlap = torch.vdot(left, right)
        worst_overlap_gap = max(worst_overlap_gap, float(abs(abs(overlap) - 1.0)))

    assert len(differences) == len(_unitary_cases())
    assert min(differences) > _STATE_ATOL, min(differences)
    # The floor the appended phase cannot go below, so the measurement above is not
    # read as an accident of these particular states.
    assert min(differences) >= 2.0 / math.sqrt(2**_CASE_WIRES) - _STATE_ATOL
    assert worst_overlap_gap < _STATE_ATOL


def test_the_dynamic_family_is_exactly_the_passes_that_own_an_undeclared_opcode() -> (
    None
):
    """Derive the observer split from the operator schema rather than restating it.

    The two classes are not a convenience: `operator_schema` declares neither
    `reset` nor `measure`, so a program carrying one cannot be given to the
    state-vector engine, and a pass whose whole subject is such an opcode can only be
    held to the outcome distribution. When a future pass reasons about an undeclared
    opcode it has to appear in `_DYNAMIC_PASSES`, and this is the test that says so.
    """

    undeclared = ("reset", "measure")
    assert [get_operator_schema(opcode) for opcode in undeclared] == [None, None]

    for name, pass_fn in _DYNAMIC_PASSES:
        assert pass_fn in {
            remove_zero_state_resets,
            remove_diagonal_gates_before_measure,
        }, name
    for name, pass_fn in _UNITARY_PASSES:
        assert pass_fn not in {
            remove_zero_state_resets,
            remove_diagonal_gates_before_measure,
        }, name

    # Every opcode the unitary cases use is declared, which is why those cases can be
    # executed and the other two cannot.
    for program in _unitary_cases():
        for instruction in program:
            assert get_operator_schema(instruction.name) is not None, instruction.name


# ---------------------------------------------------------------------------
# The shared contract, driven over every pass.
# ---------------------------------------------------------------------------


def test_no_pass_changes_the_program_scaffolding() -> None:
    failures = [
        message
        for name, pass_fn in _ALL_PASSES
        for message in _scaffolding_failures(name, pass_fn, _CASE_FAMILY[name]())
    ]
    assert not failures, "\n".join(failures)


def test_no_pass_grows_a_program() -> None:
    failures = [
        message
        for name, pass_fn in _ALL_PASSES
        for message in _growth_failures(name, pass_fn, _CASE_FAMILY[name]())
    ]
    assert not failures, "\n".join(failures)


def test_no_pass_mutates_its_input_or_second_guesses_itself() -> None:
    failures = [
        message
        for name, pass_fn in _ALL_PASSES
        for message in _mutability_failures(name, pass_fn, _CASE_FAMILY[name]())
    ]
    assert not failures, "\n".join(failures)


def test_no_pass_reorders_the_instructions_it_kept() -> None:
    failures = [
        message
        for name, pass_fn in _ALL_PASSES
        for message in _order_failures(name, pass_fn, _CASE_FAMILY[name]())
    ]
    assert not failures, "\n".join(failures)


def test_every_pass_leaves_the_state_vector_alone() -> None:
    failures = [
        message
        for name, pass_fn in _UNITARY_PASSES
        for message in _state_failures(name, pass_fn)
    ]
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# The controls: each shared check is shown refusing a pass that breaks it.
# ---------------------------------------------------------------------------


def _with_trailing_full_turn(pass_fn: _Pass) -> _Pass:
    """Return a pass that is ``pass_fn`` and then one extra operation."""

    def mutant(ir: CircuitIR) -> CircuitIR:
        out = pass_fn(ir)
        wire = ir.instructions[0].wires[0]
        return replace(
            out,
            instructions=out.instructions + (Instruction("i", (wire,)),),
        )

    return mutant


def test_the_growth_check_rejects_a_pass_that_lengthens_a_program() -> None:
    """Measure the check that decides who may sit in the fixed-point loop.

    `_with_trailing_full_turn` is the smallest pass that lengthens a program by one
    operation and is otherwise `merge_self_inverse`. The growth check has to name it on
    a majority of cases and, just as importantly, the other three checks have to stay
    silent: an appended `i` preserves the state, the scaffolding and every wire's
    order, so a growth failure here cannot be a scaffolding or order failure wearing a
    different name.
    """

    mutant = _with_trailing_full_turn(compiler_pipeline.merge_self_inverse)
    cases = _unitary_cases()
    grown = _growth_failures("mutant", mutant, cases)
    assert len(grown) > len(cases) // 2, len(grown)
    assert all("operations became" in message for message in grown)
    assert not _scaffolding_failures("mutant", mutant, cases)
    assert not _order_failures("mutant", mutant, cases)
    assert not _state_failures("mutant", mutant)


def _with_shifted_rotation(pass_fn: _Pass) -> _Pass:
    """Return a pass that is ``pass_fn`` and then advances one rotation by a full turn.

    A full turn on a rotation is a global phase of pi: the operation is still declared,
    still executable, still the same length, and the program it produces has exactly the
    same overlap with the source. The mutant only fires when the output still carries a
    rotation, which is why the test below compares it against ``pass_fn`` itself rather
    than against the source to decide which cases it acted on.
    """

    def mutant(ir: CircuitIR) -> CircuitIR:
        out = pass_fn(ir)
        for index, instruction in enumerate(out):
            if instruction.name in ("rx", "ry", "rz") and "theta" in instruction.params:
                rewritten = list(out.instructions)
                rewritten[index] = replace(
                    instruction,
                    params={
                        **instruction.params,
                        "theta": instruction.params["theta"] + 2.0 * math.pi,
                    },
                )
                return replace(out, instructions=tuple(rewritten))
        return out

    return mutant


def test_the_state_check_rejects_a_pass_that_moves_a_global_phase() -> None:
    """Measure that the state check catches the defect the overlap cannot see.

    This is the load-bearing control of the file. The mutant above differs from
    `merge_self_inverse` by a full turn on one rotation -- the same length, the same
    declared opcode, the same scaffolding -- so `1 - |<in|out>|` stays at rounding and
    an overlap-based check would pass it. The vector difference has to fail it, on a
    majority of the cases, by a margin this test measures rather than assumes.
    """

    cases = _unitary_cases()
    shipped = compiler_pipeline.merge_self_inverse
    mutant = _with_shifted_rotation(shipped)

    fired: list[tuple[CircuitIR, CircuitIR]] = []
    for program in cases:
        out = mutant(program)
        if out.instructions != shipped(program).instructions:
            fired.append((program, out))
    assert len(fired) > len(cases) // 2, len(fired)

    smallest = math.inf
    worst = 0.0
    worst_overlap_gap = 0.0
    for program, out in fired:
        left, right = _state(program), _state(out)
        difference = float(torch.max(torch.abs(left - right)))
        assert difference > _CONTROL_STATE_GAP, difference
        smallest = min(smallest, difference)
        worst = max(worst, difference)
        worst_overlap_gap = max(
            worst_overlap_gap, float(abs(abs(torch.vdot(left, right)) - 1.0))
        )

    # The check names every case the mutant acted on, and the overlap is blind to all
    # of them: that pair is the reason this file compares vectors.
    assert len(_state_failures("mutant", mutant)) == len(fired)
    assert worst_overlap_gap < _STATE_ATOL, worst_overlap_gap
    assert smallest < worst or len(fired) == 1
    assert not _growth_failures("mutant", mutant, cases)
    assert not _scaffolding_failures("mutant", mutant, cases)
    assert not _order_failures("mutant", mutant, cases)


# ---------------------------------------------------------------------------
# The two passes whose observable is the outcome distribution.
# ---------------------------------------------------------------------------


def test_the_dynamic_passes_leave_the_outcome_distribution_alone() -> None:
    """Hold the two undeclared-opcode passes to shares instead of to amplitudes.

    Neither program can be executed by the statevector engine -- `gate_matrix` has no
    entry for `reset` or `measure` -- so the state-vector check above cannot reach
    them, and this is their whole postcondition.
    """

    failures: list[str] = []
    for seed in _RESET_SEEDS:
        program = _reset_case(seed)
        out = remove_zero_state_resets(program)
        gap = _share_gap(_shares(program), _shares(out))
        if gap > _SHARE_TOLERANCE:
            failures.append(f"remove_zero_state_resets seed={seed}: shares moved {gap}")
    for seed in _DIAGONAL_SEEDS:
        program = _measurement_case(seed)
        out = remove_diagonal_gates_before_measure(program)
        gap = _share_gap(_shares(program), _shares(out))
        if gap > _SHARE_TOLERANCE:
            failures.append(
                f"remove_diagonal_gates_before_measure seed={seed}: shares moved {gap}"
            )
    assert not failures, "\n".join(failures)


def test_the_outcome_instrument_can_see_a_removal_that_is_wrong() -> None:
    """Measure both controls, so a passing share comparison means something.

    Each control is a removal the corresponding pass refuses, taken by hand, and each
    has to move a share by 1.000 -- two outcomes that certainties swap between, with no
    sampling between them. That is the measurement behind `_SHARE_TOLERANCE`: a
    tolerance loose enough to absorb a move of this size would certify any removal at
    all, and the two passes' own legitimate removals move a share by at most 0.0197.

    Both controls also record why the pass refuses: a reset is only removable when
    nothing has touched its wire, and a diagonal gate is only removable when the next
    operation on each of its wires is an unconditional measurement.
    """

    # A reset on a wire `x` has already touched. Removing it turns a certain |0> into a
    # certain |1>, which is the largest change a share can show.
    touched = CircuitIR(
        2,
        (Instruction("x", (0,)), _reset(0), _measure(0, 0), _measure(1, 1)),
        dtype="complex128",
    )
    assert len(remove_zero_state_resets(touched)) == len(touched)
    reset_gap = _share_gap(
        _shares(touched),
        _shares(
            replace(
                touched,
                instructions=touched.instructions[0:1] + touched.instructions[2:],
            )
        ),
    )

    # A diagonal gate followed by another operation on its own wire: `h z h` is `x` and
    # `h h` is the identity, so the two programs measure 1 and 0 with certainty.
    sandwiched = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("z", (0,)),
            Instruction("h", (0,)),
            _measure(0, 0),
            _measure(1, 1),
        ),
        dtype="complex128",
    )
    assert len(remove_diagonal_gates_before_measure(sandwiched)) == len(sandwiched)
    diagonal_gap = _share_gap(
        _shares(sandwiched),
        _shares(
            replace(
                sandwiched,
                instructions=sandwiched.instructions[0:1] + sandwiched.instructions[2:],
            )
        ),
    )

    assert reset_gap > _CONTROL_SHARE, reset_gap
    assert diagonal_gap > _CONTROL_SHARE, diagonal_gap


class _Recorder:
    """Record the passes `_optimize_to_fixed_point` actually calls, in order.

    The roster exists in one place -- the loop body -- and the loop imports its passes
    inside the function, so there is nothing a test can read to learn the composition.
    This context manager patches each call site where the name is actually resolved and
    records what ran, which lets the composition be checked as code against code instead
    of prose against code.

    `remove_zero_state_resets` is patched on `compiler_pipeline`, not on its own module,
    because the loop's module imports that name at module scope: patching the definition
    site would leave the loop calling the original object and the recording would silently
    miss a pass.
    """

    _TARGETS: tuple[tuple[Any, str], ...] = (
        (compiler_pipeline, "remove_zero_state_resets"),
        (diagonal_before_measure, "remove_diagonal_gates_before_measure"),
        (compiler_pipeline, "remove_identity_gates"),
        (compiler_pipeline, "merge_self_inverse"),
        (inverse_cancellation, "merge_inverse_pairs"),
        (compiler_pipeline, "merge_adjacent_rotations"),
        (commutation_cancellation, "merge_commuting_rotations"),
        (commutation_cancellation, "cancel_commuting_self_inverse"),
        (one_qubit_optimization, "collapse_one_qubit_runs"),
        (two_qubit_optimization, "collapse_two_qubit_blocks"),
    )

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._restore: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> _Recorder:
        for module, name in self._TARGETS:
            original = getattr(module, name)

            def recorder(
                ir: CircuitIR,
                *args: Any,
                _name: str = name,
                _original: Any = original,
                **kwargs: Any,
            ) -> CircuitIR:
                self.calls.append(_name)
                return _original(ir, *args, **kwargs)

            self._restore.append((module, name, original))
            setattr(module, name, recorder)
        return self

    def __exit__(self, *_: Any) -> None:
        for module, name, original in reversed(self._restore):
            setattr(module, name, original)


def test_the_roster_is_the_round_body_the_pipeline_runs() -> None:
    """Tie the two enumerations together.

    `_ROUND_BODY` is what the pipeline does, `_ROSTER` is what the shared checks are
    driven over, and the only difference allowed between them is the repeated
    `remove_identity_gates`. A pass added to the loop and not to this file fails here.
    """

    assert set(_ROSTER) == {name for name, _ in _ALL_PASSES}
    assert len(_ROSTER) == len(_ALL_PASSES)
    assert set(_ROUND_BODY) == set(_ROSTER)
    assert len(_ROUND_BODY) == len(_ROSTER) + 1

    # Exactly one pass runs twice, and it is the one the loop's comments say runs
    # twice. Any other repetition would mean the loop body has a shape this file does
    # not describe.
    repeated = sorted({name for name in _ROUND_BODY if _ROUND_BODY.count(name) > 1})
    assert repeated == ["remove_identity_gates"]

    # The recorder has to watch the same roster, or the composition test below would
    # confirm an enumeration it never observed.
    assert {name for _, name in _Recorder._TARGETS} == set(_ROSTER)
    assert len(_Recorder._TARGETS) == len(_ROSTER)


def test_optimize_calls_exactly_the_enumerated_round_body() -> None:
    """Record the real composition and compare it with the enumeration.

    The loop needs more than one round on these programs, which is the property the
    fixed-point design exists for, so the recording is also the check that the
    program has not become one a single round would settle.
    """

    for seed in (0, 1, 7, 11, 29):
        program = _unitary_case(seed)
        with _Recorder() as recorder:
            optimize(program)
        rounds, remainder = divmod(len(recorder.calls), len(_ROUND_BODY))
        assert remainder == 0, recorder.calls
        assert rounds >= 2, f"seed={seed} settled in {rounds} round(s)"
        assert recorder.calls == list(_ROUND_BODY) * rounds, recorder.calls


def test_optimize_is_a_fixed_point_that_never_grows_and_preserves_the_state() -> None:
    """The entry point's own postconditions, which the passes alone do not give.

    `optimize` promises a fixed point, and it is the entry point a caller reads, so
    the promise is checked here rather than inferred from the loop.
    """

    for program in _unitary_cases():
        out = optimize(program)
        assert len(out) <= len(program), len(program)
        assert optimize(out).instructions == out.instructions
        assert out.n_wires == program.n_wires
        difference = float(torch.max(torch.abs(_state(program) - _state(out))))
        assert difference <= _STATE_ATOL, difference


def test_every_family_is_what_this_file_claims_it_is() -> None:
    """Record the shape of the matrix the other checks are stated over.

    Each family carries a claim the checks depend on: the unitary programs are
    executable and unitary-only, the measurement programs end in a diagonal run a
    measurement reads out, and the reset programs place resets on wires the body
    leaves idle. Those are the properties that give the passes their work, so they
    are asserted rather than assumed.
    """

    cases = _unitary_cases()
    assert len(cases) == len(_UNITARY_SEEDS) + len(_BOUNDARY_SEEDS)
    assert {case.n_wires for case in cases} == {_CASE_WIRES}
    assert {len(case) for case in cases} == {_CASE_GATES}
    assert all(case.dtype == "complex128" for case in cases)
    assert all(case.measurements == () for case in cases)

    # The two unitary families have to differ in their angle population, or the second
    # one would be a duplicate of the first rather than the population where the
    # exactness claim is actually at risk.
    def angles(case: CircuitIR) -> set[float]:
        return {value for i in case for value in i.params.values()}

    # A full turn inside a folded run is the shape that can carry a global phase, so
    # the boundary family has to contain one and the generic family has to be able to
    # produce one only by accident.
    assert any(
        any(abs(abs(value) - 2.0 * math.pi) < 1.0e-12 for value in angles(case))
        for case in cases[len(_UNITARY_SEEDS) :]
    )

    measurements = _measurement_cases()
    assert len(measurements) == len(_DIAGONAL_SEEDS)
    for case in measurements:
        assert case.measurements == ()
        assert [i.name for i in case][-case.n_wires :] == ["measure"] * case.n_wires
        assert any(i.name in _DIAGONAL_ONE_WIRE for i in case)

    resets = _reset_cases()
    assert len(resets) == len(_RESET_SEEDS)
    for case in resets:
        assert case.measurements == ()
        assert sum(1 for i in case if i.name == "reset") >= 1
        # The last wire is idle before its trailing reset, so that reset is a
        # mid-program candidate and not only a leading one.
        body = [i for i in case if i.name != "reset" and i.name != "measure"]
        assert all(case.n_wires - 1 not in i.wires for i in body)
