"""Collapse a block over one ordered wire pair into one exact gate.

`pipeline` already removes identities, cancels self-inverse and declared inverse
pairs, adds up adjacent rotations that share one opcode, and folds a same-wire run
of *single*-qubit gates. None of those compose two-qubit gates, which is where a
routed program spends its budget: a conjugation (`swap cz swap` is exactly `cz`, so
a router's swap that only moves a control is removable), an accumulation
(`rzz(a) rzz(b)` is `rzz(a + b)`), and a block whose *product* is the identity, which
this pass sees and `remove_identity_gates` does not, because that pass reads a zero
parameter rather than a product. This module composes such a block and re-spells it
as one declared two-qubit opcode: the reduction Qiskit's `ConsolidateBlocks` performs
by resynthesizing a block against a target basis, and the one `OptimizeCliffords`
performs on a run of its native `Clifford` objects. It is done here over the declared
opcode vocabulary and without a target basis, because FlagQuantum carries no opaque
Clifford type, `CollectCliffords`' gate-name filter has no counterpart to select on,
and `optimize` is target-independent.

Five properties make the fold admissible.

**The block is one ordered wire pair.** A member's wires must be a subset of the
pair's, so the control/target roles are the same for every two-wire member and one
matrix convention describes all of them. `wires=(0, 1)` and `wires=(1, 0)` are
different blocks even though they share both wires.

**A single-qubit gate on one of the pair's wires is a member.** That is the
difference from a run, and it is what reaches `h(1) cz(0, 1) h(1)`, which is exactly
`cx(0, 1)`: a conjugation by a gate one of the pair's own wires carries, rather than
by a two-qubit gate alone. Which wire carries the conjugation is a property of the
pair's *order* and not of the gate -- the same three gates with the `h`s on `0`
compose to `cx(1, 0)`, a gate on the other ordered pair, and are declined, which is
what keeps this a rewrite by matrix equality rather than by relabelling. Such a
member is lifted by a Kronecker product on the side the pair gives it, so the block's
product is the 4x4 operator the program computes and not a phase-free shadow of it. A
one-wire instruction never *opens* a block, because `h(0)` alone does not say whether
it belongs to `(0, 1)` or `(0, 2)`; it is instead drawn into a block at the moment a
two-wire gate opens one, out of the contiguous single-qubit gates immediately to the
left, and it joins a block that is already open. Two open blocks can never share a
wire, which is what makes that assignment unambiguous without a general dependency
graph.

**The replacement is one declared opcode or nothing, and it is verified.** A
candidate angle is *guessed* from the product by inverting one entry of the
family's closed form; the decision is then an entry-for-entry comparison of the
whole product against the candidate's own matrix, so a wrong guess is rejected
rather than emitted. That is what lets the guess be cheap without the pass
becoming approximate in a way nothing checks. A replacement is one gate at most:
general two-qubit resynthesis is deliberately not attempted here, and
`synthesize_two_qubit` is the module that decomposes an arbitrary unitary.

**A block is replaced only when the replacement is strictly shorter, and the
narrower rule is kept as a floor.** Equal length is not a win, so a single gate is
never re-spelled as itself and the pass is idempotent, which
`_optimize_to_fixed_point` relies on. A single-gate block is therefore asked one
question only -- whether its product is the identity, which is what `cphase(2*pi)`,
`crz(4*pi)` and `rzz(4*pi)` are and what `remove_identity_gates` does not see,
because that pass reads a zero parameter. A block the widened membership cannot
re-spell is then offered to the same rule one two-qubit sub-block at a time, split at
each single-qubit member, so widening the membership can never cost a fold the
narrower rule would have taken; `benchmarks/compiler_two_qubit_optimization.py`
measures that floor as its `runs_only` arm and asserts it circuit by circuit.

**No trainable angle takes part.** Composition reads numbers, and a trainable angle
has none; the same reason `one_qubit_synthesis` refuses to select a branch on one.
A block carrying a trainable angle is left to `merge_adjacent_rotations`, which adds
angles with the caller's own objects and keeps them in the autograd graph.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import (
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
)

#: A 4x4 matrix of Python complex numbers, rows first.
Matrix = list[list[complex]]

#: The declared two-qubit unitary opcodes, read from the operator schema rather
#: than written out, so an opcode added to the table is a failing assertion here
#: instead of a gate this pass silently cannot fold.
_DECLARED_TWO_QUBIT: frozenset[str] = frozenset(
    opcode
    for opcode, schema in OPERATOR_SCHEMAS.items()
    if schema.arity == 2 and schema.semantic_kind == "unitary"
)

#: The declared two-qubit opcodes this module owns a closed form for, in the order
#: a product is offered them. A product can equal more than one -- `cphase(pi)` is
#: exactly `cz` -- so the order is a preference, not a decision: `_same` decides,
#: and the parameter-free opcodes come first because they carry no angle that a
#: reader had to reconstruct.
_FAMILIES: tuple[str, ...] = (
    "cx",
    "cy",
    "cz",
    "swap",
    "crx",
    "cry",
    "crz",
    "cphase",
    "rxx",
    "ryy",
    "rzz",
)

if (
    frozenset(_FAMILIES) != _DECLARED_TWO_QUBIT
):  # pragma: no cover - a schema edit trips this
    raise AssertionError(
        "two_qubit_optimization has no closed form for "
        f"{sorted(_DECLARED_TWO_QUBIT - frozenset(_FAMILIES))} and claims "
        f"{sorted(frozenset(_FAMILIES) - _DECLARED_TWO_QUBIT)} that the schema "
        "does not declare as a two-qubit unitary"
    )

#: The largest entry difference a candidate replacement may carry. An accepted run
#: is replaced by a gate whose matrix was reconstructed from the product, so the two
#: differ by the accumulated float noise of composing the run rather than by
#: anything the source asked for. `benchmarks/compiler_two_qubit_optimization.py`
#: measures the worst case it actually reaches and the dense sweep that bounds how
#: often the comparison can accept a product it should not.
_MATCH_EPS = 1.0e-12

#: The rotation families have period `4*pi` and `cphase` has period `2*pi`, while
#: every reader below returns the principal branch. An angle is a guess that
#: `_same` accepts or rejects, so the neighbouring branches are offered too.
_PERIOD_STEPS = (-1, 0, 1)

_IDENTITY: Matrix = [
    [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
    [0.0j, 1.0 + 0.0j, 0.0j, 0.0j],
    [0.0j, 0.0j, 1.0 + 0.0j, 0.0j],
    [0.0j, 0.0j, 0.0j, 1.0 + 0.0j],
]


def _fixed_matrix(opcode: str) -> Matrix:
    """The runtime matrix of one parameter-free two-qubit opcode.

    Read with ``wires[0]`` on the most significant index bit, which is the
    convention `simulation.gate_matrix` uses and this module may not import.
    `tests/unit/test_compilation_two_qubit_optimization.py` pins every entry
    against that table, which is the authority on what these gate names compute.
    """

    if opcode == "cx":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    if opcode == "cy":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0j],
            [0.0, 0.0, 1.0j, 0.0],
        ]
    if opcode == "cz":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    if opcode == "swap":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    raise AssertionError(f"{opcode!r} is not a parameter-free family")


def _rotation_matrix(opcode: str, theta: float) -> Matrix:
    """The runtime matrix of one rotation family at ``theta``.

    Three shapes, one per way a family couples its wires. A **controlled** rotation
    is the identity on the control-off half and a single-qubit rotation on the
    control-on half. An **entangler** rotation is ``exp(-i theta/2 P)`` for `P` one
    of `XX`, `YY`, `ZZ`, and `rxx`/`ryy` differ only in the relative sign of their
    two off-diagonal blocks, because `YY` is `XX` with the second wire conjugated by
    `Z`. `cphase` is diagonal and carries the global phase ``exp(i theta/2)`` that
    makes it `cphase` rather than `cp`: at ``theta = pi`` it is exactly `cz`, which
    is why `cz` is offered first.
    """

    cosine = math.cos(0.5 * theta)
    sine = math.sin(0.5 * theta)
    if opcode == "cphase":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, cmath.exp(1.0j * theta)],
        ]
    if opcode == "crz":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cmath.exp(-0.5j * theta), 0.0],
            [0.0, 0.0, 0.0, cmath.exp(0.5j * theta)],
        ]
    if opcode == "crx":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cosine, -1.0j * sine],
            [0.0, 0.0, -1.0j * sine, cosine],
        ]
    if opcode == "cry":
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cosine, -sine],
            [0.0, 0.0, sine, cosine],
        ]
    if opcode == "rzz":
        return [
            [cmath.exp(-0.5j * theta), 0.0, 0.0, 0.0],
            [0.0, cmath.exp(0.5j * theta), 0.0, 0.0],
            [0.0, 0.0, cmath.exp(0.5j * theta), 0.0],
            [0.0, 0.0, 0.0, cmath.exp(-0.5j * theta)],
        ]
    if opcode in ("rxx", "ryy"):
        first = 1.0j * sine if opcode == "ryy" else -1.0j * sine
        second = -first if opcode == "ryy" else first
        return [
            [cosine, 0.0, 0.0, first],
            [0.0, cosine, second, 0.0],
            [0.0, second, cosine, 0.0],
            [first, 0.0, 0.0, cosine],
        ]
    raise AssertionError(f"{opcode!r} is not a rotation family")


def _principal_angle(opcode: str, product: Matrix) -> float:
    """The principal-branch angle one product would have been built at.

    Each reader inverts the one entry of the family's closed form that a composition
    of gates leaves legible: a diagonal entry for the diagonal families, and the
    control-on block's corner for the controlled ones -- the controlled families are
    the identity on the control-off half, so their only informative entries are the
    last two rows and columns. `atan2` is used wherever the family's two entries
    give a cosine and a sine, because a phase alone cannot separate `theta` from
    `theta + 2*pi` there.
    """

    if opcode == "crz":
        return -2.0 * cmath.phase(product[2][2])
    if opcode == "cphase":
        return cmath.phase(product[3][3])
    if opcode == "rzz":
        return -2.0 * cmath.phase(product[0][0])
    if opcode == "crx":
        return 2.0 * math.atan2(-product[2][3].imag, product[2][2].real)
    if opcode == "cry":
        return 2.0 * math.atan2(-product[2][3].real, product[2][2].real)
    if opcode == "rxx":
        return 2.0 * math.atan2(-product[0][3].imag, product[0][0].real)
    if opcode == "ryy":
        return 2.0 * math.atan2(product[0][3].imag, product[0][0].real)
    raise AssertionError(f"{opcode!r} is not a rotation family")


def _same(left: Matrix, right: Matrix) -> bool:
    """Whether two 4x4 matrices agree to `_MATCH_EPS` entry for entry."""

    return all(
        abs(left[row][column] - right[row][column]) <= _MATCH_EPS
        for row in range(4)
        for column in range(4)
    )


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """The product that applies `right` first."""

    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _scalar_angle(value: Any) -> float | None:
    """`value` as one angle, or None when it is not exactly one angle.

    `one_qubit_synthesis._polar_angle` answers "does this value have a number",
    which a one-element array also does. This asks the narrower question a fold
    needs: is this value *the* angle. Only a true scalar passes, and the import is
    local for the same reason `pipeline.remove_identity_gates` uses one --
    `one_qubit_synthesis` reads `_is_zero` back out of `pipeline`.
    """

    if getattr(value, "ndim", 0) != 0:
        return None
    from .one_qubit_synthesis import _polar_angle

    return _polar_angle(value)


def _as_complex_matrix(matrix: Any) -> Matrix | None:
    """Return `matrix` as four rows of four complex numbers, or None.

    A runtime gate matrix carries a leading batch axis, so a four-by-four matrix is
    the only shape one run can compose: a batch of gates is not one operator and is
    refused rather than read at index zero.
    """

    try:
        rows = [[complex(entry) for entry in row] for row in matrix]
    except (TypeError, ValueError):
        return None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return None
    return rows


def _instruction_matrix(instruction: Instruction) -> Matrix | None:
    """One two-qubit instruction's exact matrix, or None.

    None means the instruction cannot take part in a fold: it is not a declared
    two-qubit unitary, one of its parameters is trainable and therefore has no
    number to compose with, or one of them is a batch rather than one value.
    """

    if instruction.matrix is not None:
        return _as_complex_matrix(
            getattr(instruction.matrix, "tensor", instruction.matrix)
        )
    # A fold has to read the angle, where the passes beside it only have to add it,
    # so a batched parameter is refused here rather than left to the runtime.
    # Reading one element of a batch of angles would turn a program the runtime
    # refuses into one that runs with one row broadcast to every batch entry -- a
    # silent result change, not a simplification.
    if any(_scalar_angle(value) is None for value in instruction.params.values()):
        return None
    opcode = canonical_opcode(instruction.name)
    schema = get_operator_schema(opcode)
    if schema is None or opcode not in _FAMILIES:
        return None
    if not schema.parameters:
        return _fixed_matrix(opcode)
    angle = _scalar_angle(instruction.params.get(schema.parameters[0]))
    if angle is None:  # pragma: no cover - guarded by the loop above
        return None
    return _rotation_matrix(opcode, float(angle))


def _kron(
    left: Sequence[Sequence[complex]], right: Sequence[Sequence[complex]]
) -> Matrix:
    """The Kronecker product of two 2x2 factors, as one 4x4 matrix."""

    return [
        [
            left[row // 2][column // 2] * right[row % 2][column % 2]
            for column in range(4)
        ]
        for row in range(4)
    ]


#: The factor a lifted single-qubit member puts on the wire of the pair it does not
#: touch.
_IDENTITY_FACTOR: list[list[complex]] = [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]]


def _lifted_matrix(instruction: Instruction, pair: tuple[int, ...]) -> Matrix | None:
    """One block member's exact 4x4 matrix on `pair`, or None.

    None means the instruction cannot take part in a block: its wires are not a
    subset of the pair's, it is not a declared unitary at its own arity, one of its
    angles is trainable and therefore has no number to compose with, or one of them
    is a batch rather than one value.

    A two-wire member spans the pair and is read by `_instruction_matrix`. A one-wire
    member is read as a single-qubit gate -- by `one_qubit_optimization`'s reader, so
    that the convention phase the runtime applies to `rz`, `phase` and `u1` is part
    of the factor -- and lifted by a Kronecker product on the side the pair gives it:
    `pair[0]` is the most significant index bit, so a gate on `pair[0]` multiplies
    the first factor. Lifting the runtime matrix rather than a phase-free shadow of
    it is what makes the comparison in `_replacement` a comparison against what the
    program computes.
    """

    if len(instruction.wires) == 2:
        if tuple(instruction.wires) != pair:
            return None
        return _instruction_matrix(instruction)
    if len(instruction.wires) != 1 or instruction.wires[0] not in pair:
        return None
    from .one_qubit_optimization import _instruction_matrix as _one_qubit_matrix

    factor = _one_qubit_matrix(instruction)
    if factor is None:
        return None
    if instruction.wires[0] == pair[0]:
        return _kron(factor, _IDENTITY_FACTOR)
    return _kron(_IDENTITY_FACTOR, factor)


def _block_product(
    instructions: Sequence[Instruction], pair: tuple[int, ...]
) -> Matrix | None:
    """The exact product of one block, or None when any member is unreadable."""

    product = _IDENTITY
    for instruction in instructions:
        matrix = _lifted_matrix(instruction, pair)
        if matrix is None:
            return None
        product = _matmul(matrix, product)
    return product


def _is_foldable(instruction: Instruction) -> bool:
    """Whether one instruction is a readable two-qubit unitary over its own pair.

    False for anything else -- a wider or narrower gate, a measurement, a barrier, a
    channel, an opcode no family covers, an unreadable matrix, or a trainable angle,
    which has no number to compose with -- and such an instruction *ends* every block
    on each wire it reaches rather than joining one. Only a two-wire instruction can
    open a block: a single-qubit gate on one of a block's wires is a member of it,
    but `h(0)` alone does not say whether it belongs to `(0, 1)` or `(0, 2)`.
    """

    if len(instruction.wires) != 2:
        return False
    if (
        instruction.matrix is None
        and canonical_opcode(instruction.name) not in _FAMILIES
    ):
        return False
    return _instruction_matrix(instruction) is not None


def _draws_in_as_leading_member(
    instruction: Instruction, pair: tuple[int, ...]
) -> bool:
    """Whether a one-wire instruction may be a member of a block on `pair`.

    This is the whole of what the block rule here has that the two-qubit-run rule it
    replaces did not: a readable single-qubit gate on either wire of the pair. It is a
    module-level function because it is the seam
    `benchmarks/compiler_two_qubit_optimization.py` patches to measure the pass against
    the narrower rule that had no single-qubit members at all -- patching one
    predicate, not a second copy of the loop.
    """

    if len(instruction.wires) != 1 or instruction.wires[0] not in pair:
        return False
    return _lifted_matrix(instruction, pair) is not None


def _joins_pair(instruction: Instruction, pair: tuple[int, ...]) -> bool:
    """Whether `instruction` is a member of the block open on `pair`.

    `pair` is an ordered pair, so a two-wire member has to match it exactly and keep
    the control/target roles the block was opened with. A one-wire member is decided by
    `_draws_in_as_leading_member`. Nothing else joins.
    """

    if len(instruction.wires) == 2:
        return tuple(instruction.wires) == pair and _is_foldable(instruction)
    if len(instruction.wires) != 1:
        return False
    return _draws_in_as_leading_member(instruction, pair)


def _declines_vocabulary(instructions: Sequence[Instruction], opcode: str) -> bool:
    """Whether the fold declines to re-spell a block as `opcode`.

    A parameter-free replacement is never declined: `cx`, `cy`, `cz` and `swap` are
    the gates every parameterized two-qubit gate in this schema is built out of, so
    replacing a block with one of them is a strictly shorter program in every basis
    this repository can measure.

    A parameterized replacement *is* declined unless the block was already spelled
    in that same family. Such a replacement is a different parameterization of the
    two-qubit group rather than a more primitive operation, and `optimize` is
    target-independent: it cannot see whether the target publishes the gate the run
    was spelled in or the gate it would be re-spelled as. The same-family case is
    the one a target's own lowering cannot have to undo, because the replacement is
    the very gate the run consisted of. That is the reasoning
    `one_qubit_optimization._single_basis_pair` applies to the z-rotation/pulse
    basis, and `benchmarks/compiler_two_qubit_optimization.py` measures both sides of
    it: over four seeded populations the shipped rule improves the target-legal gate
    count on every basis with no circuit regressing, while allowing the rest
    regresses up to 16 circuits of 120 on `ibm-rz-sx-cx` and leaves the legal count
    larger than it found it.
    """

    schema = get_operator_schema(opcode)
    if schema is None:  # pragma: no cover - guarded by the caller
        return True
    if not schema.parameters:
        return False
    return any(canonical_opcode(item.name) != opcode for item in instructions)


def _replacement(
    instructions: Sequence[Instruction],
    *,
    pair: tuple[int, ...],
    metadata: Mapping[str, Any],
) -> tuple[Instruction, ...] | None:
    """The exact replacement of one block, or None to leave it.

    None covers refusal as well as no gain. Every member here is foldable and
    readable, so the only refusals are the ones `_declines_vocabulary` describes and
    a product no declared opcode matches.

    A single-member block is asked one question only -- whether it *is* the identity.
    Nothing is shorter than one gate except nothing at all, and re-spelling one
    declared gate as an equal-length one is not a win and would not terminate.
    """

    product = _block_product(instructions, pair)
    if product is None:  # pragma: no cover - guarded by the caller
        return None
    if _same(product, _IDENTITY):
        return ()
    if len(instructions) == 1:
        return None
    for opcode in _FAMILIES:
        if _declines_vocabulary(instructions, opcode):
            continue
        schema = get_operator_schema(opcode)
        if schema is None:  # pragma: no cover - the assertion above guards this
            continue
        if not schema.parameters:
            if _same(product, _fixed_matrix(opcode)):
                return (Instruction(opcode, pair, metadata=metadata),)
            continue
        for theta in (
            _principal_angle(opcode, product) + step * 2.0 * math.pi
            for step in _PERIOD_STEPS
        ):
            if _same(product, _rotation_matrix(opcode, theta)):
                return (
                    Instruction(
                        opcode,
                        pair,
                        params={schema.parameters[0]: theta},
                        metadata=metadata,
                    ),
                )
    return None


def _write_replacement(
    replaced: list[Instruction | None],
    positions: Sequence[int],
    replacement: Sequence[Instruction],
) -> None:
    """Put `replacement` at the front of `positions` and drop the rest."""

    for offset, instruction in enumerate(replacement):
        replaced[positions[offset]] = instruction
    for position in positions[len(replacement) :]:
        replaced[position] = None


def _two_qubit_sub_runs(
    positions: Sequence[int], output: Sequence[Instruction]
) -> list[list[int]]:
    """The maximal all-two-qubit sub-runs of one block's member positions.

    These are the runs the pass composed before single-qubit members existed, and
    they are what the floor rule is applied to when the block as a whole is refused.
    """

    runs: list[list[int]] = []
    contiguous = False
    for position in positions:
        if len(output[position].wires) != 2:
            contiguous = False
            continue
        if contiguous:
            runs[-1].append(position)
        else:
            runs.append([position])
        contiguous = True
    return runs


def collapse_two_qubit_blocks(ir: CircuitIR) -> CircuitIR:
    """Collapse each block over one ordered wire pair into one exact gate.

    A block is maximal over one ordered wire pair. A two-qubit gate opens one, and
    draws in the contiguous single-qubit gates immediately to its left that sit on
    one of its wires -- which is what lets a conjugation by a *single*-qubit gate be
    folded, `h(1) cz(0, 1) h(1)` being exactly `cx(0, 1)`. A gate that reaches a
    block's wires and is not readable as a member ends it, because a controlled gate
    does not commute past an arbitrary single-qubit gate; a gate that touches neither
    of its wires commutes with it and leaves it open. Two open blocks can never share a wire,
    because the second gate to reach a wire joins the block already there.

    Every instruction belongs to **at most one** block, which the draw-in has to be
    told rather than derive: a single-qubit gate between two blocks on different
    ordered pairs is a legitimate trailing member of the first and a legitimate
    leading member of the second, so a backward scan that ignored the first block
    would claim it twice and then write two replacements through it. The scan
    therefore stops at any index already claimed, closed blocks included.

    The replacement is written at the block's earliest member, so it keeps its place
    in program order, and every other member is pairwise commuting with everything it
    moves past.

    A block is left untouched when a member carries a trainable angle or a batch of
    angles, when its product matches no declared two-qubit opcode, or when the
    replacement would not be strictly shorter. A block the widened membership cannot
    re-spell is then offered to the narrower rule -- one all-two-qubit sub-run at a
    time -- so this pass never returns a longer program than the one that composed
    two-qubit runs alone.
    """

    output: list[Instruction] = []
    open_blocks: dict[tuple[int, ...], list[int]] = {}
    closed_blocks: list[tuple[tuple[int, ...], list[int]]] = []
    #: Output indices already owned by a block, open or closed. Membership is
    #: exclusive because two blocks that both fold through one instruction would each
    #: write their replacement at it, and only one of the two can be what the program
    #: computes.
    claimed: set[int] = set()

    def close_interfering(touched: Sequence[int], keep: tuple[int, ...] | None) -> None:
        """End every open block this instruction reaches, except `keep`.

        A block on a pair this instruction does not touch commutes with it and stays
        open. `keep` is the pair this instruction is joining, which must survive its
        own arrival.
        """

        arriving = set(touched)
        for pair in [p for p in open_blocks if p != keep and arriving & set(p)]:
            closed_blocks.append((pair, open_blocks.pop(pair)))

    for instruction in ir:
        joining: tuple[int, ...] | None = None
        for pair in open_blocks:
            if _joins_pair(instruction, pair):
                joining = pair
                break
        close_interfering(instruction.wires, joining)
        if joining is not None:
            open_blocks[joining].append(len(output))
            claimed.add(len(output))
        elif len(instruction.wires) == 2 and _is_foldable(instruction):
            pair = tuple(instruction.wires)
            members: list[int] = []
            back = len(output) - 1
            while back >= 0 and back not in claimed:
                if not _draws_in_as_leading_member(output[back], pair):
                    break
                members.append(back)
                back -= 1
            members.reverse()
            members.append(len(output))
            open_blocks[pair] = members
            claimed.update(members)
        output.append(instruction)
    for pair, members in open_blocks.items():
        closed_blocks.append((pair, members))

    replaced: list[Instruction | None] = list(output)
    for pair, members in closed_blocks:
        # Program order: the assembly appends members in the order the program puts
        # them in and the sub-run splitter reads that order, so an out-of-order member
        # list would split a block into the wrong sub-runs rather than fail.
        positions = sorted(members)
        block = _replacement(
            [output[position] for position in positions],
            pair=pair,
            metadata=output[positions[0]].metadata,
        )
        if block is not None:
            _write_replacement(replaced, positions, block)
            continue
        for run in _two_qubit_sub_runs(positions, output):
            narrower = _replacement(
                [output[position] for position in run],
                pair=pair,
                metadata=output[run[0]].metadata,
            )
            if narrower is not None:
                _write_replacement(replaced, run, narrower)
    return replace(
        ir,
        instructions=tuple(item for item in replaced if item is not None),
    )


__all__ = ["collapse_two_qubit_blocks"]
