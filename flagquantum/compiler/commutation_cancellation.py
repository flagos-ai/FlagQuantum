"""Cancel self-inverse instructions that a commuting gap was hiding.

`pipeline.merge_self_inverse` removes two identical self-inverse gates when the
second immediately follows the first on its wires. "Immediately" is measured on
*wires*, so a gate on another wire in between is already no obstacle, but a gate
that touches one of the same wires is:

    cx(0, 1)  rz(0)  cx(0, 1)

is a bare `rz(0)`, and `merge_self_inverse` cannot see it, because the latest
writer of wire 0 is the `rz` rather than the first `cx`. `rz` on the *control* of
a `cx` commutes with it -- the `cx` is `sum_a |a><a|_0 (x) X^a`, and a diagonal
gate on wire 0 only rescales each term -- so the pair annihilates across the gap.
This pass is the consumer of `commutation.analyze_commutation`, which is what
proves that.

Three properties make a cancellation admissible.

**The gap must be proven, not assumed.** Two occurrences of one self-inverse
instruction are removed only when every instruction between them that shares a
wire with them is in the same commuting block on that wire. Instructions between
them that share no wire commute by disjointness, so that condition is exactly
"every instruction between them commutes with them", which is what makes the pair
reduce to `G G = I`. The proof does not care how many other pairs are removed at
the same time: an intervening instruction a pair relies on has already been proven
to commute with it, and deleting an instruction cannot make a true commutation
false.

**Nothing is removed that a later pass would need to see.** Only parameter-free
members of `_SELF_INVERSE` with no `matrix` override are candidates, so no angle
is read and no caller-supplied matrix is second-guessed. A `measure`, a `reset`, a
`barrier`, a non-unitary channel, and any opcode the rule source has no rule for
are not candidates and do not commute with the gates they touch, which makes them
barriers that stop a cancellation rather than participants in one.

**An odd count leaves one gate where it was.** A group of `2k` members loses all
of them; a group of `2k + 1` keeps its last, so what the block preserves is
applied as late as it was before. Two adjacent `cx` gates, the shape
`merge_self_inverse` already handles, are one such group, so this pass is a strict
superset of that one rather than a second opinion about it.

**Two rotations of one opcode merge across the same gap.** The cancellation above
is limited to gates that are their own inverse, which leaves the larger family of
*runs* untouched: `rz(0.3) cx(0.1) rz(0.4)` on the control is one `rz(0.7)` and
`cx(0.1)`, because a diagonal gate on the control of a `cx` commutes with it, and
`merge_adjacent_rotations` cannot see it because the latest writer of wire 0 is the
second `rz` rather than the first. This is the rule Qiskit's
`CommutativeCancellation` applies through the same analysis. Nothing new is
assumed: the group is the same group, the proof is the same proof, and the sum is
the same `pipeline._add_values` the adjacent merge already performs, with the same
`pipeline._is_zero` deciding that a vanished sum is a deletion. What changes is
only *where* the merged rotation may be written, which is the placement the
commuting gap licenses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import canonical_opcode
from .commutation import CommutationAnalysis, analyze_commutation
from .pipeline import (
    _ROTATION_PARAM,
    _SELF_INVERSE,
    _add_values,
    _is_zero,
    _replace_param,
)

#: A group key: the instruction's name, its wires, and its commuting block on
#: each of those wires. Two positions sharing one are the same operator on the
#: same wires with a proven commuting gap between them.
GroupKey = tuple[str, tuple[int, ...], tuple[int, ...]]


def _is_cancellable(instruction: Instruction) -> bool:
    """Whether ``instruction`` is one this pass may remove in a proven pair."""

    return (
        instruction.matrix is None
        and canonical_opcode(instruction.name) in _SELF_INVERSE
        and not instruction.params
    )


def _rotation_param(instruction: Instruction) -> str | None:
    """The rotation parameter ``instruction`` may have added, or None.

    The conditions are `pipeline.merge_adjacent_rotations`'s own, deliberately:
    a rotation this rule may add is one that pass would have added had the two
    halves been adjacent, so the commuting merge cannot read an angle the
    adjacent merge refuses. A `matrix` override is not second-guessed, an opcode
    `_ROTATION_PARAM` has no name for is not a rotation, and a missing parameter
    is declined rather than defaulted to zero.
    """

    if instruction.matrix is not None:
        return None
    param_name = _ROTATION_PARAM.get(instruction.name)
    if param_name is None or param_name not in instruction.params:
        return None
    return param_name


def cancellable_positions(
    instructions: Sequence[Instruction], analysis: CommutationAnalysis
) -> Mapping[GroupKey, tuple[int, ...]]:
    """Group the cancellable positions of a program, keyed by block membership.

    The key is the opcode, the wires, and the block index on each of those wires.
    Two positions sharing a key are therefore the same self-inverse instruction on
    the same wires, with a proven commuting gap between them -- which is exactly
    the precondition `cancel_commuting_self_inverse` removes on. Exposed so that a
    benchmark can report how many groups a circuit offered, which is the only way
    to tell a pass that found nothing from a pass that was never asked.
    """

    groups: dict[GroupKey, list[int]] = {}
    for position, instruction in enumerate(instructions):
        if not _is_cancellable(instruction):
            continue
        signature = analysis.signature(position, instruction.wires)
        if signature is None:
            continue
        key = (canonical_opcode(instruction.name), instruction.wires, signature)
        groups.setdefault(key, []).append(position)
    return MappingProxyType({key: tuple(value) for key, value in groups.items()})


def rotation_groups(
    instructions: Sequence[Instruction], analysis: CommutationAnalysis
) -> Mapping[GroupKey, tuple[int, ...]]:
    """Group the positions whose rotation parameters may be added together.

    The keying is `cancellable_positions`'s, and for the same reason: one key
    means one operator on one set of wires, with a proven commuting gap between
    every pair of members. A member may be merged into the group's first position
    because everything the group spans commutes with all of it, so the block the
    group forms is applied where the earliest member already sat.
    """

    groups: dict[GroupKey, list[int]] = {}
    for position, instruction in enumerate(instructions):
        if _rotation_param(instruction) is None:
            continue
        signature = analysis.signature(position, instruction.wires)
        if signature is None:
            continue
        key = (instruction.name, instruction.wires, signature)
        groups.setdefault(key, []).append(position)
    return MappingProxyType({key: tuple(value) for key, value in groups.items()})


def _repeats_an_opcode(instructions: Sequence[Instruction]) -> bool:
    """Whether some self-inverse opcode appears twice on the same wires.

    A short circuit, not a semantic one. A circuit that does not repeat one of
    these opcodes on one set of wires has no pair for this pass to remove, so
    answering ``False`` here is final; the analysis behind the removal is
    quadratic in the length of a commuting block, so it is not worth running to
    learn that. The converse does not hold, and the direction matters: a repeat
    may still be split across two commuting blocks and therefore stay, so
    ``True`` is permission to look rather than a promise that something will be
    removed. Being wrong in that direction costs one analysis call; being wrong
    in the other would skip a removal the caller could have had.
    """

    seen: set[tuple[str, tuple[int, ...]]] = set()
    for instruction in instructions:
        if not _is_cancellable(instruction):
            continue
        key = (canonical_opcode(instruction.name), instruction.wires)
        if key in seen:
            return True
        seen.add(key)
    return False


def _repeats_a_rotation(instructions: Sequence[Instruction]) -> bool:
    """Whether some rotation opcode appears twice on the same wires.

    `_repeats_an_opcode`'s short circuit for `merge_commuting_rotations`, with
    the same one-sided reading: ``False`` is final, because a circuit that does
    not repeat one of these opcodes on one set of wires has no group of two to
    add up, and ``True`` is permission to look rather than a promise that
    anything will merge.
    """

    seen: set[tuple[str, tuple[int, ...]]] = set()
    for instruction in instructions:
        if _rotation_param(instruction) is None:
            continue
        key = (instruction.name, instruction.wires)
        if key in seen:
            return True
        seen.add(key)
    return False


def merge_commuting_rotations(ir: CircuitIR) -> CircuitIR:
    """Add up rotations of one opcode that a proven commuting gap separated.

    Returns ``ir`` unchanged when no group holds two members, which keeps the
    pass cheap on a circuit that has nothing to give and makes it idempotent in
    the sense `_optimize_to_fixed_point` needs: a merge lowers the number of
    rotation instructions on those wires, so a second application finds every
    group at one member and adds nothing.

    A group whose angles add to zero loses its first position as well, which is
    the same deletion `merge_adjacent_rotations` performs for the adjacent case
    and uses the same ``_is_zero``. A trainable angle is never zero by that
    reading, so a merged rotation that is still differentiable stays in the
    program and stays in the autograd graph.
    """

    instructions = tuple(ir)
    if not _repeats_a_rotation(instructions):
        return ir

    groups = rotation_groups(instructions, analyze_commutation(ir))
    rewritten: dict[int, Instruction] = {}
    removed: set[int] = set()
    for positions in groups.values():
        if len(positions) < 2:
            continue
        head = instructions[positions[0]]
        param_name = _rotation_param(head)
        if param_name is None:
            continue
        total: Any = head.params[param_name]
        for position in positions[1:]:
            total = _add_values(total, instructions[position].params[param_name])
            removed.add(position)
        if _is_zero(total):
            removed.add(positions[0])
        else:
            rewritten[positions[0]] = _replace_param(head, param_name, total)

    if not removed:
        return ir

    return replace(
        ir,
        instructions=tuple(
            rewritten.get(position, instruction)
            for position, instruction in enumerate(instructions)
            if position not in removed
        ),
    )


def cancel_commuting_self_inverse(ir: CircuitIR) -> CircuitIR:
    """Remove self-inverse instructions separated only by proven commuters.

    Returns ``ir`` unchanged when no group offers a pair, which keeps the pass
    cheap on a circuit that has nothing to give and makes it idempotent: the
    surviving members of an odd group are one instruction apart in no block with
    each other, so a second application finds no new pair.
    """

    instructions = tuple(ir)
    if not _repeats_an_opcode(instructions):
        return ir

    groups = cancellable_positions(instructions, analyze_commutation(ir))
    removed: set[int] = set()
    for positions in groups.values():
        removed.update(positions[: len(positions) - len(positions) % 2])
    if not removed:
        return ir

    return replace(
        ir,
        instructions=tuple(
            instruction
            for position, instruction in enumerate(instructions)
            if position not in removed
        ),
    )


__all__ = [
    "cancel_commuting_self_inverse",
    "cancellable_positions",
    "merge_commuting_rotations",
    "rotation_groups",
]
