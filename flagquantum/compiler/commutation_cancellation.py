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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import cast

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import canonical_opcode
from .commutation import CommutationAnalysis, analyze_commutation
from .pipeline import _SELF_INVERSE


def _self_inverse_table(analyses: Mapping[str, object] | None) -> frozenset[str]:
    """The self-inverse opcode set this pass was handed, or the pipeline's own.

    The pass reads the fact rather than the pipeline's module global so that a
    caller can hand it a registry with a different `self_inverse_opcodes` and
    change what this pass removes without editing either module. Called with no
    facts it falls back to the same table the analysis returns, so the two routes
    cannot disagree about the default.
    """

    if analyses is None:
        return _SELF_INVERSE
    return cast(frozenset[str], analyses["self_inverse_opcodes"])


def _is_cancellable(
    instruction: Instruction, self_inverse: frozenset[str] = _SELF_INVERSE
) -> bool:
    """Whether ``instruction`` is one this pass may remove in a proven pair."""

    return (
        instruction.matrix is None
        and canonical_opcode(instruction.name) in self_inverse
        and not instruction.params
    )


def cancellable_positions(
    instructions: Sequence[Instruction],
    analysis: CommutationAnalysis,
    self_inverse: frozenset[str] = _SELF_INVERSE,
) -> Mapping[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[int, ...]]:
    """Group the cancellable positions of a program, keyed by block membership.

    The key is the opcode, the wires, and the block index on each of those wires.
    Two positions sharing a key are therefore the same self-inverse instruction on
    the same wires, with a proven commuting gap between them -- which is exactly
    the precondition `cancel_commuting_self_inverse` removes on. Exposed so that a
    benchmark can report how many groups a circuit offered, which is the only way
    to tell a pass that found nothing from a pass that was never asked.
    """

    groups: dict[tuple[str, tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for position, instruction in enumerate(instructions):
        if not _is_cancellable(instruction, self_inverse):
            continue
        signature = analysis.signature(position, instruction.wires)
        if signature is None:
            continue
        key = (canonical_opcode(instruction.name), instruction.wires, signature)
        groups.setdefault(key, []).append(position)
    return MappingProxyType({key: tuple(value) for key, value in groups.items()})


def _repeats_an_opcode(
    instructions: Sequence[Instruction],
    self_inverse: frozenset[str] = _SELF_INVERSE,
) -> bool:
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
        if not _is_cancellable(instruction, self_inverse):
            continue
        key = (canonical_opcode(instruction.name), instruction.wires)
        if key in seen:
            return True
        seen.add(key)
    return False


def cancel_commuting_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    """Remove self-inverse instructions separated only by proven commuters.

    Returns ``ir`` unchanged when no group offers a pair, which keeps the pass
    cheap on a circuit that has nothing to give and makes it idempotent: the
    surviving members of an odd group are one instruction apart in no block with
    each other, so a second application finds no new pair.

    Args:
        ir: The program to transform.
        analyses: The facts the pass declared it reads, as the pass manager hands
            them: ``self_inverse_opcodes`` and ``commutation_blocks``. When it is
            ``None`` the facts are read from the modules that own them, which is
            what a direct caller gets.
    """

    instructions = tuple(ir)
    self_inverse = _self_inverse_table(analyses)
    if not _repeats_an_opcode(instructions, self_inverse):
        return ir

    blocks = (
        analyze_commutation(ir)
        if analyses is None
        else cast(CommutationAnalysis, analyses["commutation_blocks"])
    )
    groups = cancellable_positions(instructions, blocks, self_inverse)
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


__all__ = ["cancel_commuting_self_inverse", "cancellable_positions"]
