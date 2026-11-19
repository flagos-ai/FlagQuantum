"""Six-round Kaliski inversion and compute/use/uncompute point addition."""

from dataclasses import dataclass

from .arithmetic import (
    MODULUS,
    ROUNDS,
    Gate,
    ReversibleBlock,
    Word,
    accumulate,
    equal,
    permute,
    product,
    rotate,
    translate,
    xor_function,
)


def _decision(u: int, v: int) -> int:
    """Classify one binary almost-inverse recurrence from its live words."""
    if not v:
        return 0
    if u % 2 == 0:
        return 1
    if v % 2 == 0:
        return 2
    return 3 if u > v else 4


@dataclass(frozen=True)
class InverseWorkspace:
    u: Word
    v: Word
    r: Word
    s: Word
    decisions: tuple[Word, ...]
    selected: int

    @classmethod
    def starting_at(cls, origin: int) -> "InverseWorkspace":
        words = tuple(tuple(range(i, i + 3)) for i in range(origin, origin + 30, 3))
        return cls(words[0], words[1], words[2], words[3], words[4:], origin + 30)


def inverse(source: Word, target: Word, workspace: InverseWorkspace) -> ReversibleBlock:
    """XOR the inverse into target, restoring all six decision histories.

    Six padded rounds give the Montgomery factor 2**6 == 1 modulo 7.
    The initial residue may be zero, and the unused encoding 7 also yields zero.
    """
    u, v, r, s = workspace.u, workspace.v, workspace.r, workspace.s
    forward = ReversibleBlock()
    for wire in u:
        forward.flip(wire)
    forward.flip(s[0])
    for wire, input_wire in zip(v, source, strict=True):
        forward.flip(wire, ((input_wire, 1),))

    double_r, double_s = ReversibleBlock(), ReversibleBlock()
    rotate(double_r, r)
    rotate(double_s, s)
    halve_u, halve_v = ReversibleBlock(), ReversibleBlock()
    rotate(halve_u, u, right=True)
    rotate(halve_v, v, right=True)
    reduce_u, reduce_v = ReversibleBlock(), ReversibleBlock()
    accumulate(reduce_u, v, u, scale=-1, modulus=8)
    reduce_u.extend(halve_u)
    accumulate(reduce_u, s, r)
    reduce_u.extend(double_s)
    accumulate(reduce_v, u, v, scale=-1, modulus=8)
    reduce_v.extend(halve_v)
    accumulate(reduce_v, r, s)
    reduce_v.extend(double_r)
    halve_u.extend(double_s)
    halve_v.extend(double_r)

    for decision in workspace.decisions:
        xor_function(
            forward,
            u + v,
            decision,
            lambda pair: _decision(pair & 7, pair >> 3),
        )
        for case, change in enumerate((double_r, halve_u, halve_v, reduce_u, reduce_v)):
            match = equal(decision, case)
            forward.flip(workspace.selected, match)
            forward.extend(change, ((workspace.selected, 1),))
            forward.flip(workspace.selected, match)

    result = ReversibleBlock()
    result.extend(forward)
    xor_function(result, r, target, lambda value: -value % MODULUS)
    result.extend(forward, inverse=True)
    return result


@dataclass(frozen=True)
class PointWorkspace:
    slope: Word
    reciprocal: Word
    inverse: InverseWorkspace
    ladder: Word

    @classmethod
    def starting_at(cls, origin: int) -> "PointWorkspace":
        return cls(
            tuple(range(origin, origin + 3)),
            tuple(range(origin + 3, origin + 6)),
            InverseWorkspace.starting_at(origin + 6),
            tuple(
                range(origin + 7 + 12 + 3 * ROUNDS, origin + 7 + 12 + 3 * ROUNDS + 5)
            ),
        )


def arithmetic_point_add(
    x: Word, y: Word, addend: tuple[int, int], workspace: PointWorkspace
) -> ReversibleBlock:
    """Translate the original nonsingular orbit with affine field arithmetic."""
    gx, gy = addend
    slope, reciprocal = workspace.slope, workspace.reciprocal
    reciprocal_block = inverse(x, reciprocal, workspace.inverse)
    slope_block = ReversibleBlock()
    slope_block.extend(reciprocal_block)
    slope_block.extend(product(slope, reciprocal, y))
    slope_block.extend(reciprocal_block, inverse=True)

    negate_x = ReversibleBlock()
    permute(negate_x, x, [(-i) % MODULUS for i in range(MODULUS)] + [MODULUS])
    slope_times_x = product(y, slope, x)
    block = ReversibleBlock()
    block.extend(translate(x, -gx))
    block.extend(translate(y, -gy))
    block.extend(slope_block)
    block.extend(slope_times_x, inverse=True)
    block.extend(product(x, slope, slope, scale=-1))
    block.extend(translate(x, 3 * gx))
    block.extend(slope_times_x)
    block.extend(slope_block, inverse=True)
    block.extend(negate_x)
    block.extend(translate(x, gx))
    block.extend(translate(y, -gy))
    return block


def point_add_gates(
    x: Word,
    y: Word,
    addend: tuple[int, int],
    workspace: PointWorkspace,
    *,
    control: int | None = None,
) -> list[Gate]:
    base = arithmetic_point_add(x, y, addend, workspace)
    if control is not None:
        controlled = ReversibleBlock()
        controlled.extend(base, ((control, 1),))
        base = controlled
    return base.lower(workspace.ladder)
