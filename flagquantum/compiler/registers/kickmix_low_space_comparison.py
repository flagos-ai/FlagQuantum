# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_cmp_low_space.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators/gen_cmp_low_space.cc

"""Three-clean-wire comparison and its measured prefix corrections."""

from ...core.ir import Instruction
from ._kickmix_comparison_gates import _flip, _parity_controlled_flip
from .finite_field import ClassicalFieldCoefficient
from .measurement import measure_x_clear_ir
from .polynomial import _validate_registers


def _comparison_passes(width: int) -> tuple[tuple[int, int], ...]:
    passes: list[tuple[int, int]] = []
    while (shift := (2 << len(passes)) + len(passes) + 1) <= width:
        passes.append((width - shift, 2 << len(passes)))
    return tuple(passes)


def _prefix_phase(
    targets: tuple[int, ...],
    masks: tuple[ClassicalFieldCoefficient, ...],
    clean: tuple[int, int],
    dirty: tuple[int, ...],
    scratch_bit: int | None,
) -> tuple[Instruction, ...]:
    n = len(targets)
    gates = list(_flip(None, targets[0], masks[0]))
    if n <= 3:
        if n >= 2:
            gates.extend(_flip(None, targets[0], targets[1], masks[1]))
        if n == 3:
            gates.extend(_flip(None, *targets, masks[2]))
        return tuple(gates)
    # The native dirty-prefix schedule emits the second prefix through
    # the first carry, so the direct n == 2 operation is omitted here.
    anc = (*clean, *dirty)
    if scratch_bit is None:
        raise ValueError("Prefix cleanup requires an extra classical result slot")
    outcome = ClassicalFieldCoefficient(scratch_bit)
    for step in range(2):
        for k in reversed(range(n - 4)):
            if step == 0 and k < len(clean):
                continue
            if step == 1 and k + 1 < len(clean):
                gates.extend(measure_x_clear_ir(anc[k + 1], scratch_bit))
                gates.extend(_flip(None, targets[k + 2], anc[k], outcome))
            else:
                gates.extend(_flip(anc[k + 1], targets[k + 2], anc[k]))
        if step == 1:
            gates.extend(measure_x_clear_ir(anc[0], scratch_bit))
            gates.extend(_flip(None, targets[0], targets[1], outcome))
        else:
            gates.extend(_flip(anc[0], targets[0], targets[1]))
            for k in range(min(len(clean), n - 4)):
                gates.extend(_flip(anc[k + 1], targets[k + 2], anc[k]))
        for k in range(len(clean), n - 4):
            gates.extend(_flip(anc[k + 1], targets[k + 2], anc[k]))
        for k in range(n - 3):
            if k >= len(clean) or step == 0:
                gates.extend(_flip(None, masks[k + 1], anc[k]))
        if len(clean) < n - 3 or step == 0:
            gates.extend(_flip(None, masks[n - 2], targets[n - 2], anc[n - 4]))
            gates.extend(
                _flip(None, masks[n - 1], targets[n - 1], targets[n - 2], anc[n - 4])
            )
    return tuple(gates)


def _low_space_core(
    lhs: tuple[int, ...],
    rhs: tuple[bool | ClassicalFieldCoefficient, ...],
    target: int | None,
    clean: tuple[int, int, int],
    bits: tuple[int, ...],
    or_equal: int | bool | ClassicalFieldCoefficient,
    passes: tuple[tuple[int, int], ...],
) -> tuple[Instruction, ...]:
    n = len(lhs)
    if n == 0:
        return _flip(target, or_equal)
    if n == 1:
        mix = _flip(lhs[0], rhs[0])
        # The parity operands are classical unless the equality input is a qubit.

        return (
            *mix,
            *_parity_controlled_flip(target, rhs[0], or_equal, lhs[0]),
            *mix,
            *_flip(target, or_equal),
        )
    a, b, c = clean
    invert = tuple(Instruction("x", (q,)) for q in lhs)
    if n == 2:

        outcome = ClassicalFieldCoefficient(bits[0])
        return (
            *invert,
            *_flip(lhs[0], or_equal),
            *_parity_controlled_flip(a, or_equal, rhs[0], lhs[0]),
            *_flip(a, or_equal),
            *_flip(lhs[1], a),
            *_flip(a, rhs[1]),
            *_flip(target, lhs[1], a),
            *_flip(a, rhs[1]),
            *_flip(lhs[1], a),
            *_flip(target, a),
            *measure_x_clear_ir(a, bits[0]),
            *_flip(None, lhs[0], rhs[0], outcome),
            *_flip(None, lhs[0], or_equal, outcome),
            *_flip(None, or_equal, outcome),
            *_flip(lhs[0], or_equal),
            *invert,
        )

    difference = tuple(
        gate
        for q, coefficient in zip(lhs, rhs, strict=True)
        for gate in _flip(q, coefficient)
    )
    gates = [
        *invert,
        *difference,
        *_flip(target, rhs[-1]),
        *_parity_controlled_flip(target, rhs[-1], rhs[-2], lhs[-1]),
        *_flip(a, lhs[-1], lhs[-2]),
        *_flip(c, a),
    ]

    for depth, (offset, run) in enumerate(passes):
        qa, qb = (b, a) if depth % 2 == 0 else (a, b)
        gates.extend(invert[offset + 1 : offset + run + 1])
        for k in range(min(run, offset)):
            gates.extend(
                _flip(lhs[offset + k + 1], lhs[offset - k - 1], lhs[offset + k])
            )
        if depth:
            gates.extend(_flip(qb, qa, lhs[n - depth]))
            gates.extend(measure_x_clear_ir(qa, bits[depth - 1]))
        if depth == len(passes) - 1:
            gates.extend(_flip(target, or_equal, qb, lhs[2 * offset]))

        gates.extend(_flip(qa, rhs[offset + 1]))
        gates.extend(_flip(qa, rhs[offset]))
        for k in range(min(run, offset)):
            gates.extend(_flip(qa, lhs[offset + k], rhs[offset - k]))
            gates.extend(_flip(qa, lhs[offset + k], rhs[offset - k - 1]))
        if offset <= run:
            gates.extend(_flip(qa, lhs[2 * offset], rhs[0]))
        gates.extend(_flip(target, qa, qb))
        gates.extend(measure_x_clear_ir(qa, bits[depth]))
        outcome = ClassicalFieldCoefficient(bits[depth])
        gates.extend(_flip(None, rhs[offset + 1], outcome))
        gates.extend(_flip(None, rhs[offset], outcome))
        for k in range(min(run, offset)):
            gates.extend(_flip(None, lhs[offset + k], rhs[offset - k], outcome))
            gates.extend(_flip(None, lhs[offset + k], rhs[offset - k - 1], outcome))
        if offset <= run:
            gates.extend(_flip(None, lhs[2 * offset], rhs[0], outcome))

    depth = len(passes)
    gates.extend(measure_x_clear_ir(b if depth % 2 == 0 else a, bits[depth - 1]))
    masks = tuple(ClassicalFieldCoefficient(bit) for bit in bits[:depth])
    gates.extend(
        _prefix_phase(
            (c, *reversed(lhs[n - depth + 1 :])),
            masks,
            (a, b),
            lhs[: max(0, depth - 3)],
            bits[depth + 1] if depth >= 4 else None,
        )
    )
    gates.extend(measure_x_clear_ir(c, bits[depth]))
    for offset, run in reversed(passes):
        for k in reversed(range(min(offset, run))):
            gates.extend(
                _flip(lhs[offset + k + 1], lhs[offset - k - 1], lhs[offset + k])
            )
        gates.extend(invert[offset + 1 : offset + run + 1])
    gates.extend(_flip(None, lhs[-1], lhs[-2], ClassicalFieldCoefficient(bits[depth])))
    return (*gates, *invert, *difference)


def kickmix_low_space_less_than_ir(
    left: tuple[int, ...],
    right: tuple[bool | ClassicalFieldCoefficient, ...],
    target: int | None,
    workspace: tuple[int, ...],
    *,
    measurement_bits: tuple[int, ...] = (),
    or_equal: int | bool | ClassicalFieldCoefficient = False,
    inverted: int | bool | ClassicalFieldCoefficient = False,
    control: int | bool = True,
) -> tuple[Instruction, ...]:
    """Emit the locked three-clean-wire comparison for a classical right word.

    Big-endian right words may be shorter and are padded with leading zeros.
    A quantum control appends a high comparison bit as in the source.
    Measurement slots are distinct from all caller-owned input slots.
    """
    if len(right) > len(left) or any(
        not isinstance(value, (bool, ClassicalFieldCoefficient)) for value in right
    ):
        raise ValueError(
            "Low-space classical comparison requires a fitting classical right word"
        )
    if len(workspace) < 3:
        raise ValueError("Low-space classical comparison requires three clean wires")
    flags = tuple(q for q in (or_equal, inverted, control) if type(q) is int)
    _validate_registers(left, () if target is None else (target,), workspace, flags)
    classical_inputs = {
        q.slot
        for q in (*right, or_equal, inverted)
        if isinstance(q, ClassicalFieldCoefficient)
    }
    if len(set(measurement_bits)) != len(measurement_bits) or any(
        type(bit) is not int or bit < 0 or bit in classical_inputs
        for bit in measurement_bits
    ):
        raise ValueError("Comparison requires distinct nonnegative result slots")
    if control is False:
        return ()
    lhs = left[::-1]
    if type(control) is int:
        lhs = (*lhs, control)
    rhs = (*right[::-1], *((False,) * (len(lhs) - len(right))))
    n = len(lhs)
    passes = _comparison_passes(n)
    depth = len(passes)
    required = depth + 1 + int(depth >= 4) if depth else int(n == 2)
    if len(measurement_bits) < required:
        raise ValueError(
            f"Low-space classical comparison requires {required} result slots"
        )

    initial = _flip(target, inverted, control)
    enable = (Instruction("x", (control,)),) if type(control) is int else ()
    return (
        *initial,
        *enable,
        *_low_space_core(
            lhs,
            rhs,
            target,
            (workspace[0], workspace[1], workspace[2]),
            measurement_bits,
            or_equal,
            passes,
        ),
        *enable,
    )
