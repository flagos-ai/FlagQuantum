"""Coherent rank-conditioned amplitude amplification for a bounded index."""

from math import acos, ldexp, sqrt, tau

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..comparison.less_equal import less_equal_ir, less_equal_requirements
from ..control.multi_control import multi_control_x_ir
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .prepare_reflection import prepare_reflection_ir


def rank_rotation_word(rank: int, bits: int) -> int:
    """Round the one-step amplification angle to a binary fraction of a turn."""
    if type(rank) is not int or rank < 1 or type(bits) is not int or bits < 1:
        raise ValueError("Rank and rotation precision must be positive integers")
    capacity = 1 << (rank - 1).bit_length()
    angle = 2 * acos(sqrt(capacity / (4 * rank)))
    numerator, denominator = (angle / tau).as_integer_ratio()
    return ((numerator << (bits + 1)) + denominator) // (2 * denominator)


def rank_uniform_requirements(width: int) -> tuple[int, int]:
    """Return scratch and reusable feedback slots for an index of this width."""
    if type(width) is not int or width < 0:
        raise ValueError("Rank register width must be a nonnegative integer")
    compare_work, compare_slots = (
        less_equal_requirements(width, width) if width else (0, 0)
    )
    return max(0, width - 1) + max(2, 1 + compare_work, width + 1), max(
        1, compare_slots, width
    )


def rank_uniform_ir(
    rank_minus_one: tuple[int, ...],
    angle_word: tuple[int, ...],
    index: tuple[int, ...],
    rotation: int,
    success: int,
    workspace: tuple[int, ...],
    slots: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply the Appendix C rank preparation or its adjoint.

    Rank and angle words are read-only and big-endian. With zero signal input,
    the success flag is set exactly on valid indices with rotation zero.
    Angle quantization can leave a coherent failure component. The rotation
    and success wires are logical outputs, while workspace always clears.
    """
    width = len(index)
    work, bits = rank_uniform_requirements(width)
    if (
        len(rank_minus_one) != width
        or not angle_word
        or len(workspace) != work
        or len(slots) != bits
    ):
        raise ValueError("Rank preparation requires matching registers, angle and work")
    if type(adjoint) is not bool:
        raise ValueError("Rank preparation direction must be boolean")
    _validate_wires(
        (*rank_minus_one, *angle_word, *index, rotation, success, *workspace)
    )
    _validate_wires(slots)
    prefix_size = max(0, width - 1)
    prefix, scratch = workspace[:prefix_size], workspace[prefix_size:]
    activity = (*rank_minus_one[:1], *prefix)
    prefix_inputs = tuple(
        zip(activity[:prefix_size], rank_minus_one[1:], prefix, strict=True)
    )
    compute = tuple(
        gate
        for a, b, target in prefix_inputs
        for gate in (
            *and_compute_ir(a, b, target, values=(0, 0)),
            Instruction("x", (target,)),
        )
    )
    erase = tuple(
        gate
        for (a, b, target), slot in reversed(
            tuple(zip(prefix_inputs, slots[:prefix_size], strict=True))
        )
        for gate in (
            Instruction("x", (target,)),
            *and_uncompute_ir(a, b, target, slot, values=(0, 0)),
        )
    )

    hadamards = tuple(
        Instruction("ch", (bit, wire))
        for bit, wire in zip(activity, index, strict=True)
    )
    angles = tuple(ldexp(tau, -i - 1) for i in range(len(angle_word)))
    rotate = tuple(
        Instruction("cry", (control, rotation), params={"theta": angle})
        for control, angle in zip(angle_word, angles, strict=True)
    )
    unrotate = tuple(
        Instruction("cry", (control, rotation), params={"theta": -angle})
        for control, angle in reversed(tuple(zip(angle_word, angles, strict=True)))
    )
    prepare = (*hadamards, *rotate)
    unprepare = (*unrotate, *reversed(hadamards))
    valid = scratch[0]
    compare_work, compare_slots = (
        less_equal_requirements(width, width) if width else (0, 0)
    )
    bound = (
        less_equal_ir(
            index,
            rank_minus_one,
            valid,
            scratch[1 : 1 + compare_work],
            slots[:compare_slots],
        )
        if width
        else (Instruction("x", (valid,)),)
    )
    flip = Instruction("x", (rotation,))
    mark = (*bound, flip, Instruction("cz", (valid, rotation)), flip, *bound)
    record = (
        *bound,
        *multi_control_x_ir(
            (valid, rotation), (1, 0), success, scratch[1:2], slots[:1]
        ),
        *bound,
    )
    zero = prepare_reflection_ir(
        (*index, rotation),
        scratch[: width + 1],
        slots[:width],
        global_phase=-1,
        adjoint=adjoint,
    )
    body = (
        (*record, *unprepare, *zero, *prepare, *mark, *unprepare)
        if adjoint
        else (*prepare, *mark, *unprepare, *zero, *prepare, *record)
    )
    return (*compute, *body, *erase)
