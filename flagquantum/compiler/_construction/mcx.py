"""Exact multi-control X with one proven-clean flag and restored borrowed controls."""

from collections.abc import Sequence

from ...core.ir import Instruction


def _dirty_mcx(
    controls: Sequence[int], target: int, borrowed: Sequence[int]
) -> tuple[Instruction, ...]:
    """Toggle a target with arbitrary borrowed bits, restoring every borrowed bit."""
    count = len(controls)
    if count == 1:
        return (Instruction("cx", (controls[0], target)),)
    if count == 2:
        return (Instruction("ccx", (*controls, target)),)
    top = Instruction("ccx", (controls[-1], borrowed[count - 3], target))
    # The first ladder changes a borrowed bit by the control conjunction.
    # Applying the target before and after cancels the unknown borrowed value.
    ladder = tuple(
        Instruction("ccx", (controls[i], borrowed[i - 2], borrowed[i - 1]))
        for i in range(count - 2, 1, -1)
    )
    bottom = Instruction("ccx", (controls[0], controls[1], borrowed[0]))
    toggle = (*ladder, bottom, *reversed(ladder))
    return (top, *toggle, top, *toggle)


def one_clean_mcx(
    controls: tuple[int, ...], target: int, flag: int
) -> tuple[Instruction, ...]:
    """Lower an all-ones X using one zero flag; other bits may be entangled.

    Split controls so each half can borrow the other half for a dirty ladder.
    Compute the first conjunction into the clean flag, toggle the target using
    that flag and the remaining controls, then uncompute the flag. Borrowed
    controls, target and flag must be distinct. No other storage is accessed.
    """
    if len(controls) < 3 or len(set((*controls, target, flag))) != len(controls) + 2:
        raise ValueError(
            "one-clean MCX requires at least three distinct controls and disjoint target/flag"
        )
    split = (len(controls) + 1) // 2
    first, second = controls[:split], controls[split:]
    compute = _dirty_mcx(first, flag, second)
    toggle = _dirty_mcx((flag, *second), target, first)
    return (*compute, *toggle, *reversed(compute))
