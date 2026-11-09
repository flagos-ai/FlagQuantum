"""Exact preparation of a bounded triangular register pair."""

from math import acos, pi, sin, sqrt

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..comparison.less_equal import less_equal_ir, less_equal_requirements
from ..comparison.less_than import (
    less_than_constant_ir,
    less_than_constant_workspace_size,
)
from ..control.multi_control import multi_control_z_ir
from .prepare_reflection import prepare_reflection_ir


def triangular_uniform_requirements(extent: int) -> tuple[int, int]:
    """Return reusable scratch and feedback slots for 0 <= p <= q < extent."""
    if type(extent) is not int or extent < 1:
        raise ValueError("Triangular extent must be a positive integer")
    width = (extent - 1).bit_length()
    if not width:
        return 0, 0
    order_work, order_slots = less_equal_requirements(width, width)
    bound_work = less_than_constant_workspace_size(width, extent)
    return max(2 + max(order_work, bound_work, 1), 2 * width + 1), max(
        order_slots, 2 * width
    )


def triangular_uniform_ir(
    extent: int,
    p: tuple[int, ...],
    q: tuple[int, ...],
    rotation: int,
    workspace: tuple[int, ...],
    slots: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Prepare positive uniform amplitudes; the rotation and scratch end at zero."""
    width = (extent - 1).bit_length()
    work, bits = triangular_uniform_requirements(extent)
    if (
        len(p) != width
        or len(q) != width
        or len(workspace) != work
        or len(slots) != bits
    ):
        raise ValueError("Triangular preparation requires matching registers and work")
    _validate_wires((*p, *q, rotation, *workspace))
    if slots:
        _validate_wires(slots)
    if type(adjoint) is not bool:
        raise ValueError("Triangular preparation direction must be boolean")
    if not width:
        return ()
    bound, ordered, *scratch = workspace
    bound_work = less_than_constant_workspace_size(width, extent)
    order_work, order_slots = less_equal_requirements(width, width)
    bound_check = less_than_constant_ir(
        q,
        extent,
        bound,
        tuple(scratch[:bound_work]),
        slots[:width] if bound_work else (),
    )
    order_check = less_equal_ir(
        p, q, ordered, tuple(scratch[:order_work]), slots[:order_slots]
    )
    marked = (
        *bound_check,
        *order_check,
        Instruction("x", (rotation,)),
        *multi_control_z_ir(
            (bound, ordered), (1, 1), rotation, tuple(scratch[:1]), slots[:1]
        ),
        Instruction("x", (rotation,)),
        *order_check,
        *bound_check,
    )
    domain = extent * (extent + 1) // 2
    capacity = 1 << (2 * width)
    rounds = 1 if 4 * domain >= capacity else 2
    theta = 2 * acos(min(1.0, sin(pi / (4 * rounds + 2)) * sqrt(capacity / domain)))
    address = (*p, *q)
    hadamards = tuple(Instruction("h", (wire,)) for wire in address)
    initialize = (*hadamards, Instruction("ry", (rotation,), params={"theta": theta}))
    uninitialize = (
        Instruction("ry", (rotation,), params={"theta": -theta}),
        *reversed(hadamards),
    )
    zero = prepare_reflection_ir(
        (*address, rotation),
        workspace[: 2 * width + 1],
        slots[: 2 * width],
        global_phase=-1,
        adjoint=adjoint,
    )
    if adjoint:
        return (*((*uninitialize, *zero, *initialize, *marked) * rounds), *uninitialize)
    return (*initialize, *((*marked, *uninitialize, *zero, *initialize) * rounds))
