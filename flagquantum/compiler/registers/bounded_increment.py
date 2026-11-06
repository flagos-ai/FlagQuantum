"""Increment a bounded binary word while preserving its padding states."""

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost, CostCall
from .arithmetic import add_constant_adjoint_ir, add_constant_ir
from .arithmetic_resource_model import source_add_cost
from .bitwise import _validate_wires, controlled_xor_constant_ir
from .comparison import less_than_constant_ir
from .controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from .multi_control import multi_and_ir, multi_control_x_ir


def bounded_increment_requirements(
    width: int, modulus: int, controls: int = 0
) -> tuple[int, int]:
    """Return clean work and reusable feedback slots for either direction."""
    if (
        type(width) is not int
        or width < 0
        or type(modulus) is not int
        or not 1 <= modulus <= 1 << width
        or type(controls) is not int
        or controls < 0
    ):
        raise ValueError("Bounded increment needs a fitting positive modulus")
    if modulus == 1:
        return 0, 0
    if modulus == 1 << width:
        predicate = max(0, controls - 1)
        return predicate + 2 * width - 1, max(predicate, width - 1)
    return 1 + max(width + 1, controls + max(2 * width - 1, width + 1)), max(
        controls, width
    )


def bounded_increment_ir(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    controls: tuple[int, ...] = (),
    control_values: tuple[int, ...] = (),
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Lower the unit-step ModAddK permutation on a big-endian register.

    Values at or above the modulus are fixed, including on active controls.
    Work starts and ends at zero. The same feedback slots can be reused
    after each comparator, arithmetic carry or predicate has cleared.
    """
    width = len(target)
    work, slots = bounded_increment_requirements(width, modulus, len(controls))
    if controls or target or workspace:
        _validate_wires((*controls, *target, *workspace))
    if classical_bits:
        _validate_wires(classical_bits)
    if (
        len(workspace) != work
        or len(classical_bits) != slots
        or len(controls) != len(control_values)
        or any(type(bit) is not int or bit not in (0, 1) for bit in control_values)
        or type(adjoint) is not bool
    ):
        raise ValueError("Bounded increment requires matching work, slots and controls")
    if modulus == 1:
        return ()

    add = add_constant_adjoint_ir if adjoint else add_constant_ir
    cadd = controlled_add_constant_adjoint_ir if adjoint else controlled_add_constant_ir
    if modulus == 1 << width and not controls:
        return add(target, 1, workspace[:width], workspace[width:], classical_bits)
    if modulus == 1 << width and len(controls) == 1:
        return cadd(
            controls[0],
            target,
            1,
            workspace[:width],
            workspace[width:],
            classical_bits,
            control_value=control_values[0],
        )

    membership: tuple[Instruction, ...] = ()
    if modulus < 1 << width:
        valid = workspace[0]
        scratch = workspace[1:]
        membership = less_than_constant_ir(
            target, modulus, valid, scratch[: width + 1], classical_bits[:width]
        )
        controls = (*controls, valid)
        control_values = (*control_values, 1)
    else:
        scratch = workspace

    prefix = len(controls) - 1
    predicate, scratch = scratch[:prefix], scratch[prefix:]
    compute = multi_and_ir(controls, control_values, predicate) if prefix else ()
    erase = (
        multi_and_ir(
            controls,
            control_values,
            predicate,
            classical_bits=classical_bits[:prefix],
            uncompute=True,
        )
        if prefix
        else ()
    )
    enable = predicate[-1] if prefix else controls[0]
    ring = cadd(
        enable,
        target,
        1,
        scratch[:width],
        scratch[width : 2 * width - 1],
        classical_bits[: width - 1],
    )
    wrap: tuple[Instruction, ...] = ()
    if membership:
        marker, *carry = scratch[: width + 1]
        flags = (enable, *target)
        at_modulus = (1, *(modulus >> i & 1 for i in reversed(range(width))))
        at_zero = (1,) + (0,) * width
        before, after = (at_zero, at_modulus) if adjoint else (at_modulus, at_zero)
        wrap = (
            *multi_control_x_ir(
                flags, before, marker, tuple(carry), classical_bits[:width]
            ),
            *controlled_xor_constant_ir(marker, target, modulus),
            *multi_control_x_ir(
                flags, after, marker, tuple(carry), classical_bits[:width]
            ),
        )
    shift = (*wrap, *ring) if adjoint else (*ring, *wrap)
    return (*membership, *compute, *shift, *erase, *membership)


def source_bounded_increment_cost(width: int, controls: int = 0) -> ComponentCost:
    """Keep the locked ModAddK declaration of five same-width additions."""
    if type(controls) is not int or controls < 0:
        raise ValueError("Bounded increment control count must be nonnegative")
    add = source_add_cost(width, width)
    return ComponentCost("ModAddK", width + controls, calls=(CostCall(add, 5),))
