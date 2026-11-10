"""Validate concrete quantum widths and external CBit values at physical entry."""

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.register_verifier import (
    executable_register_width,
    register_port_types,
)
from ...core.ir.program.registers import RegisterSide


def entry_input_layout(
    function: FunctionRecord, classical_inputs: tuple[int | None, ...]
) -> tuple[tuple[int, ...], tuple[int | None, ...]]:
    """Resolve flattened physical widths and explicit per-element initial values.

    Classical positions require a built-in bit value. Quantum positions use None,
    including zero-width words. Empty classical arrays have no input elements.
    """
    dtypes = (
        register_port_types(function, RegisterSide.LEFT)
        if function.registers is not None
        else (quantum_bit_type(),) * len(function.blocks[0].arguments)
    )
    classical = tuple(dtype == classical_bit_type() for dtype in dtypes)
    if not isinstance(classical_inputs, tuple):
        raise ValueError("classical inputs require an immutable tuple of bit values")
    if not classical_inputs:
        if any(classical):
            raise ValueError(
                "unsupported executable register: uninitialized CBit input; "
                "provide classical_inputs in flattened entry order"
            )
        values: tuple[int | None, ...] = (None,) * len(dtypes)
    else:
        if len(classical_inputs) != len(dtypes):
            raise ValueError("classical input values must match all entry elements")
        for is_classical, value in zip(classical, classical_inputs, strict=True):
            if is_classical:
                if type(value) not in (int, bool) or value not in (0, 1):
                    raise ValueError(
                        "external CBit inputs require built-in zero/one bits"
                    )
            elif value is not None:
                raise ValueError(
                    "quantum input positions require None classical values"
                )
        values = classical_inputs
    widths = tuple(
        0 if is_classical else executable_register_width(dtype)
        for dtype, is_classical in zip(dtypes, classical, strict=True)
    )
    return widths, values
