"""Validate concrete quantum widths and encoded classical entry values."""

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.register_verifier import (
    executable_classical_register_width,
    executable_register_width,
    register_port_types,
)
from ...core.ir.program.registers import RegisterSide
from .classical_word_storage import classical_word_bits


def validate_executable_registers(function: FunctionRecord) -> None:
    """Check declared physical types even when their arrays have no elements."""
    for register in function.registers or ():
        dtype = register.dtype
        if dtype == classical_bit_type() or dtype.parameters[:1] == (
            ("storage", "classical"),
        ):
            executable_classical_register_width(dtype)
        else:
            executable_register_width(dtype)


def entry_input_layout(
    function: FunctionRecord, classical_inputs: tuple[int | None, ...]
) -> tuple[tuple[int, ...], tuple[tuple[int, ...] | None, ...]]:
    """Resolve flattened physical widths and explicit per-element initial values.

    Classical positions require one built-in stored integer. Quantum positions use None,
    including zero-width words. Empty classical arrays have no input elements.
    """
    dtypes = (
        register_port_types(function, RegisterSide.LEFT)
        if function.registers is not None
        else (quantum_bit_type(),) * len(function.blocks[0].arguments)
    )
    classical = tuple(
        dtype == classical_bit_type()
        or dtype.parameters[:1] == (("storage", "classical"),)
        for dtype in dtypes
    )
    if not isinstance(classical_inputs, tuple):
        raise ValueError("classical inputs require an immutable tuple of bit values")
    if not classical_inputs:
        if any(classical):
            raise ValueError(
                "unsupported executable register: uninitialized CBit input; "
                "provide classical_inputs in flattened entry order"
            )
        values: tuple[tuple[int, ...] | None, ...] = (None,) * len(dtypes)
    else:
        if len(classical_inputs) != len(dtypes):
            raise ValueError("classical input values must match all entry elements")
        encoded: list[tuple[int, ...] | None] = []
        for dtype, is_classical, value in zip(
            dtypes, classical, classical_inputs, strict=True
        ):
            if is_classical:
                if value is None:
                    if dtype == classical_bit_type():
                        raise ValueError(
                            "external CBit inputs require built-in zero/one bits"
                        )
                    raise ValueError("classical entry elements require integer values")
                encoded.append(classical_word_bits(dtype, value))
            elif value is not None:
                raise ValueError(
                    "quantum input positions require None classical values"
                )
            else:
                encoded.append(None)
        values = tuple(encoded)

    widths = tuple(
        0 if is_classical else executable_register_width(dtype)
        for dtype, is_classical in zip(dtypes, classical, strict=True)
    )
    return widths, values
