"""Shared physical replacement configuration for owned constructor libraries."""

from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import ProgramRecord


def bind_quantum_library(
    record: ProgramRecord, inputs: tuple[tuple[int, ...], ...], width: int
) -> BoundRegisterProgram:
    """Bind finite quantum inputs with an idle borrowed spectator and owned pool."""
    return bind_register_program(
        record,
        inputs,
        clean=tuple(range(width + 1, 3 * width + 80)),
        borrowed=(width,),
        input_ownership=tuple((i, False) for i in range(width)),
        allow_workspace_outputs=True,
    )
