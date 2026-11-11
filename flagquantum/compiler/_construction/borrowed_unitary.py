"""Checked views of pure helpers on initialized borrowed quantum storage."""

from .._hybrid.model import HybridProgram, Operation, Value
from .._hybrid.unitary_effects import unitary_regions
from .profile import _fail
from .register_profile import _borrowed_program


def unitary_template_parts(
    program: HybridProgram,
) -> tuple[tuple[Value, ...], tuple[Value, ...], tuple[Operation, ...]]:
    """Require only incoming storage and closed reversible scratch in a template.

    The ordinary register verifier separately checks Const, initialization,
    effect order and direct-compute cleanup. The prefix below establishes a
    layout for incoming writable operands, never an allocation at the call site.
    """
    block = program.body.blocks[0]
    readonly = block.arguments[1:]
    writable = tuple(
        op.results[0] for op in block.operations if op.name == "qreg.declare"
    )
    prefix = block.operations[: 2 * len(writable)]
    if (
        not writable
        or len(block.operations) <= len(prefix) + 1
        or any(
            declaration.name != "qreg.declare"
            or declaration.results != (value,)
            or initialization.name != "qreg.allocate"
            or initialization.operands[1:] != (value,)
            for value, declaration, initialization in zip(
                writable, prefix[::2], prefix[1::2], strict=True
            )
        )
        or block.operations[-1].name != "qreg.return"
        or block.operations[-1].operands[1:] != (*readonly, *writable)
    ):
        _fail(
            "register.unitary_call",
            "unitary helpers require initialized borrowed storage",
        )
    effects = block.operations[len(prefix) : -1]
    try:
        unitary_regions(effects)
    except ValueError as error:
        _fail("register.unitary_call", str(error))
    return readonly, writable, effects


def owned_unitary_layout(program: HybridProgram) -> HybridProgram:
    """Give the shared scheduler a layout without reinitializing caller storage."""
    readonly, writable, effects = unitary_template_parts(program)
    return _borrowed_program(
        effects, (), (*readonly, *writable), name=program.name + ".layout"
    )
