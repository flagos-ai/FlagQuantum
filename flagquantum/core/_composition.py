"""One instruction rewritten by one construction-time composition member.

:meth:`flagquantum.Circuit.adjoint`, :meth:`~flagquantum.Circuit.power`, and
:meth:`~flagquantum.Circuit.control` each apply one composition member to a program one
instruction at a time, and each has to answer the same question about every instruction it
meets: is there a rewrite the IR can express? This module is that answer for the three of
them, in one place, so the three refusal vocabularies can be read side by side instead of
being spread through the circuit type.

The rewrites themselves are declarations rather than opinions. :data:`ADJOINT_RULES`,
:data:`POWER_RULES`, and :data:`CONTROL_RULES` in
:mod:`flagquantum.core.operator_schema` say which opcode has which form, and
:mod:`flagquantum.core.controlled` builds the controlled expansion. What is left here is
the per-instruction adapter: read the instruction, pick the route its own record selects
-- a carried matrix, a channel, a dynamic operation, a conditioned gate, execution
metadata -- and refuse by name when no route applies.

Refusing is the whole point of the module. Every branch that cannot produce an
instruction sequence raises :class:`flagquantum.errors.CapabilityError` with the reason it
met, because a program builder cannot check the precondition that would make a lossy
rewrite safe: a user-supplied matrix is what executes rather than the opcode's own
declaration, a channel is not a unitary, and an instruction's metadata is behaviour the
rewrite would drop. A silent approximation would change the program while the instruction
listing still looked plausible.

The module is private to :mod:`flagquantum.core` and is not part of the exported Core
surface: the three methods on :class:`flagquantum.Circuit` are the public statement of the
same behaviour, and this module must not grow a second one. It is backend-neutral in the
same sense :mod:`flagquantum.core.controlled` is -- it reads the schema and emits
instructions, never a matrix -- so what a program builder records, what the IR validates,
and what any executor runs are the same instructions.
"""

from __future__ import annotations

from ..errors import CapabilityError
from .controlled import controlled_instructions
from .ir import Instruction
from .operator_schema import get_operator_schema, inverse_operator, power_operator


def inverted_instruction(instruction: Instruction) -> Instruction:
    """Return the single gate that undoes one recorded instruction.

    The matrix recorded on an instruction is what the simulator executes, so it decides
    the inverse whenever it is present: an operation with an explicit matrix is inverted
    by conjugate transpose, which is the only rule that inverts a user-supplied unitary.
    Opcode rules apply to the remaining instructions, and an instruction that neither
    route can invert is refused instead of being copied forward unchanged.

    ``CapabilityError`` is the class the errors-module boundary reserves for a capability
    that is absent rather than a value that is wrong, which is the case here: the program
    is well formed, and undoing it is the part that is not available.
    """

    reason: str | None = None
    if instruction.metadata.get("is_channel"):
        reason = "it is a noise channel, which has no unitary inverse"
    elif instruction.metadata.get("is_dynamic"):
        reason = "it is a dynamic operation, which has no fixed inverse"
    elif instruction.metadata.get("conditions"):
        reason = "it is classically conditioned, which has no fixed inverse"

    if reason is None:
        matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
        if matrix is not None:
            inverted_matrix = getattr(matrix, "mH", None)
            if inverted_matrix is None:
                reason = "its matrix does not expose a conjugate transpose"
            else:
                return Instruction(
                    name=instruction.name,
                    wires=instruction.wires,
                    params=dict(instruction.params),
                    matrix=inverted_matrix,
                    metadata=dict(instruction.metadata),
                )

    if reason is None:
        schema = get_operator_schema(instruction.name)
        if schema is None:
            reason = "it is an unknown opcode with no matrix"
        else:
            inverted = inverse_operator(schema, instruction.params)
            if inverted is None:
                reason = (
                    f"its adjoint declaration {schema.adjoint!r} names no inverse gate"
                )
            else:
                opcode, params = inverted
                return Instruction(
                    name=opcode,
                    wires=instruction.wires,
                    params=params,
                    metadata=dict(instruction.metadata),
                )

    raise CapabilityError(
        f"Cannot invert instruction {instruction.name!r} on qubits "
        f"{instruction.wires}: {reason}."
    )


def powered_instructions(instruction: Instruction, *, count: int) -> list[Instruction]:
    """Return how the power of a one-instruction program is written.

    The only rewrite a power performs is the angle scaling declared by the opcode: a
    gate whose declaration is the exponential of one parameter has a one-instruction
    ``count``-fold action. Everything else is written out as ``count`` copies, which is
    exact for every instruction and needs no property of the gate.

    A carried matrix suppresses the rewrite, exactly as it decides the inverse in
    :func:`inverted_instruction`: a user-supplied unitary is repeated, because
    scaling its angles would apply the opcode's declaration rather than the matrix that
    actually executes.

    The caller must only reach this for a program of exactly one instruction. For a
    longer program the copies of this gate are separated by the other gates and do not
    compose into one instruction, so :meth:`Circuit.power` repeats the program instead.
    """

    matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
    schema = get_operator_schema(instruction.name)
    if matrix is None and schema is not None:
        rewritten = power_operator(schema, instruction.params, count)
        if rewritten is not None:
            opcode, params = rewritten
            return [
                Instruction(
                    name=opcode,
                    wires=instruction.wires,
                    params=dict(params),
                    metadata=dict(instruction.metadata),
                )
            ]

    # A fresh instruction per copy, so the powered program holds distinct records
    # rather than one object repeated: a later edit or an identity comparison must see
    # the copies the program is actually made of.
    return [copy_instruction(instruction) for _ in range(count)]


def copy_instruction(instruction: Instruction) -> Instruction:
    """Return an independent record of one instruction.

    A power writes the same operation more than once, and appending the same
    :class:`Instruction` object twice would make the program's length a claim about a
    list rather than about a program. ``Instruction`` is frozen, so a copy differs from
    its source only in identity -- which is exactly what is wanted here, and what keeps
    a later in-place edit of one copy from reaching the others.
    """

    return Instruction(
        name=instruction.name,
        wires=instruction.wires,
        params=dict(instruction.params),
        matrix=instruction.matrix,
        metadata=dict(instruction.metadata),
    )


def controlled_instruction(
    instruction: Instruction, controls: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Return the gates that replace one instruction once ``controls`` are added.

    The receiver is expanded one instruction at a time, because a control on one qubit
    commutes with conjugating that same qubit by another single-qubit gate: turning each
    gate into its controlled form and appending those forms in the receiver's own order
    is the controlled program. One rule per opcode stays in
    :mod:`flagquantum.core.controlled`, which is where the expansion itself belongs.

    An instruction carrying its own matrix, a noise channel, and a dynamic or
    classically conditioned operation are all refused rather than expanded, because none
    of them has an opcode whose controlled form the IR describes: the matrix is what
    executes, and controlling a recorded matrix would need a second matrix the IR has no
    field for. The three named reasons are the ones :func:`inverted_instruction`
    reports for the same instructions, and any remaining execution metadata is refused
    too, so that nothing an instruction carries is quietly dropped.

    Raises:
        CapabilityError: If the instruction has no controlled form, with the reason the
            refusal met.
    """

    reason: str | None = None
    if instruction.metadata.get("is_channel"):
        reason = "it is a noise channel, which has no controlled form"
    elif instruction.metadata.get("is_dynamic"):
        reason = "it is a dynamic operation, which has no controlled form"
    elif instruction.metadata.get("conditions"):
        reason = "it is classically conditioned, which has no controlled form"
    elif getattr(instruction.matrix, "tensor", instruction.matrix) is not None:
        reason = "it carries its own matrix, which no opcode describes controlling"

    if reason is None:
        schema = get_operator_schema(instruction.name)
        # An opcode the registry does not know and an opcode whose declaration names no
        # controlled form are the same absence reported the same way: in both cases no
        # rule exists to build the controlled program from. The two are also unreachable
        # from a well-formed program for the same reason -- the first must carry a matrix
        # to exist at all, which the branch above already refused, and the second is
        # excluded by the opcode census this contract records.
        expansion = (
            controlled_instructions(
                schema, instruction.params, instruction.wires, controls
            )
            if schema is not None
            else None
        )
        if expansion is None:
            reason = "no controlled form is declared for it"
        elif instruction.metadata:
            reason = "it carries execution metadata a controlled form would drop"
        else:
            return expansion

    raise CapabilityError(
        f"Cannot control instruction {instruction.name!r} on qubits "
        f"{instruction.wires}: {reason}."
    )
