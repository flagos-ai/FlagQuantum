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
same behaviour, and this module must not grow a second one. It chooses no matrix: what it
emits is either the instructions :mod:`flagquantum.core.controlled` builds from a
declaration, or one instruction carrying the caller's own operand, which
:func:`controlled_operand` wraps rather than replaces. An adjoint of such an operand is
emitted only after the unitary-inverse precondition is checked on the operand's own
device. So what a program builder records, what the IR validates, and what any executor
runs are the same instructions.
"""

from __future__ import annotations

from typing import Any

import torch

from ..errors import CapabilityError
from .classical_control import is_classical_control_instruction
from .controlled import controlled_instructions
from .ir import Instruction
from .operator_schema import get_operator_schema, inverse_operator, power_operator


def _is_finite_unitary(matrix: torch.Tensor) -> bool:
    """Validate a dense operand without detaching its eventual adjoint.

    A conjugate transpose is an inverse only for a unitary. The absolute Gram
    residual tolerance is 32 machine epsilons of the real component dtype;
    relative tolerance is zero. This is a floating-point admission check, not an
    exact or interval certificate. Validation must not enter the autograd graph.
    """
    with torch.no_grad():
        operand = matrix.detach()
        if not (operand.is_floating_point() or operand.is_complex()):
            operand = operand.to(dtype=torch.float64)
        if not bool(torch.isfinite(operand).all()):
            return False
        identity = torch.eye(
            operand.shape[-1], dtype=operand.dtype, device=operand.device
        )
        tolerance = 32 * torch.finfo(operand.real.dtype).eps
        return torch.allclose(
            operand.mH @ operand, identity, atol=tolerance, rtol=0
        ) and torch.allclose(operand @ operand.mH, identity, atol=tolerance, rtol=0)


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
    elif is_classical_control_instruction(instruction):
        reason = "it is classically conditioned, which has no fixed inverse"

    if reason is None:
        matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
        if matrix is not None:
            if not isinstance(matrix, torch.Tensor):
                reason = "its matrix does not expose a conjugate transpose"
            else:
                dimension = 2 ** len(instruction.wires)
                operand = matrix.reshape(dimension, dimension)
                if not _is_finite_unitary(operand):
                    reason = "its matrix is not a finite unitary"
                else:
                    return Instruction(
                        name=instruction.name,
                        wires=instruction.wires,
                        params=dict(instruction.params),
                        matrix=operand.mH,
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
                    wires=(
                        tuple(reversed(instruction.wires))
                        if schema.adjoint == "reverse_operands"
                        else instruction.wires
                    ),
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


def controlled_operand(matrix: Any, n_controls: int) -> Any | None:
    """Wrap one carried operand so it applies only where every control is set.

    The result is the block diagonal ``diag(I, U)``: the identity on the branches where
    at least one control is clear, and ``U`` on the single branch where all of them are
    set. ``U`` lands in the trailing block because an instruction's matrix reads its
    qubits in order with the last one least significant, and the controls are written
    ahead of the operand's own qubits, so they occupy the leading, most significant,
    factors. Reading the two orderings the same way is what makes the wrap a block
    diagonal rather than a permutation.

    The operand is the caller's own matrix, so this is the one matrix the composition
    members do not choose: it is read, sized and wrapped. ``None`` is returned when the
    operand publishes no square two-dimensional shape, which is the case for a nested
    Python sequence and for a shape such as ``(2, 8)`` that holds the right number of
    entries without being a matrix. A wrapper that guessed the shape would be guessing
    which qubit each entry belongs to.
    """

    carried = getattr(matrix, "tensor", matrix)
    shape = getattr(carried, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    side = int(shape[0])
    if side < 2 or side != int(shape[1]) or side & (side - 1):
        return None
    total = (1 << n_controls) * side
    clear = total - side
    wrapped = torch.zeros((total, total), dtype=carried.dtype, device=carried.device)
    wrapped[:clear, :clear].fill_diagonal_(1)
    wrapped[clear:, clear:] = carried
    return wrapped


def _controlled_operand_name(name: str) -> str:
    """The name a controlled operand is recorded under.

    A name the registry declares is a claim that the instruction *is* that gate, and the
    IR enforces the claim by fixing the name's arity: an ``x`` carries one qubit. A
    controlled operand carries more, so it cannot keep a declared name. The operand keeps
    its own name when the registry does not declare it, which is the custom-operation
    route, and otherwise takes ``any`` -- the name ``Circuit.any`` already gives an
    instruction whose only description is its matrix. Renaming is confined to the case
    where the name was already a label over a caller-supplied matrix rather than a
    description of what executes.
    """

    return name if get_operator_schema(name) is None else "any"


def controlled_instruction(
    instruction: Instruction, controls: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Return the gates that replace one instruction once ``controls`` are added.

    The receiver is expanded one instruction at a time, because a control on one qubit
    commutes with conjugating that same qubit by another single-qubit gate: turning each
    gate into its controlled form and appending those forms in the receiver's own order
    is the controlled program. One rule per opcode stays in
    :mod:`flagquantum.core.controlled`, which is where the expansion itself belongs.

    An instruction that carries its own operand has no opcode whose controlled form the
    registry declares, so the operand itself is wrapped: :func:`controlled_operand`
    turns it into the block diagonal that applies it only where every control is set,
    and the wrapped matrix is recorded on one instruction spanning the controls and the
    operand's own qubits. That is the only route for a caller's matrix, because the
    matrix is what executes and no declaration describes it. A noise channel, a dynamic
    operation, and a classically conditioned operation are still refused: each is an
    execution behaviour rather than an operand, and the three named reasons are the ones
    :func:`inverted_instruction` reports for the same instructions. Any remaining
    execution metadata is refused too, so that nothing an instruction carries is quietly
    dropped.

    Raises:
        CapabilityError: If the instruction has no controlled form, with the reason the
            refusal met.
    """

    reason: str | None = None
    if instruction.metadata.get("is_channel"):
        reason = "it is a noise channel, which has no controlled form"
    elif instruction.metadata.get("is_dynamic"):
        reason = "it is a dynamic operation, which has no controlled form"
    elif is_classical_control_instruction(instruction):
        reason = "it is classically conditioned, which has no controlled form"
    elif getattr(instruction.matrix, "tensor", instruction.matrix) is not None:
        wrapped = controlled_operand(instruction.matrix, len(controls))
        if wrapped is None:
            reason = "its matrix exposes no square operand a control can wrap around"
        elif instruction.metadata:
            reason = "it carries execution metadata a controlled form would drop"
        else:
            return (
                Instruction(
                    name=_controlled_operand_name(instruction.name),
                    wires=(*controls, *instruction.wires),
                    params=dict(instruction.params),
                    matrix=wrapped,
                ),
            )

    if reason is None:
        schema = get_operator_schema(instruction.name)
        # An opcode the registry does not know and an opcode whose declaration names no
        # controlled form are the same absence reported the same way: in both cases no
        # rule exists to build the controlled program from. The first is unreachable
        # from a well-formed program, because an instruction with an unknown opcode must
        # carry a matrix to exist at all and that route is above; the second is excluded
        # by the opcode census this contract records.
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
