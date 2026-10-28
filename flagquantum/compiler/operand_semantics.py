"""What an instruction's operands require of a device, stated once.

Three facts about operands decide what a topology must provide, and each one is
recorded here rather than rediscovered by every pass that needs it:

* a two-qubit opcode is either operand-symmetric -- its unitary is invariant
  under exchanging the two operands, so either direction of a physical link
  satisfies it -- or it is a control/target opcode whose physical edge has to
  run the way the operands are written;
* a three-or-more-qubit opcode interacts over named operand pairs rather than
  over a chain of them, and no router in this package synthesizes one onto
  physical couplings;
* a control/target CX whose physical edge runs the other way is realized by
  conjugating the reversed CX with Hadamards on both qubits.

Both tables are total over the opcodes Core declares: an opcode with no entry is
refused instead of guessed, and tests fail when a new Core opcode arrives without
one. The direction and locality rules live here, apart from the routers that ask
them, because two routers and one cost model ask the same questions and a second
copy of an answer is a second thing to keep true.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.ir import Instruction
from .ordering_barrier import is_ordering_barrier

# Operand semantics of native two-qubit instructions on an ordered physical
# graph. ``False`` marks a control/target opcode whose physical direction must
# match the declared edge; ``True`` marks an opcode whose unitary is invariant
# under exchanging its two operands, so either direction of a physical link
# satisfies it. The table is total over the two-qubit opcodes of
# ``flagquantum.core.operator_schema.OPERATOR_SCHEMAS``; an unlisted opcode is
# refused instead of being guessed, and a test fails when a new two-qubit opcode
# is added to Core without an entry here.
_TWO_WIRE_OPERAND_SYMMETRY: dict[str, bool] = {
    "cx": False,
    "cy": False,
    "crx": False,
    "cry": False,
    "crz": False,
    "cz": True,
    "cphase": True,
    "rxx": True,
    "ryy": True,
    "rzz": True,
    "swap": True,
    "iswap": True,
    "iswapdg": True,
    "ch": False,
    "csx": False,
    "csxdg": False,
    "dcx": False,
    "ecr": False,
}

# The physical couplings a multi-qubit instruction needs, as operand index pairs.
# This is the authoritative vocabulary for multi-qubit locality in the Compiler:
# the pairs are the instruction's real two-qubit interactions, taken from Core's
# operand order (`ccx` is control1/control2/target and conjugates on the target;
# `cswap` is control/target1/target2 and exchanges under the control), so the
# requirement is not that the operands form a chain. The opcodes are Core's
# arity-three operands, and the `tests/team/compiler` locality suite fails when a
# Core arity of three or more has no entry here.
_MULTI_QUBIT_OPERAND_PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "ccx": ((0, 2), (1, 2)),
    "cswap": ((0, 1), (0, 2)),
}


def _operand_symmetry(instruction: Instruction) -> bool | None:
    """Return whether an opcode's two operands may be exchanged.

    ``None`` means the opcode has no recorded operand semantics, so no direction
    rule applies to it. The caller owns that refusal, because the two callers
    refuse in their own vocabulary: legalization names the missing rule, and the
    cost model names the strategy it cannot price.
    """

    return _TWO_WIRE_OPERAND_SYMMETRY.get(instruction.name)


def _reverse_cx_rewrite(
    control: int,
    target: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return one CX reversed onto the opposite edge of the same physical link.

    A CX whose physical edge runs target to control is the same operator as the
    reversed CX conjugated by a Hadamard on both qubits, because
    ``CX(c, t) = (H . H) CX(t, c) (H . H)``. The five instructions are returned
    as ``(name, qubits)`` pairs so each caller builds them with its own metadata
    rather than sharing one instruction shape.
    """

    return (
        ("h", (control,)),
        ("h", (target,)),
        ("cx", (target, control)),
        ("h", (control,)),
        ("h", (target,)),
    )


def _require_multi_qubit_device_local(
    instruction: Instruction,
    qubits: tuple[int, ...],
    has_edge: Callable[[int, int], bool],
) -> None:
    """Fail closed unless a multi-qubit instruction is executable on the device.

    No router in this package synthesizes a three-or-more-qubit instruction onto
    physical couplings, and a routing strategy cannot insert SWAPs for one, so a
    multi-qubit instruction has to be decomposable where it stands. The required
    couplings are read from `_MULTI_QUBIT_OPERAND_PAIRS` rather than inferred from
    the operand order, because the two known multi-qubit instructions do not
    interact on consecutive operands; an instruction with no entry is refused
    instead of assumed local. Two-qubit instructions keep their existing
    SWAP-based handling.

    A barrier is the one multi-qubit instruction this rule does not apply to: it
    carries no unitary and constrains only the order of the operations around it,
    so it needs no coupling and cannot be decomposed onto one. The optimizer, the
    commutation rules, and the schedule legalizer already read it that way, and
    the router is where a directive was being turned into a device operation.
    :func:`~flagquantum.compiler.ordering_barrier.is_ordering_barrier` tells the
    two apart, so an instruction that names the barrier while carrying a matrix
    is a gate wearing the name and keeps the connectivity requirement below.
    """

    if len(qubits) < 3:
        return
    if is_ordering_barrier(instruction):
        return
    required = _MULTI_QUBIT_OPERAND_PAIRS.get(instruction.name)
    if required is None:
        raise ValueError(
            f"multi-qubit instruction {instruction.name!r} has no verified physical "
            f"connectivity rule: physical qubits {qubits}; decompose multi-qubit "
            "instructions onto device couplings before routing"
        )
    missing = tuple(
        (qubits[left], qubits[right])
        for left, right in required
        if not has_edge(qubits[left], qubits[right])
    )
    if missing:
        raise ValueError(
            f"multi-qubit instruction {instruction.name!r} requires physical "
            f"couplings {missing} the device does not carry: physical qubits "
            f"{qubits}; decompose multi-qubit instructions onto device couplings "
            "before routing"
        )


__all__ = (
    "_MULTI_QUBIT_OPERAND_PAIRS",
    "_TWO_WIRE_OPERAND_SYMMETRY",
    "_operand_symmetry",
    "_require_multi_qubit_device_local",
    "_reverse_cx_rewrite",
)
