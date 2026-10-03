"""Cancel the two members of a declared inverse pair adjacent on their wires.

`pipeline.merge_self_inverse` removes a run of one self-inverse opcode and
`pipeline.merge_adjacent_rotations` merges a run of one rotation. Neither can see
the shape that is left over, because it repeats no opcode:

    s(0)  sdg(0)

is the identity, and ``optimize`` keeps both, since `sdg` is not a member of
`_SELF_INVERSE` and no pass merges two different opcodes. `t`/`tdg` and
`sx`/`sxdg` are the same case.

This pass reads its rule from the schema's existing `adjoint` declaration rather
than from a table of its own. `OperatorSchema.adjoint` already names the inverse
of every unitary gate, `operator_schema.inverse_operator` already turns that
declaration into an opcode and a parameter mapping, and
`flagquantum.circuit.Circuit.adjoint` is already its consumer. A second table
here would be a second source of truth for one fact, free to drift from the
first; instead this pass reads the declaration the rest of the product reads.

Three properties make a cancellation admissible.

**Only a declared partner is cancelled.** `inverse_operator` answers a
self-inverse gate with that same opcode, and that shape belongs to
`merge_self_inverse`, so this pass requires the partner to be a *different*
opcode. Its candidate set is therefore disjoint from `_SELF_INVERSE`, which makes
it an extension of the existing adjacency passes rather than a second opinion
about them.

**No parameter value is read.** Both members must be free of parameters and of a
caller-supplied matrix. Every gate whose `adjoint` is `negate_parameters` or one
of the two angle rules is therefore declined: its product is the identity only at
one specific angle, and `merge_adjacent_rotations` is the pass that reads the
angle. Nothing here guesses an angle, so nothing here can be wrong about one.

**The pair must be adjacent on the wire it acts on.** A member is removed only
when the latest surviving instruction touching its wire is its declared partner.
Every instruction between the two either does not touch that wire, and so
commutes with both by disjointness, or was itself removed by a proven pair
earlier in the same scan. Either way what lies between the pair is the identity,
which is what makes the two gates multiply to `I`. An odd or mismatched count
leaves its last member where it was, so what survives a run is applied as late as
it was before.

Cancelling across a gap that only *commutes* rather than being empty is a strictly
harder question, and this pass does not attempt it: it reads the inverse
declaration and nothing else, so `s(0) cx(0, 1) sdg(0)` is left alone even though
the pair would be removable there. That is an owned, measured gap rather than an
oversight. `benchmarks/compiler_inverse_cancellation.py` drives every pair through
ten gaps and reports, for each row, whether the gap's operator commutes with the
pair and whether the pair was in fact removable, which separates the refusals that
had to happen from the reach this pass leaves on the table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import (
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
    inverse_operator,
)
from .pipeline import _WireLocalProgram


def _declared_partner(opcode: str) -> str | None:
    """Return the opcode the schema declares as the inverse of ``opcode``.

    ``None`` means the opcode is outside this pass's scope: it is unknown, it is
    not unitary, it takes parameters, its declaration names no invertible gate, or
    the declaration names the opcode itself and the shape therefore belongs to
    `merge_self_inverse`.
    """

    schema = get_operator_schema(opcode)
    if schema is None or not schema.unitary or schema.parameters:
        return None
    inverse = inverse_operator(schema, {})
    if inverse is None:
        return None
    partner, _ = inverse
    return None if partner == schema.opcode else partner


def inverse_pairs() -> Mapping[str, str]:
    """Return every opcode this pass can cancel, mapped to its declared partner.

    Derived from the schema on each call rather than pinned in a table here, so
    that adding an inversion declaration to `operator_schema` widens this pass
    without an edit to it. Exposed because a reach report has to name the opcodes
    its rule covers, and a test can only hold that to a number it can ask for.
    """

    pairs: dict[str, str] = {}
    for schema in OPERATOR_SCHEMAS.values():
        partner = _declared_partner(schema.opcode)
        if partner is not None:
            pairs[schema.opcode] = partner
    return MappingProxyType(pairs)


def _is_cancellable(instruction: Instruction, partner: str) -> bool:
    """Whether ``instruction`` is the cancellable ``partner`` half of a pair."""

    return (
        instruction.matrix is None
        and not instruction.params
        and canonical_opcode(instruction.name) == partner
    )


def merge_inverse_pairs(ir: CircuitIR) -> CircuitIR:
    """Remove declared inverse pairs adjacent on the wire they act on.

    Returns ``ir`` itself when no pair is adjacent, so a caller can tell "nothing to
    do" from "something changed" without diffing, and the pass stays cheap on a
    circuit whose shape it has nothing to say about.
    """

    program = _WireLocalProgram()
    removed = 0
    for instruction in ir:
        partner = _declared_partner(instruction.name)
        position = program.last_touching(instruction.wires)
        previous = None if position is None else program[position]
        if (
            partner is not None
            and position is not None
            and previous is not None
            and previous.wires == instruction.wires
            and _is_cancellable(previous, partner)
        ):
            program.pop(position)
            removed += 1
        else:
            program.append(instruction)
    if not removed:
        return ir
    return replace(ir, instructions=tuple(program))


__all__ = ["inverse_pairs", "merge_inverse_pairs"]
